"""文档第5.6、5.7和第6章对应的站星规划、实时调度与站点评估。"""
from __future__ import annotations
import csv, heapq, math, time
from collections import deque
from dataclasses import dataclass, field
from typing import Sequence
from .rain_model import link_rain_state, scenario_attenuation_db, scenario_spatial_factor

EARTH_RADIUS_M=6_378_135.0
C_KM_PER_MS=299.792458
GATEWAY_CHANNEL_CAPACITY_GBPS=2.0

@dataclass(frozen=True)
class ModcodRow:
    name:str; minimum_snr_db:float; spectral_efficiency:float; rate_gbps:float|None=None

# ETSI EN 302 307-1, normal FECFRAME, ideal Es/N0 (not receiver C/N).
DEFAULT_MODCOD_TABLE=(
 ModcodRow("QPSK 1/4",-2.35,.490243),ModcodRow("QPSK 1/3",-1.24,.656448),
 ModcodRow("QPSK 2/5",-.30,.789412),ModcodRow("QPSK 1/2",1.,.988858),
 ModcodRow("QPSK 3/5",2.23,1.188304),ModcodRow("QPSK 2/3",3.10,1.322253),
 ModcodRow("QPSK 3/4",4.03,1.487473),ModcodRow("QPSK 4/5",4.68,1.587196),
 ModcodRow("QPSK 5/6",5.18,1.654663),ModcodRow("8PSK 3/5",5.50,1.779991),
 ModcodRow("8PSK 2/3",6.62,1.980636),ModcodRow("8PSK 3/4",7.91,2.228124),
 ModcodRow("16APSK 2/3",8.97,2.637201),ModcodRow("8PSK 5/6",9.35,2.478562),
 ModcodRow("16APSK 3/4",10.21,2.966728),ModcodRow("16APSK 4/5",11.03,3.165623),
 ModcodRow("16APSK 5/6",11.61,3.300184),ModcodRow("32APSK 3/4",12.73,3.703295),
 ModcodRow("32APSK 4/5",13.64,3.951571))

def load_modcod_table(path):
    if not path:return DEFAULT_MODCOD_TABLE
    rows=[]
    with open(path,"r",encoding="utf-8-sig",newline="") as f:
        for i,r in enumerate(csv.DictReader(f),1):
            name=r.get("modcod") or r.get("name") or r.get("档位") or f"MCS-{i}"
            snr=r.get("minimum_snr_db") or r.get("snr_db") or r.get("最低SNR_dB")
            eta=r.get("spectral_efficiency") or r.get("eta") or r.get("频谱效率")
            rate=r.get("rate_gbps") or r.get("速率_Gbps")
            if snr is None or (eta is None and rate is None):raise ValueError("SNR—速率表字段不完整")
            rows.append(ModcodRow(str(name),float(snr),float(eta or 0),float(rate) if rate not in (None,"") else None))
    if not rows:raise ValueError("SNR—速率表没有有效数据行")
    return tuple(sorted(rows,key=lambda x:x.minimum_snr_db))

@dataclass
class PlannerConfig:
    planning_period_s:float=300.; planning_horizon_s:float=1200.; planning_slot_s:float=15.
    fast_evaluation_period_s:float=1.; fast_adjust_period_s:float=1.; route_update_period_s:float=15.
    freeze_window_s:float=30.; bandwidth_mhz:float=500.; snr_margin_db:float=1.
    # Reference downlink hardware assumptions, configurable in the scene.
    feeder_frequency_ghz:float=39.; feeder_rolloff:float=.20
    satellite_eirp_dbw:float=26.; gateway_gt_dbk:float=30.; gateway_array_loss_db:float=1.
    reference_satellite_altitude_km:float=508.
    capacity_threshold_ratio:float=.85; switching_loss_ratio:float=.02
    modcod_table:Sequence[ModcodRow]=DEFAULT_MODCOD_TABLE
    rain_enabled:bool=True; rain_sense_sigma:float=.20
    rain_forecast_hours:float=2.; rain_use_p90:bool=True
    minimum_dwell_s:float=5.; switching_gain_ratio:float=.03
    source_exit_hold_s:float=60.
    traffic_landing_ratio:float=.30
    geo_interference_enabled:bool=True; geo_satellite_count:int=3
    gso_i_over_n_db:float=-12.; gso_protection_angle_deg:float=10.
    capacity_weight:float=0.; delay_weight:float=0.
    capacity_reserve_ratio:float=.20
    offline_mode:bool=True
    priority_task_regions:Sequence[dict]=field(default_factory=tuple)
    priority_task_weight:float=10.
    # One-way service-satellite -> gateway propagation is normally only a few
    # milliseconds in this LEO scenario.  A 100 ms reference compressed all
    # delay qualities into roughly 0.9--1.0 and made the delay weight inert.
    delay_reference_ms:float=15.
    # The legacy selector uses a capacity target. The weighted window selector
    # treats link capacity as a score while physical link limits still apply.
    feeder_min_separation_deg:float=8.
    feeder_diversity_target_deg:float=25.
    planning_exit_limit_per_source:int=8
    realtime_exit_limit_per_source:int=8
    selection_strategy:str="legacy"

def gso_interference_penalty_db(cfg):
    if not cfg.geo_interference_enabled or cfg.geo_satellite_count<=0:return 0.
    aggregate_i_over_n=10*math.log10(max(1,cfg.geo_satellite_count))+cfg.gso_i_over_n_db
    return 10*math.log10(1+10**(aggregate_i_over_n/10))

def _surface_distance_km(lat1,lon1,lat2,lon2):
    p1,p2=math.radians(float(lat1)),math.radians(float(lat2));dp=p2-p1;dl=math.radians(float(lon2)-float(lon1))
    value=math.sin(dp/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 6371.0088*2*math.asin(min(1.,math.sqrt(max(0.,value))))

def _candidate_angle_deg(first,second):
    a=first.get("los_unit");b=second.get("los_unit")
    if not a or not b:return 180.
    dot=max(-1.,min(1.,sum(float(x)*float(y) for x,y in zip(a,b))))
    return math.degrees(math.acos(dot))

def _candidate_diversity(candidate,selected,target_deg):
    same=[item for item in selected if int(item.get("gateway",-1))==int(candidate.get("gateway",-2))]
    if not same:return 1.,180.,1.
    minimum=min(_candidate_angle_deg(candidate,item) for item in same)
    angular=min(1.,minimum/max(.001,float(target_deg)))
    plane_bonus=1. if all(int(item.get("plane",-1))!=int(candidate.get("plane",-1)) for item in same) else 0.
    return .8*angular+.2*plane_bonus,minimum,plane_bonus

def _spherical_ecef(lat_deg,lon_deg,alt_m=0.):
    lat,lon=map(math.radians,(float(lat_deg),float(lon_deg)));radius=EARTH_RADIUS_M+float(alt_m)
    return (radius*math.cos(lat)*math.cos(lon),radius*math.cos(lat)*math.sin(lon),radius*math.sin(lat))

def _slant_distance_km(altitude_km,elevation_deg):
    radius=EARTH_RADIUS_M/1000.;e=math.radians(float(elevation_deg));h=max(0.,float(altitude_km))
    return math.sqrt((radius+h)**2-(radius*math.cos(e))**2)-radius*math.sin(e)

def link_state(elevation,cfg,rain_attenuation_db=0.,slant_distance_km=None):
    """Reference downlink budget: satellite EIRP -> gateway G/T, Es/N0.

    The configured bandwidth is occupied RF bandwidth. DVB-S2 spectral
    efficiencies are per symbol, so symbol rate is bandwidth/(1+roll-off).
    No pointing loss is added: the gateway is assumed to track ideally.
    """
    distance=max(.001,float(slant_distance_km if slant_distance_km is not None else
                            _slant_distance_km(cfg.reference_satellite_altitude_km,elevation)))
    frequency=max(.001,float(cfg.feeder_frequency_ghz));symbol_rate_mhz=max(.001,float(cfg.bandwidth_mhz))/(1+max(0.,float(cfg.feeder_rolloff)))
    fspl_db=92.45+20*math.log10(frequency)+20*math.log10(distance)
    es_n0=(float(cfg.satellite_eirp_dbw)+float(cfg.gateway_gt_dbk)-fspl_db+228.6
           -10*math.log10(symbol_rate_mhz*1e6)-float(cfg.gateway_array_loss_db)
           -max(0.,float(rain_attenuation_db))-gso_interference_penalty_db(cfg))
    selected=None;usable=es_n0-cfg.snr_margin_db
    for row in cfg.modcod_table:
        if usable>=row.minimum_snr_db and (selected is None or row.spectral_efficiency>selected.spectral_efficiency):selected=row
    if selected is None:return round(es_n0,3),"不可用",0.
    raw_rate=selected.rate_gbps if selected.rate_gbps is not None else symbol_rate_mhz*selected.spectral_efficiency/1000
    return round(es_n0,3),selected.name,round(min(GATEWAY_CHANNEL_CAPACITY_GBPS,raw_rate),6)

def _graph(positions,links):
    xyz=[]
    for r in positions:
        lat,lon=map(math.radians,(r[2],r[3])); radius=EARTH_RADIUS_M+r[4]
        xyz.append((radius*math.cos(lat)*math.cos(lon),radius*math.cos(lat)*math.sin(lon),radius*math.sin(lat)))
    g=[[] for _ in positions]
    for a,b,*_ in links:
        d=math.dist(xyz[a],xyz[b]);g[a].append((b,d));g[b].append((a,d))
    return g

def _nearest_demand(g,demands):
    hops=[10**9]*len(g); strength=[0.]*len(g);q=deque()
    for s,d in demands.items():hops[s]=0;strength[s]=d;q.append(s)
    while q:
        n=q.popleft()
        for v,_ in g[n]:
            if hops[v]>hops[n]+1:hops[v]=hops[n]+1;strength[v]=strength[n];q.append(v)
            elif hops[v]==hops[n]+1:strength[v]=max(strength[v],strength[n])
    return hops,strength

def _nearest_demand_distance(g,demands):
    distance=[math.inf]*len(g);queue=[]
    for satellite in demands:distance[satellite]=0.;heapq.heappush(queue,(0.,satellite))
    while queue:
        current,node=heapq.heappop(queue)
        if current!=distance[node]:continue
        for neighbor,edge_distance in g[node]:
            candidate=current+edge_distance
            if candidate<distance[neighbor]:distance[neighbor]=candidate;heapq.heappush(queue,(candidate,neighbor))
    return distance

def _component_demand(g,demands):
    """标出每个ISL连通分量内可汇聚的业务，避免把容量方案投到不可达分量。"""
    values=[0.]*len(g);seen=set()
    for start in range(len(g)):
        if start in seen:continue
        stack=[start];seen.add(start);nodes=[]
        while stack:
            n=stack.pop();nodes.append(n)
            for v,_ in g[n]:
                if v not in seen:seen.add(v);stack.append(v)
        demand=sum(demands.get(n,0.) for n in nodes)
        for n in nodes:values[n]=demand
    return values

def _component_ids(g):
    ids=[-1]*len(g);component=0
    for start in range(len(g)):
        if ids[start]>=0:continue
        ids[start]=component;stack=[start]
        while stack:
            node=stack.pop()
            for neighbor,_distance in g[node]:
                if ids[neighbor]<0:ids[neighbor]=component;stack.append(neighbor)
        component+=1
    return ids

def _k_nearest_candidate_options(g,candidates,demands,k=3):
    """Return at most K reachable feeder choices for every traffic satellite.

    Candidate labels propagate together in one graph search.  The graph keeps
    only its K best labels at each node, avoiding one Dijkstra run per feeder.
    Initial distance includes the candidate's feeder-link distance.
    """
    k=max(1,int(k));labels=[{} for _ in g];queue=[]
    # A gateway-satellite pair is one label: the same satellite can have
    # different GSL distances to different gateways.
    for index,item in enumerate(candidates):
        node=int(item["satellite"])
        if not 0<=node<len(g):continue
        distance=max(0.,float(item.get("gsl_distance_km",0)))*1000.
        bucket=labels[node]
        if len(bucket)>=k:
            worst_index,worst_distance=max(bucket.items(),key=lambda item:item[1])
            if distance>=worst_distance:continue
            del bucket[worst_index]
        bucket[index]=distance;heapq.heappush(queue,(distance,node,index))
    while queue:
        distance,node,index=heapq.heappop(queue)
        if labels[node].get(index)!=distance:continue
        for neighbor,edge_distance in g[node]:
            candidate_distance=distance+edge_distance;bucket=labels[neighbor]
            previous=bucket.get(index)
            if previous is not None and previous<=candidate_distance:continue
            if previous is None and len(bucket)>=k:
                worst_index,worst_distance=max(bucket.items(),key=lambda item:item[1])
                if candidate_distance>=worst_distance:continue
                del bucket[worst_index]
            bucket[index]=candidate_distance;heapq.heappush(queue,(candidate_distance,neighbor,index))
    demand_labels={}
    for source,demand in demands.items():
        if not (0<=source<len(labels)):continue
        demand_labels[source]=(float(demand),labels[source])
    options=[[] for _ in candidates]
    for candidate_index,candidate in enumerate(candidates):
        for source,(demand,source_labels) in demand_labels.items():
            total_distance=source_labels.get(candidate_index)
            if total_distance is not None:options[candidate_index].append((source,demand,total_distance/1000/C_KM_PER_MS))
    for rows in options:rows.sort(key=lambda row:row[2])
    return options

def _normalized_objective_weights(capacity_weight,delay_weight):
    capacity_weight=max(0.,float(capacity_weight));delay_weight=max(0.,float(delay_weight))
    total=capacity_weight+delay_weight
    if total<=0:return 1.,0.
    return capacity_weight/total,delay_weight/total

def _marginal_utility(carried_gbps,delay_ms,total_demand_gbps,capacity_weight,delay_weight,delay_reference_ms):
    """Comparable gain in the global capacity-delay objective.

    Capacity is normalized by total demand. Delay is an independent path
    quality term and is deliberately not multiplied by carried traffic or
    demand. The caller enforces capacity feasibility separately.
    """
    carried=max(0.,float(carried_gbps));demand=max(1e-12,float(total_demand_gbps))
    reference=max(1e-6,float(delay_reference_ms));wc,wd=_normalized_objective_weights(capacity_weight,delay_weight)
    capacity_gain=carried/demand
    # Smooth quality retains a preference between paths above the reference.
    delay_quality=reference/(reference+max(0.,float(delay_ms)))
    delay_gain=delay_quality
    return wc*capacity_gain+wd*delay_gain,capacity_gain,delay_gain,delay_quality

def _routes(g,feeders,demands,feeder_capacities=None,candidate_count=8,capacity_weight=1.,delay_weight=0.,delay_reference_ms=15.,feeder_gsl_distances=None,max_active_exits_per_source=2,source_exit_limits=None,source_excluded_feeders=None,source_priorities=None):
    """Route traffic to nearby feeders without overloading one exit.

    The previous nearest-only multi-source Dijkstra ignored feeder capacity.
    It could send most traffic to a few exits while many selected links stayed
    idle.  Keep a small set of nearby feeder labels at every node, then assign
    each traffic source to the nearest label that still has capacity.
    """
    feeders=sorted({int(s) for s in feeders});k=max(1,min(int(candidate_count),len(feeders)))
    if not feeders:return []
    feeder_gsl_distances=feeder_gsl_distances or {}
    labels=[{} for _ in g];next_hop=[{} for _ in g];q=[]
    for label,feeder in enumerate(feeders):
        initial=max(0.,float(feeder_gsl_distances.get(feeder,0)))*1000.
        labels[feeder][label]=initial;next_hop[feeder][label]=-1;heapq.heappush(q,(initial,feeder,label))
    while q:
        distance,node,label=heapq.heappop(q)
        if labels[node].get(label)!=distance:continue
        for neighbor,edge_distance in g[node]:
            candidate=distance+edge_distance;bucket=labels[neighbor];previous=bucket.get(label)
            if previous is not None and previous<=candidate:continue
            if previous is None and len(bucket)>=k:
                worst_label,worst_distance=max(bucket.items(),key=lambda item:item[1])
                if candidate>=worst_distance:continue
                del bucket[worst_label];next_hop[neighbor].pop(worst_label,None)
            bucket[label]=candidate;next_hop[neighbor][label]=node;heapq.heappush(q,(candidate,neighbor,label))
    if feeder_capacities is None:remaining={s:math.inf for s in feeders}
    else:remaining={s:max(0.,float(feeder_capacities.get(s,0.))) for s in feeders}
    result=[];total_demand=max(sum(max(0.,float(value)) for value in demands.values()),1e-12)
    source_exit_limits=dict(source_exit_limits or {});source_excluded_feeders=source_excluded_feeders or {};source_priorities=source_priorities or {}
    max_active_exits_per_source=max(1,int(max_active_exits_per_source))
    for source,demand in demands.items():
        if int(source) not in source_exit_limits:
            # 出口数是上限，不应按单条链路的标称容量提前压成
            # 一个出口。第一个出口剩余容量不足时，第二出口负责补齐业务。
            source_exit_limits[int(source)]=max_active_exits_per_source
    residual={int(source):max(0.,float(demand)) for source,demand in demands.items()};exits={int(source):set() for source in demands};route_by_pair={}
    def allocate(source,target,force_self=False):
        requested=max(0.,float(demands.get(source,0.)));target=min(max(0.,float(target)),residual.get(source,0.));available_labels=set(labels[source])
        exit_limit=max(0,min(max_active_exits_per_source,int(source_exit_limits.get(int(source),max_active_exits_per_source))))
        excluded={int(value) for value in source_excluded_feeders.get(int(source),())}
        while target>1e-12 and available_labels:
            options=[]
            usable_capacity={label:max(0.,remaining.get(feeders[label],0.)) for label in available_labels
                             if feeders[label] not in excluded}
            for label in available_labels:
                distance=labels[source][label];feeder=feeders[label];available=remaining.get(feeder,0.)
                if feeder in excluded or (feeder not in exits[source] and len(exits[source])>=exit_limit):continue
                if available<=1e-12:continue
                carried=min(target,available)
                total_distance_km=distance/1000
                delay_ms=total_distance_km/C_KM_PER_MS
                utility,capacity_gain,delay_gain,delay_quality=_marginal_utility(
                    carried,delay_ms,total_demand,capacity_weight,delay_weight,delay_reference_ms)
                utility*=max(1.,float(source_priorities.get(int(source),1.)))
                # The coverage pass may allocate only a small fair-share amount.
                # Rank the first/second exit by how much of this source's whole
                # residual demand it can ultimately carry; otherwise a nearby
                # low-capacity exit can consume one of the two exit slots and
                # strand hotspot demand despite ample network-wide capacity.
                source_carry_capacity=min(residual.get(source,0.),available)
                future_slots=max(0,exit_limit-len(exits[source])-(feeder not in exits[source]))
                future_capacity=sum(sorted((value for other,value in usable_capacity.items() if other!=label),reverse=True)[:future_slots])
                feasible=available+future_capacity+1e-9>=residual.get(source,0.)
                options.append((feasible,1 if feeder==int(source) else 0,source_carry_capacity,carried,utility,-delay_ms,available,label,feeder,capacity_gain,delay_gain,delay_quality))
            if not options:return
            own=next((row for row in options if row[8]==int(source)),None) if force_self else None
            feasible_options=[row for row in options if row[0]]
            if own is not None:choice=own
            elif capacity_weight==0 and delay_weight==0:
                # Lexicographic objective: a feasible exit first, then the
                # shortest path. If none can carry the source, carry as much
                # as possible before comparing path delay.
                choice=(max(feasible_options,key=lambda row:(row[5],row[2])) if feasible_options
                        else max(options,key=lambda row:(row[2],row[5])))
            elif feasible_options:choice=max(feasible_options,key=lambda row:(row[4],row[1],row[2],row[5]))
            else:choice=max(options,key=lambda row:(row[2],row[3],row[4],row[1],row[5]))
            _feasible,_local_priority,_source_carry_capacity,carried,utility,_negative_delay,_available,label,feeder,capacity_gain,delay_gain,delay_quality=choice
            available_labels.discard(label);distance=labels[source][label]
            path=[int(source)];node=int(source);seen={node}
            while node!=feeder and len(path)<=len(g):
                node=next_hop[node].get(label,-1)
                if node<0 or node in seen:break
                seen.add(node);path.append(node)
            if not path or path[-1]!=feeder:continue
            isl_distance=max(0.,distance-max(0.,float(feeder_gsl_distances.get(feeder,0)))*1000.)
            remaining[feeder]-=carried;residual[source]=max(0.,residual[source]-carried);target-=carried;exits[source].add(feeder)
            key=(source,feeder);route=route_by_pair.get(key)
            if route is None:
                route={"source":int(source),"feeder":feeder,"path":path,"hops":len(path)-1,
                       "distance_km":round(isl_distance/1000,3),"isl_propagation_delay_ms":round(isl_distance/1000/C_KM_PER_MS,4),
                       "demand_gbps":0.,"requested_demand_gbps":round(requested,6),
                       "route_objective_score":round(utility,9),"route_capacity_gain":round(capacity_gain,9),
                       "route_delay_gain":round(delay_gain,9),"route_delay_quality":round(delay_quality,9)}
                route_by_pair[key]=route;result.append(route)
            route["demand_gbps"]=round(float(route["demand_gbps"])+carried,6)
    # A selected feeder first lands its own traffic through its zero-hop GSL.
    # Otherwise large remote sources can consume its capacity first and force
    # the feeder satellite's small local demand onto a multi-hop ISL detour.
    local_sources=[source for source in residual if int(source) in remaining]
    for source in sorted(local_sources,key=lambda value:residual[value],reverse=True):
        allocate(source,min(residual[source],remaining.get(int(source),0.)),force_self=True)
    # Coverage pass: reserve a fair first share for every business satellite.
    # Process hotspot sources first because they have the strictest two-exit
    # feasibility condition. Small sources can use many residual fragments;
    # doing them first can consume both usable exits of a hotspot and leave
    # capacity stranded elsewhere in the network.
    finite_capacity=sum(value for value in remaining.values() if math.isfinite(value));fair_share=min(total_demand/max(len(demands),1),finite_capacity/max(len(demands),1)) if finite_capacity else total_demand/max(len(demands),1)
    for source in sorted(residual,key=lambda value:(float(source_priorities.get(int(value),1.)),residual[value]),reverse=True):allocate(source,min(residual[source],fair_share))
    # Throughput pass: use the second exit, when needed, to fill residual demand.
    for source in sorted(residual,key=lambda value:(float(source_priorities.get(int(value),1.)),residual[value]),reverse=True):allocate(source,residual[source])
    return result

def _routes_with_capacity_completion(g,feeders,demands,feeder_capacities=None,candidate_count=8,
                                     capacity_weight=1.,delay_weight=0.,delay_reference_ms=15.,
                                     feeder_gsl_distances=None,max_active_exits_per_source=2,source_priorities=None):
    """Allocate nearby exits first, then expose unused feeder capacity.

    A bounded nearest-exit search is necessary for real-time performance, but a
    single pass can strand capacity behind saturated nearby exits.
    The second pass runs only when demand and feeder capacity both remain.  It
    searches among the residual feeders and respects the configured exit limit.
    """
    capacities={int(key):max(0.,float(value)) for key,value in (feeder_capacities or {}).items()}
    routes=_routes(g,feeders,demands,capacities,candidate_count=candidate_count,
                   capacity_weight=capacity_weight,delay_weight=delay_weight,
                   delay_reference_ms=delay_reference_ms,feeder_gsl_distances=feeder_gsl_distances,
                   max_active_exits_per_source=max_active_exits_per_source,source_priorities=source_priorities)
    used_by_feeder={};allocated_by_source={};exits_by_source={}
    for route in routes:
        source=int(route["source"]);feeder=int(route["feeder"]);amount=float(route.get("demand_gbps",0))
        used_by_feeder[feeder]=used_by_feeder.get(feeder,0.)+amount
        allocated_by_source[source]=allocated_by_source.get(source,0.)+amount
        exits_by_source.setdefault(source,set()).add(feeder)
    residual_capacities={feeder:max(0.,capacity-used_by_feeder.get(feeder,0.))
                         for feeder,capacity in capacities.items()
                         if capacity-used_by_feeder.get(feeder,0.)>1e-9}
    residual_demands={int(source):max(0.,float(value)-allocated_by_source.get(int(source),0.))
                      for source,value in demands.items()
                      if float(value)-allocated_by_source.get(int(source),0.)>1e-9 and
                         len(exits_by_source.get(int(source),set()))<max_active_exits_per_source}
    # Expand only over feeders that still have capacity. Usually one pass is
    # sufficient; the bound of three prevents pathological work while allowing
    # an augmenting assignment to escape local capacity fragmentation.
    for _pass in range(3):
        if not residual_capacities or not residual_demands:break
        exit_limits={source:max_active_exits_per_source-len(exits_by_source.get(source,set()))
                     for source in residual_demands}
        extra=_routes(g,residual_capacities,residual_demands,residual_capacities,
                      candidate_count=min(12,len(residual_capacities)),
                      capacity_weight=capacity_weight,delay_weight=delay_weight,
                      delay_reference_ms=delay_reference_ms,feeder_gsl_distances=feeder_gsl_distances,
                      max_active_exits_per_source=max_active_exits_per_source,
                      source_exit_limits=exit_limits,source_excluded_feeders=exits_by_source,source_priorities=source_priorities)
        if not extra:break
        progress=sum(float(route.get("demand_gbps",0)) for route in extra)
        if progress<=1e-9:break
        routes.extend(extra)
        for route in extra:
            source=int(route["source"]);feeder=int(route["feeder"]);amount=float(route.get("demand_gbps",0))
            residual_demands[source]=max(0.,residual_demands.get(source,0.)-amount)
            residual_capacities[feeder]=max(0.,residual_capacities.get(feeder,0.)-amount)
            exits_by_source.setdefault(source,set()).add(feeder)
        residual_demands={source:value for source,value in residual_demands.items()
                          if value>1e-9 and len(exits_by_source.get(source,set()))<max_active_exits_per_source}
        residual_capacities={feeder:value for feeder,value in residual_capacities.items() if value>1e-9}
    # A source may already occupy its two allowed exits while both exits are
    # nearly exhausted.  The append-only completion above cannot use a third
    # feeder even when ample capacity remains elsewhere.  Rebalance all
    # deficient sources as one batch: release their old allocations, route
    # them again with the same two-exit limit, and accept the batch only when
    # aggregate delivered traffic increases.  This preserves the physical
    # two-exit constraint without stranding usable feeder capacity.
    for _rebalance in range(2):
        allocated={}
        for route in routes:
            source=int(route["source"]);allocated[source]=allocated.get(source,0.)+float(route.get("demand_gbps",0))
        deficient={int(source):float(value) for source,value in demands.items()
                   if float(value)-allocated.get(int(source),0.)>1e-6}
        if not deficient:break
        retained=[route for route in routes if int(route["source"]) not in deficient]
        used={}
        for route in retained:
            feeder=int(route["feeder"]);used[feeder]=used.get(feeder,0.)+float(route.get("demand_gbps",0))
        available={feeder:max(0.,capacity-used.get(feeder,0.)) for feeder,capacity in capacities.items()
                   if capacity-used.get(feeder,0.)>1e-9}
        rerouted=_routes(g,available,deficient,available,
                         candidate_count=min(24,len(available)),capacity_weight=capacity_weight,
                         delay_weight=delay_weight,delay_reference_ms=delay_reference_ms,
                         feeder_gsl_distances=feeder_gsl_distances,
                         max_active_exits_per_source=max_active_exits_per_source,
                         source_priorities=source_priorities) if available else []
        old_amount=sum(float(route.get("demand_gbps",0)) for route in routes)
        new_routes=retained+rerouted
        new_amount=sum(float(route.get("demand_gbps",0)) for route in new_routes)
        if new_amount<=old_amount+1e-6:break
        routes=new_routes
    return routes

def _repair_routes_local(g,demands,feeder_capacities,feeder_gsl_distances=None,
                         capacity_weight=1.,delay_weight=0.,delay_reference_ms=15.,
                         excluded_feeders=None,exit_limits=None,max_visited_per_source=2000,
                         max_exits_per_source=2):
    """Repair affected sources with bounded source-local Dijkstra searches."""
    remaining={int(k):max(0.,float(v)) for k,v in feeder_capacities.items() if float(v)>1e-9}
    feeder_set=set(remaining);gsl=feeder_gsl_distances or {};excluded_feeders=excluded_feeders or {};exit_limits=exit_limits or {}
    total=max(sum(max(0.,float(v)) for v in demands.values()),1e-12);result=[]
    for source,requested in sorted(demands.items(),key=lambda item:(int(item[0]) in feeder_set,item[1]),reverse=True):
        source=int(source);left=max(0.,float(requested));limit=max(0,min(max_exits_per_source,int(exit_limits.get(source,max_exits_per_source))))
        if left<=1e-9 or limit<=0 or not remaining:continue
        distances={source:0.};previous={source:-1};queue=[(0.,source)];found=[];visited=0;excluded=set(excluded_feeders.get(source,()))
        while queue and visited<max_visited_per_source and len(found)<max(12,limit):
            distance,node=heapq.heappop(queue)
            if distance!=distances.get(node):continue
            visited+=1
            if node in feeder_set and node not in excluded and remaining.get(node,0)>1e-9:found.append((distance,node))
            for neighbor,edge_distance in g[node]:
                candidate=distance+edge_distance
                if candidate<distances.get(neighbor,math.inf):distances[neighbor]=candidate;previous[neighbor]=node;heapq.heappush(queue,(candidate,neighbor))
        options=[]
        for distance,feeder in found:
            carried=min(left,remaining.get(feeder,0.));delay=(distance/1000+float(gsl.get(feeder,0)))/C_KM_PER_MS
            utility,*_= _marginal_utility(carried,delay,total,capacity_weight,delay_weight,delay_reference_ms)
            options.append((1 if feeder==source else 0,-delay,carried,feeder,distance))
        for _local_priority,_negative_delay,_initial_carried,feeder,distance in sorted(options,reverse=True)[:limit]:
            carried=min(left,remaining.get(feeder,0.))
            if carried<=1e-9:continue
            path=[];node=feeder
            while node>=0:path.append(node);node=previous.get(node,-1)
            path.reverse()
            if not path or path[0]!=source:continue
            delay=(distance/1000+float(gsl.get(feeder,0)))/C_KM_PER_MS
            result.append({"source":source,"feeder":feeder,"path":path,"hops":len(path)-1,
                "distance_km":round(distance/1000,3),"isl_propagation_delay_ms":round(distance/1000/C_KM_PER_MS,4),
                "propagation_delay_ms":round(delay,4),"demand_gbps":round(carried,6),
                "requested_demand_gbps":round(float(requested),6),"route_objective_score":round(utility,9)})
            remaining[feeder]-=carried;left-=carried
            if left<=1e-9:break
    return result

def _gsl_distance_km(altitude_m,elevation_deg):
    radius=EARTH_RADIUS_M;orbit_radius=radius+max(0.,float(altitude_m));elevation=math.radians(float(elevation_deg))
    return (math.sqrt(max(0.,orbit_radius*orbit_radius-(radius*math.cos(elevation))**2))-radius*math.sin(elevation))/1000

def _attach_feeder_delay(routes,links):
    feeder_links={int(x["satellite"]):x for x in links}
    for route in routes:
        link=feeder_links.get(int(route.get("feeder",-1)));gsl_distance=float((link or {}).get("gsl_distance_km",0))
        route["gsl_distance_km"]=round(gsl_distance,3)
        route["gsl_propagation_delay_ms"]=round(gsl_distance/C_KM_PER_MS,4)
        route["propagation_delay_ms"]=round(float(route.get("isl_propagation_delay_ms",0))+gsl_distance/C_KM_PER_MS,4)
    return routes

def _compact_routes(routes,demands,max_exits_per_source=2):
    """Keep at most one allocation row per source/exit and cap stale demand."""
    grouped={}
    for row in routes:
        source=int(row.get("source",-1));feeder=int(row.get("feeder",-1))
        if source not in demands or feeder<0:continue
        key=(source,feeder);amount=max(0.,float(row.get("demand_gbps",0)))
        if key not in grouped:
            grouped[key]=row.copy();grouped[key]["demand_gbps"]=amount
        else:
            grouped[key]["demand_gbps"]+=amount
            if float(row.get("propagation_delay_ms",math.inf))<float(grouped[key].get("propagation_delay_ms",math.inf)):
                kept=row.copy();kept["demand_gbps"]=grouped[key]["demand_gbps"];grouped[key]=kept
    result=[]
    for source in demands:
        remaining=max(0.,float(demands[source]))
        rows=sorted((row for (sid,_),row in grouped.items() if sid==source),key=lambda x:float(x.get("propagation_delay_ms",math.inf)))[:max_exits_per_source]
        for row in rows:
            amount=min(remaining,float(row.get("demand_gbps",0)))
            if amount<=1e-9:continue
            row["demand_gbps"]=round(amount,6);result.append(row);remaining-=amount
    return result

def _route_metrics(routes):
    if not routes:return 0.,0.,0.,0.,0.
    demand=sum(x["demand_gbps"] for x in routes)
    weighted_delay=sum(float(x.get("propagation_delay_ms",x["distance_km"]/C_KM_PER_MS))*x["demand_gbps"] for x in routes)/max(demand,1e-12)
    ordered=sorted(routes,key=lambda x:float(x.get("propagation_delay_ms",0)));cumulative=0.;p95=0.
    for route in ordered:
        cumulative+=float(route["demand_gbps"])
        if cumulative>=.95*demand:p95=float(route.get("propagation_delay_ms",0));break
    return (round(sum(x["hops"] for x in routes)/len(routes),3),
            round(sum(x["hops"]*x["demand_gbps"] for x in routes)/max(demand,1e-12),3),
            round(weighted_delay,3),round(p95,3),round(max(float(x.get("propagation_delay_ms",0)) for x in routes),3))

@dataclass
class PlanningState:
    generated_at_s:float=-math.inf; generation:int=0; plans:list=field(default_factory=list)
    active_plan_id:str="slow_plan"; last_switch_s:float=-math.inf; switch_count:int=0
    last_scheduled_pairs:set=field(default_factory=set)
    last_evaluation_s:float=-math.inf; last_adjust_s:float=-math.inf
    last_route_update_s:float=-math.inf; last_result:dict|None=None
    execution_plan:dict|None=None
    last_route_sources:set=field(default_factory=set)
    source_exits:dict=field(default_factory=dict)
    source_exit_changed_at:dict=field(default_factory=dict)
    last_fast_epoch:int=-1
    last_demands:dict=field(default_factory=dict)
    last_feeder_capacities:dict=field(default_factory=dict)
    last_capacity_reroute_s:float=-math.inf

class SlowFastLoopPlanner:
    """慢环生成一套规划骨架，快环逐帧执行和局部修复。"""
    def __init__(self,stations,config=None):
        self.stations=list(stations);self.config=config or PlannerConfig();self.state=PlanningState()
        self._offline_graph_key=None;self._offline_graph=None
    def _candidates(self,gsls,t,planning=False,rain_scenario_percentile=None,rain_grade=None,positions=None,window_rates=None):
        out=[]
        for gid,sid,elev in gsls:
            rain=link_rain_state(self.stations[int(gid)],float(elev),t,self.config.rain_enabled,self.config.rain_sense_sigma,self.config.rain_forecast_hours,self.config.rain_use_p90)
            statistical_fade=rain["planning_attenuation_db"] if planning else rain["sensed_attenuation_db"]
            if planning and rain_grade and self.config.rain_enabled:
                fade=scenario_attenuation_db(self.stations[int(gid)],float(elev),rain_grade,t)
            else:fade=max(0.,float(statistical_fade))
            altitude=float(positions[int(sid)][4])/1000 if positions is not None else self.config.reference_satellite_altitude_km
            slant=_slant_distance_km(altitude,float(elev))
            snr,mcs,rate=link_state(float(elev),self.config,fade,slant)
            if planning and self.config.selection_strategy=="window_margin" and window_rates is not None:
                window_state=window_rates.get((int(gid),int(sid))) or {}
                rate=max(0.,float(window_state.get("rate_gbps",0.)))
                snr=float(window_state.get("snr_db",snr))
                mcs=str(window_state.get("modcod",mcs))
                fade=float(window_state.get("rain_attenuation_db",fade))
            if rate>0:out.append({"gateway":int(gid),"satellite":int(sid),"elevation_deg":float(elev),"snr_db":snr,
                                  "modcod":mcs,"rate_gbps":rate,"connection":1,"array_mode":1,"power_factor":1.0,
                                  "window_min_rate_gbps":rate if planning and self.config.selection_strategy=="window_margin" else None,
                                  "window_mean_rate_gbps":window_state.get("mean_rate_gbps") if planning and self.config.selection_strategy=="window_margin" and window_rates is not None else None,
                                  "window_worst_elevation_deg":window_state.get("worst_elevation_deg") if planning and self.config.selection_strategy=="window_margin" and window_rates is not None else None,
                                  "window_worst_time_s":window_state.get("worst_time_s") if planning and self.config.selection_strategy=="window_margin" and window_rates is not None else None,
                                  "gso_interference_db":round(gso_interference_penalty_db(self.config),3),
                                  "rain_attenuation_db":fade,"rain_statistical_db":statistical_fade,
                                  "rain_spatial_factor":scenario_spatial_factor(self.stations[int(gid)],rain_grade,t) if planning and rain_grade else 1.0,
                                  "rain_scenario_percentile":rain_scenario_percentile,"rain_actual_db":rain["actual_attenuation_db"],
                                  "rain_rate_mmh":rain["rain_rate_mmh"],"rain_severity":rain_grade or rain["severity"]})
        return out
    def _select(self,candidates,g,demands,capacity_only=False):
        component_demand=_component_demand(g,demands)
        component_by_node=_component_ids(g)
        if demands:candidates=[x for x in candidates if component_demand[x["satellite"]]>0]
        if not candidates:return []
        capacity_weight,delay_weight=_normalized_objective_weights(self.config.capacity_weight,self.config.delay_weight)
        total_demand=max(sum(demands.values()),1e-12);delay_reference=max(.001,float(self.config.delay_reference_ms))
        def select_with_k(k):
            options=_k_nearest_candidate_options(g,candidates,demands,k)
            remaining={i:max(0,int(s.get("channel_count",s.get("antenna_count",20)))) for i,s in enumerate(self.stations)}
            unmet={int(source):float(value) for source,value in demands.items()};selected=[];used=set();available=set(range(len(candidates)))
            while available and sum(remaining.values())>0 and sum(unmet.values())>1e-12:
                evaluated=[]
                for index in tuple(available):
                    item=candidates[index];gateway=int(item["gateway"]);satellite=int(item["satellite"])
                    if remaining.get(gateway,0)<=0 or satellite in used:continue
                    capacity=max(0.,float(item["rate_gbps"]));delivered=0.;weighted_delay=0.;allocation=[]
                    for source,_demand,delay in options[index]:
                        amount=min(unmet.get(source,0.),capacity-delivered)
                        if amount<=0:continue
                        delivered+=amount;weighted_delay+=amount*delay;allocation.append((source,amount,delay))
                        if delivered>=capacity-1e-12:break
                    if delivered<=0:continue
                    average_delay=weighted_delay/delivered
                    score,capacity_score,delay_gain,delay_score=_marginal_utility(
                        delivered,average_delay,total_demand,capacity_weight,delay_weight,delay_reference)
                    diversity,minimum_angle,plane_bonus=_candidate_diversity(item,selected,self.config.feeder_diversity_target_deg)
                    evaluated.append((delivered,score,diversity,minimum_angle,plane_bonus,index,allocation,average_delay,capacity_score,delay_gain,delay_score))
                if not evaluated:break
                # When remaining station slots can cover the unmet business,
                # let delay and angular diversity decide among useful links.
                # Otherwise keep the narrow capacity-first band.
                best_delivery=max(row[0] for row in evaluated)
                unmet_total=sum(unmet.values())
                by_gateway={gateway:[] for gateway in remaining}
                for row in evaluated:
                    by_gateway[int(candidates[row[5]]["gateway"])].append(row)
                prefixes={};rank_by_index={};all_gateway_top=0.
                for gateway,rows in by_gateway.items():
                    rows.sort(key=lambda row:row[0],reverse=True)
                    prefix=[0.]
                    for rank,item in enumerate(rows):
                        rank_by_index[item[5]]=rank
                        prefix.append(prefix[-1]+item[0])
                    prefixes[gateway]=prefix
                    all_gateway_top+=prefix[min(remaining[gateway],len(rows))]
                feasible=[]
                for row in evaluated:
                    gateway=int(candidates[row[5]]["gateway"])
                    rows=by_gateway[gateway];prefix=prefixes[gateway]
                    future_slots=max(0,remaining[gateway]-1)
                    rank=rank_by_index[row[5]]
                    replacement_sum=(prefix[min(future_slots+1,len(rows))]-row[0]
                                     if rank<future_slots else prefix[min(future_slots,len(rows))])
                    future_capacity=(all_gateway_top-prefix[min(remaining[gateway],len(rows))]
                                     +replacement_sum)
                    if row[0]>=max(.1,.5*best_delivery) and row[0]+.85*future_capacity>=unmet_total:
                        feasible.append(row)
                if self.config.selection_strategy=="window_margin" and not capacity_only:
                    # In the weighted variant every useful, five-minute-visible
                    # candidate competes; delivery and reserve are outcomes.
                    eligible=feasible if feasible else evaluated
                elif feasible and not capacity_only:
                    eligible=feasible
                else:
                    capacity_floor=max(0.,best_delivery-.05)
                    eligible=[row for row in evaluated if row[0]>=capacity_floor]
                # Delay weight also controls how strongly we prefer angular
                # diversity among candidates with essentially equal throughput.
                # Keep path quality dominant so dispersion cannot cause a long detour.
                diversity_weight=.25*delay_weight
                margin_scores={}
                if self.config.selection_strategy=="window_margin" and not capacity_only:
                    margins=[max(0.,float(candidates[row[5]].get("window_mean_rate_gbps") or candidates[row[5]]["rate_gbps"])) for row in eligible]
                    qualities=[row[10] for row in eligible]
                    min_margin,max_margin=min(margins),max(margins)
                    min_quality,max_quality=min(qualities),max(qualities)
                    for row,margin,quality in zip(eligible,margins,qualities):
                        margin_norm=(margin-min_margin)/(max_margin-min_margin) if max_margin-min_margin>1e-9 else .5
                        delay_norm=(quality-min_quality)/(max_quality-min_quality) if max_quality-min_quality>1e-9 else .5
                        utility=capacity_weight*margin_norm+delay_weight*delay_norm
                        margin_scores[row[5]]=(utility,margin_norm,delay_norm,margin)
                    chosen_row=max(eligible,key=lambda row:(margin_scores[row[5]][0],row[0],-row[7]))
                elif self.config.selection_strategy=="legacy" and self.config.capacity_weight==0 and self.config.delay_weight==0:
                    chosen_row=(min(eligible,key=lambda row:(row[7],-row[0])) if feasible and not capacity_only
                                else max(eligible,key=lambda row:(row[0],-row[7])))
                else:
                    chosen_row=max(eligible,key=lambda row:(row[1]+diversity_weight*row[2],
                        row[3]>=self.config.feeder_min_separation_deg,row[1],-row[7]))
                delivered,score,diversity,minimum_angle,plane_bonus,index,allocation,average_delay,capacity_score,delay_gain,delay_score=chosen_row
                rank=(score+diversity_weight*diversity,delivered,-average_delay,float(candidates[index]["elevation_deg"]))
                if self.config.selection_strategy=="legacy" and self.config.capacity_weight==0 and self.config.delay_weight==0:
                    rank=(-average_delay if feasible and not capacity_only else delivered,delivered,-average_delay,float(candidates[index]["elevation_deg"]))
                window_capacity=None
                if index in margin_scores:
                    rank=(margin_scores[index][0],delivered,-average_delay,float(candidates[index]["elevation_deg"]))
                    capacity_score=margin_scores[index][1]
                    delay_score=margin_scores[index][2]
                    window_capacity=margin_scores[index][3]
                item=candidates[index].copy()
                item.update(selection_score=round(rank[0],9),capacity_score=round(capacity_score,9),delay_score=round(delay_score,9),
                            delay_gain=round(delay_gain,9),
                            estimated_weighted_delay_ms=round(average_delay,6),assigned_demand_gbps=round(sum(x[1] for x in allocation),6),
                            diversity_score=round(diversity,6),minimum_angular_separation_deg=round(minimum_angle,3),
                             orbital_plane_bonus=round(plane_bonus,3),selection_reason=("五分钟平均容量与时延加权" if self.config.selection_strategy=="window_margin" else "容量可行集内优先时延"))
                if window_capacity is not None:item["estimated_window_capacity_gbps"]=round(window_capacity,6)
                selected.append(item);remaining[item["gateway"]]-=1;used.add(item["satellite"]);available.discard(index)
                for source,amount,_delay in allocation:unmet[source]=max(0.,unmet.get(source,0.)-amount)
                available={i for i in available if candidates[i]["satellite"]!=item["satellite"]}
            can_expand=bool(available) and sum(remaining.values())>0
            return selected,sum(unmet.values()),can_expand,unmet
        # One bounded multi-label graph pass is enough for the joint score.
        # Repeating it for K=2,4,8... dominated a frame on large constellations.
        # Residual demand is handled by the capacity fallback below.
        # Six labels retain enough exit diversity for the capacity-delay score
        # while keeping the first global plan below the one-second frame budget
        # on the 20,000-satellite scenario.  The residual-capacity fallback
        # below still considers every visible candidate.
        feeder_count=len({int(item["satellite"]) for item in candidates})
        k=min(8 if self.config.selection_strategy=="window_margin" else 6,feeder_count)
        best_selected,best_unmet,_can_expand,unmet=select_with_k(k);self._last_candidate_k=k
        # K-nearest pruning must never leave usable station resources idle while
        # landing demand remains. Fill residual quotas with the best remaining
        # capacity/latency candidates using one demand-to-network distance pass.
        if best_unmet>1e-9 and self.config.selection_strategy!="window_margin":
            used={int(item["satellite"]) for item in best_selected}
            remaining={i:max(0,int(s.get("channel_count",s.get("antenna_count",20)))-sum(int(x["gateway"])==i for x in best_selected)) for i,s in enumerate(self.stations)}
            component_unmet={}
            for source,value in unmet.items():
                component=component_by_node[source];component_unmet[component]=component_unmet.get(component,0.)+max(0.,value)
            nearest=_nearest_demand_distance(g,demands);fallback=[]
            for item in candidates:
                gateway=int(item["gateway"]);satellite=int(item["satellite"])
                if satellite in used or remaining.get(gateway,0)<=0 or not math.isfinite(nearest[satellite]):continue
                component=component_by_node[satellite];residual=component_unmet.get(component,0.)
                if residual<=1e-12:continue
                delay=(nearest[satellite]/1000+float(item.get("gsl_distance_km",0)))/C_KM_PER_MS
                capacity=min(float(item["rate_gbps"]),residual)
                score,capacity_score,delay_gain,delay_score=_marginal_utility(
                    capacity,delay,total_demand,capacity_weight,delay_weight,delay_reference)
                fallback.append((score,capacity,-delay,float(item["elevation_deg"]),capacity_score,delay_gain,delay_score,component,item))
            for score,capacity,negative_delay,_elevation,capacity_score,delay_gain,delay_score,component,raw in sorted(fallback,reverse=True,key=lambda row:row[:4]):
                gateway=int(raw["gateway"]);satellite=int(raw["satellite"])
                residual=component_unmet.get(component,0.)
                if best_unmet<=1e-9 or residual<=1e-12 or satellite in used or remaining.get(gateway,0)<=0:continue
                item=raw.copy();assigned=min(capacity,residual);delay=-negative_delay
                item.update(selection_score=round(score,9),capacity_score=round(capacity_score,9),delay_gain=round(delay_gain,9),
                            delay_score=round(delay_score,9),estimated_weighted_delay_ms=round(delay,6),
                            assigned_demand_gbps=round(assigned,6),selection_reason="剩余容量兜底")
                best_selected.append(item);used.add(satellite);remaining[gateway]-=1
                component_unmet[component]=max(0.,residual-assigned);best_unmet=max(0.,best_unmet-assigned)
        return best_selected
    def _enforce_capacity_target(self,candidates,selected,g,demands,target_gbps):
        """Repair the weighted selection until the reserve target is feasible.

        Capacity is an SLA constraint. Delay weights rank alternatives only
        after that constraint is met; they must not make a 110-Gb/s set win
        when the same station/channel pool can provide 148 Gb/s.
        """
        selected=[dict(item) for item in selected];target=max(0.,float(target_gbps))
        total=sum(float(item.get("rate_gbps",0)) for item in selected)
        if total+1e-9>=target:return selected,total,True
        component_demand=_component_demand(g,demands);nearest=_nearest_demand_distance(g,demands)
        used={int(item["satellite"]) for item in selected}
        while total+1e-9<target:
            best=None
            for candidate in candidates:
                satellite=int(candidate["satellite"]);gateway=int(candidate["gateway"])
                if satellite in used or component_demand[satellite]<=0:continue
                same_gateway=[(index,item) for index,item in enumerate(selected) if int(item["gateway"])==gateway]
                channel_limit=max(0,int(self.stations[gateway].get("channel_count",self.stations[gateway].get("antenna_count",20))))
                if len(same_gateway)<channel_limit:
                    gain=float(candidate.get("rate_gbps",0));replace_index=None
                else:
                    replace_index,old=min(same_gateway,key=lambda row:float(row[1].get("rate_gbps",0)))
                    gain=float(candidate.get("rate_gbps",0))-float(old.get("rate_gbps",0))
                if gain<=1e-9:continue
                delay=(nearest[satellite]/1000+float(candidate.get("gsl_distance_km",0)))/C_KM_PER_MS
                rank=(gain,-delay,float(candidate.get("elevation_deg",0)))
                if best is None or rank>best[0]:best=(rank,replace_index,candidate)
            if best is None:break
            gain,replace_index,candidate=best[0][0],best[1],dict(best[2])
            candidate["selection_reason"]="容量余量硬约束补强"
            if replace_index is None:selected.append(candidate)
            else:
                used.discard(int(selected[replace_index]["satellite"]));selected[replace_index]=candidate
            used.add(int(candidate["satellite"]));total+=gain
        return selected,total,total+1e-9>=target
    def generate(self,t,positions,links,gsls,satellite_traffic,rain_scenario_percentile=None,rain_grade=None,commit=True,window_rates=None):
        selection_started=time.perf_counter()
        candidates=self._candidates(gsls,t,planning=True,rain_scenario_percentile=rain_scenario_percentile,rain_grade=rain_grade,positions=positions,window_rates=window_rates);demands={int(x[0]):float(x[1])*self.config.traffic_landing_ratio for x in satellite_traffic if float(x[1])>0}
        graph_key=(id(positions),id(links),len(positions),len(links)) if not commit else None
        if graph_key is not None and graph_key==self._offline_graph_key and self._offline_graph is not None:g=self._offline_graph
        else:
            g=_graph(positions,links)
            if graph_key is not None:self._offline_graph_key=graph_key;self._offline_graph=g
        position_by_id={int(row[0]):row for row in positions};priorities={};priority_sources=set()
        station_xyz={gid:_spherical_ecef(station.get("lat",0),station.get("lon",0),station.get("alt",0)) for gid,station in enumerate(self.stations)}
        for candidate in candidates:
            row=position_by_id.get(int(candidate["satellite"]));ground=station_xyz.get(int(candidate["gateway"]))
            if row and ground:
                satellite_xyz=_spherical_ecef(row[2],row[3],row[4]);vector=tuple(satellite_xyz[i]-ground[i] for i in range(3));norm=max(1.,math.sqrt(sum(value*value for value in vector)))
                candidate["los_unit"]=[round(value/norm,9) for value in vector]
                candidate["plane"]=int(row[1]);candidate["latitude_deg"]=float(row[2]);candidate["longitude_deg"]=float(row[3])
        for source in demands:
            row=position_by_id.get(source)
            if row and any(_surface_distance_km(row[2],row[3],region.get("lat",0),region.get("lon",0))<=max(0.,float(region.get("radius_km",0))) for region in self.config.priority_task_regions):
                priorities[source]=max(1.,float(self.config.priority_task_weight));priority_sources.add(source)
        for x in candidates:x["gsl_distance_km"]=_gsl_distance_km(positions[int(x["satellite"])][4],x["elevation_deg"])
        reserve=max(0.,float(self.config.capacity_reserve_ratio))
        planning_demands=({source:value*(1.+reserve) for source,value in demands.items()}
                          if self.config.selection_strategy!="window_margin" else demands)
        chosen=self._select(candidates,g,planning_demands)
        capacity_target=sum(demands.values())*(1.+reserve)
        chosen,selected_capacity,target_satisfied=self._enforce_capacity_target(
            candidates,chosen,g,planning_demands,capacity_target)
        selection_compute_ms=round((time.perf_counter()-selection_started)*1000,3)
        capacities={int(x["satellite"]):float(x["rate_gbps"]) for x in chosen}
        gsl_distances={int(x["satellite"]):float(x.get("gsl_distance_km",0)) for x in chosen}
        routes=[]
        # Offline planning uses one complete bounded search.  The former
        # 8/16/24 sequence rebuilt almost the same multi-source labels three
        # times for every scenario and dominated total planning time.
        routes=_routes_with_capacity_completion(g,capacities,demands,capacities,candidate_count=24,
                       capacity_weight=self.config.capacity_weight,delay_weight=self.config.delay_weight,
                       delay_reference_ms=self.config.delay_reference_ms,
                       feeder_gsl_distances=gsl_distances,
                       max_active_exits_per_source=self.config.planning_exit_limit_per_source,
                       source_priorities=priorities) if chosen else []
        # Joint topology/allocation repair.  Link selection precedes routing for
        # performance, so its demand-coverage estimate can still choose feeder
        # links that are globally capacious but unusable by a hotspot after the
        # exit constraint is applied.  Replace only idle/low-use links at
        # the same gateway with links close to deficient sources, reroute the
        # whole demand set, and keep the trial only when delivered traffic
        # increases.  Same-gateway replacement preserves channel quotas.
        joint_repair_applied=False;joint_repair_score_before=0.;joint_repair_score_after=0.
        def repair_objective(route_rows,feeder_gsl):
            delivered=sum(float(row.get("demand_gbps",0)) for row in route_rows);total=max(sum(demands.values()),1e-12)
            if delivered<=1e-12:return 0.,0.,0.,0.
            delay_rows=[];weighted=0.
            for row in route_rows:
                amount=float(row.get("demand_gbps",0));feeder=int(row.get("feeder",-1))
                latency=(float(row.get("distance_km",0))+float(feeder_gsl.get(feeder,0)))/C_KM_PER_MS
                delay_rows.append((latency,amount));weighted+=latency*amount
            average_delay=weighted/delivered;delay_rows.sort();cumulative=0.;p95_delay=delay_rows[-1][0]
            for latency,amount in delay_rows:
                cumulative+=amount
                if cumulative>=.95*delivered:p95_delay=latency;break
            reference=max(.001,float(self.config.delay_reference_ms));delivery_ratio=min(1.,delivered/total)
            latency_quality=.5*reference/(reference+average_delay)+.5*reference/(reference+p95_delay)
            if self.config.selection_strategy=="legacy" and self.config.capacity_weight==0 and self.config.delay_weight==0:
                return delivery_ratio,delivered,average_delay,p95_delay
            wc,wd=_normalized_objective_weights(self.config.capacity_weight,self.config.delay_weight)
            return wc*delivery_ratio+wd*latency_quality,delivered,average_delay,p95_delay
        # One bounded repair is enough.  A second complete reroute nearly
        # tripled batch planning time while producing only marginal gains.
        for _joint_repair in range(0 if self.config.selection_strategy=="window_margin" else 1):
            allocated={}
            for route in routes:
                source=int(route.get("source",-1));allocated[source]=allocated.get(source,0.)+float(route.get("demand_gbps",0))
            deficient={source:max(0.,float(demand)-allocated.get(source,0.)) for source,demand in demands.items()
                       if float(demand)-allocated.get(source,0.)>1e-6}
            if not deficient:break
            total_gap=sum(deficient.values());nearest=_nearest_demand_distance(g,deficient)
            # The plan already meets its 99% delivery target.  Do not spend a
            # full reroute merely to chase rounding-level residual demand.
            if total_gap<=.01*max(sum(demands.values()),1e-12):break
            used_by_feeder={}
            for route in routes:
                feeder=int(route.get("feeder",-1));used_by_feeder[feeder]=used_by_feeder.get(feeder,0.)+float(route.get("demand_gbps",0))
            selected_satellites={int(item["satellite"]) for item in chosen};wc,wd=_normalized_objective_weights(self.config.capacity_weight,self.config.delay_weight)
            replacements=[]
            for raw in candidates:
                satellite=int(raw["satellite"]);gateway=int(raw["gateway"])
                if satellite in selected_satellites or not math.isfinite(nearest[satellite]):continue
                delay=(nearest[satellite]/1000+float(raw.get("gsl_distance_km",0)))/C_KM_PER_MS
                useful=min(total_gap,max(0.,float(raw.get("rate_gbps",0))))
                score,*_=_marginal_utility(useful,delay,max(sum(demands.values()),1e-12),wc,wd,self.config.delay_reference_ms)
                replacements.append((score,useful,-delay,float(raw.get("elevation_deg",0)),raw))
            replacements.sort(reverse=True,key=lambda row:row[:4]);trial=[dict(item) for item in chosen];protected=set();swap_limit=min(12,max(2,len(deficient)*2));swaps=0
            for _score,_useful,_negative_delay,_elevation,raw in replacements:
                gateway=int(raw["gateway"]);new_rate=float(raw.get("rate_gbps",0));new_satellite=int(raw["satellite"])
                if new_satellite in selected_satellites:continue
                removable=[]
                for index,item in enumerate(trial):
                    if index in protected or int(item["gateway"])!=gateway:continue
                    old_rate=max(1e-12,float(item.get("rate_gbps",0)));old_used=used_by_feeder.get(int(item["satellite"]),0.)
                    if new_rate+1e-9<old_rate:continue
                    old_distance=nearest[int(item["satellite"])] if int(item["satellite"])<len(nearest) else math.inf
                    removable.append((old_used/old_rate,-old_distance,index))
                if not removable:continue
                _utilization,_distance,index=min(removable)
                old_satellite=int(trial[index]["satellite"]);selected_satellites.discard(old_satellite)
                trial[index]=dict(raw);selected_satellites.add(new_satellite);protected.add(index);swaps+=1
                if swaps>=swap_limit:break
            if not swaps:break
            trial_capacities={int(x["satellite"]):float(x["rate_gbps"]) for x in trial}
            trial_gsl={int(x["satellite"]):float(x.get("gsl_distance_km",0)) for x in trial}
            trial_routes=_routes_with_capacity_completion(g,trial_capacities,demands,trial_capacities,candidate_count=24,
                         capacity_weight=self.config.capacity_weight,delay_weight=self.config.delay_weight,
                         delay_reference_ms=self.config.delay_reference_ms,feeder_gsl_distances=trial_gsl,
                         max_active_exits_per_source=self.config.planning_exit_limit_per_source,
                         source_priorities=priorities)
            old_score,old_delivery,_old_average,_old_p95=repair_objective(routes,gsl_distances)
            new_score,new_delivery,_new_average,_new_p95=repair_objective(trial_routes,trial_gsl)
            joint_repair_score_before=old_score;joint_repair_score_after=new_score
            # Only accept a replacement that delivers more traffic. Path
            # latency ranks feasible exits during routing, not this repair.
            if new_delivery<=old_delivery+1e-6 or new_score<=old_score+1e-9:break
            chosen,routes,capacities,gsl_distances=trial,trial_routes,trial_capacities,trial_gsl
            joint_repair_applied=True
        capacity_fallback_used=False
        capacity_fallback_ms=0.
        total_business=sum(demands.values())
        if (self.config.selection_strategy!="window_margin" and
                sum(float(row.get("demand_gbps",0)) for row in routes)<.99*total_business):
            fallback_started=time.perf_counter()
            conservative=self._select(candidates,g,planning_demands,capacity_only=True)
            conservative,conservative_capacity,conservative_target=self._enforce_capacity_target(
                candidates,conservative,g,planning_demands,capacity_target)
            conservative_rates={int(item["satellite"]):float(item["rate_gbps"]) for item in conservative}
            conservative_gsl={int(item["satellite"]):float(item.get("gsl_distance_km",0)) for item in conservative}
            conservative_routes=_routes_with_capacity_completion(
                g,conservative_rates,demands,conservative_rates,candidate_count=24,
                capacity_weight=self.config.capacity_weight,delay_weight=self.config.delay_weight,
                delay_reference_ms=self.config.delay_reference_ms,
                feeder_gsl_distances=conservative_gsl,
                max_active_exits_per_source=self.config.planning_exit_limit_per_source,
                source_priorities=priorities) if conservative else []
            if (sum(float(row.get("demand_gbps",0)) for row in conservative_routes)>
                    sum(float(row.get("demand_gbps",0)) for row in routes)+1e-6):
                chosen,routes,capacities,gsl_distances=conservative,conservative_routes,conservative_rates,conservative_gsl
                selected_capacity,target_satisfied=conservative_capacity,conservative_target
                capacity_fallback_used=True
            capacity_fallback_ms=round((time.perf_counter()-fallback_started)*1000,3)
        _attach_feeder_delay(routes,chosen);avg,bweighted,delay,p95_delay,max_delay=_route_metrics(routes)
        usage=[{"gateway":i,"used":sum(x["gateway"]==i for x in chosen),"available":int(s.get("channel_count",s.get("antenna_count",20)))} for i,s in enumerate(self.stations)]
        routed_business=sum(float(x.get("demand_gbps",0)) for x in routes);total_business=sum(demands.values())
        priority_demand=sum(float(demands.get(source,0)) for source in priority_sources)
        priority_delivered=sum(float(row.get("demand_gbps",0)) for row in routes if int(row.get("source",-1)) in priority_sources)
        delivery_ratio=1.0 if total_business<=1e-12 else min(1.,routed_business/total_business)
        if self.config.selection_strategy=="window_margin":
            load_by_feeder={}
            for row in routes:
                feeder=int(row.get("feeder",-1))
                load_by_feeder[feeder]=load_by_feeder.get(feeder,0.)+float(row.get("demand_gbps",0.))
            for link in chosen:
                load=load_by_feeder.get(int(link["satellite"]),0.)
                link["planned_load_gbps"]=round(load,6)
                link["planned_window_spare_gbps"]=round(max(0.,float(link["rate_gbps"])-load),6)
        plan={"id":"offline_plan","name":"离线站星连接规划","generated_at_s":t,
          "valid_from_s":t,"valid_until_s":t+self.config.planning_horizon_s,
          "slot_s":self.config.planning_slot_s,"slot_count":max(1,math.ceil(self.config.planning_horizon_s/self.config.planning_slot_s)),
          "freeze_until_s":t+self.config.freeze_window_s,"links":chosen,"routes":routes,
          "total_capacity_gbps":round(sum(x["rate_gbps"] for x in chosen),6),"total_demand_gbps":round(sum(demands.values()),6),
          "capacity_reserve_ratio":round(reserve,6),"capacity_target_gbps":round(capacity_target,6),
          "selection_strategy":self.config.selection_strategy,
          "rain_grade":rain_grade or "预测雨衰","rain_scenario_percentile":float(rain_scenario_percentile if rain_scenario_percentile is not None else 1.0),
          "capacity_target_satisfied":bool(target_satisfied),"capacity_target_gap_gbps":round(max(0.,capacity_target-selected_capacity),6),
          "traffic_satellite_count":len(demands),
          "candidate_link_count":len(candidates),"selection_compute_ms":selection_compute_ms,
          "reachable_demand_gbps":round(routed_business,6),"planned_delivery_ratio":round(delivery_ratio,6),
          "delivery_target_ratio":.99,"delivery_target_satisfied":bool(delivery_ratio>=.99-1e-9),
          "delivery_gap_gbps":round(max(0.,total_business-routed_business),6),
          "active_exit_limit_per_source":self.config.planning_exit_limit_per_source,
          "priority_region_count":len(self.config.priority_task_regions),"priority_satellite_count":len(priority_sources),
          "priority_demand_gbps":round(priority_demand,6),"priority_delivered_gbps":round(priority_delivered,6),
          "priority_delivery_ratio":round(1. if priority_demand<=1e-12 else min(1.,priority_delivered/priority_demand),6),
          "gateway_count":len({x["gateway"] for x in chosen}),
          "average_hops":avg,"business_weighted_hops":bweighted,"business_propagation_delay_ms":delay,
          "p95_propagation_delay_ms":p95_delay,"maximum_propagation_delay_ms":max_delay,"latency_scope":"服务卫星至信关站（ISL+馈电GSL）","station_usage":usage}
        plan.update(capacity_weight=round(self.config.capacity_weight,6),delay_weight=round(self.config.delay_weight,6),
                    adaptive_candidate_k=int(getattr(self,"_last_candidate_k",0)),delay_reference_ms=round(self.config.delay_reference_ms,6),
                    selection_method="有界多源最短路 + 容量时延双目标边际优化",
                    objective_formula="wc*(新增落地业务/总需求)+wd*参考时延/(参考时延+路径时延)",
                    objective_capacity_is_hard_constraint=self.config.selection_strategy!="window_margin",
                    objective_gateway_channels_are_hard_constraints=True)
        plan.update(joint_repair_applied=joint_repair_applied,
                    capacity_fallback_used=capacity_fallback_used,
                    capacity_fallback_ms=capacity_fallback_ms,
                    joint_repair_score_before=round(joint_repair_score_before,9),
                    joint_repair_score_after=round(joint_repair_score_after,9))
        if self.config.selection_strategy=="window_margin":
            plan["selection_method"]="five-minute visible links; minimum rate for feasibility, mean rate and path delay for scoring"
            plan["objective_formula"]="wc*normalized(window_mean_rate)+wd*normalized(path_delay_quality); minimum-rate capacity target repair"
        elif self.config.capacity_weight==0 and self.config.delay_weight==0:
            plan["selection_method"]="capacity feasibility, then shortest feasible path"
            plan["objective_formula"]="lexicographic: satisfy capacity and gateway quotas; minimize path delay"
        objective_rows=[x for x in chosen if "selection_score" in x]
        plan["objective_diagnostics"]={
          "selected_signature":[[int(x["gateway"]),int(x["satellite"])] for x in chosen],
          "selected_link_count":len(chosen),
          "selected_rate_gbps":round(sum(float(x.get("rate_gbps",0)) for x in chosen),6),
          "assigned_demand_gbps":round(sum(float(x.get("assigned_demand_gbps",0)) for x in chosen),6),
          "score_min":round(min((float(x.get("selection_score",0)) for x in objective_rows),default=0.),9),
          "score_max":round(max((float(x.get("selection_score",0)) for x in objective_rows),default=0.),9),
          "estimated_delay_min_ms":round(min((float(x.get("estimated_weighted_delay_ms",0)) for x in objective_rows),default=0.),6),
          "estimated_delay_max_ms":round(max((float(x.get("estimated_weighted_delay_ms",0)) for x in objective_rows),default=0.),6)}
        if not commit:return plan
        plans=[plan]
        self.state.generated_at_s=t;self.state.generation+=1;self.state.plans=plans;self.state.execution_plan=plan
        self.state.last_route_update_s=t;self.state.last_route_sources={int(x[0]) for x in satellite_traffic if float(x[1])>0}
        exits={}
        for row in plan.get("routes",[]):exits.setdefault(int(row.get("source",-1)),set()).add(int(row.get("feeder",-1)))
        self.state.source_exits=exits;self.state.source_exit_changed_at={source:t for source in exits}
        self.state.last_fast_epoch=int(math.floor(t/max(self.config.fast_adjust_period_s,1e-6)))
        return plans

    def install_offline_plan(self,plan):
        """Install a precomputed plan without running global optimization."""
        plan=dict(plan);self.state.generation=int(plan.get("generation",self.state.generation+1))
        self.state.plans=[plan];self.state.execution_plan=plan.copy();self.state.generated_at_s=float(plan.get("generated_at_s",0))
        self.state.active_plan_id=str(plan.get("id","offline_plan"));self.state.last_result=None
        self.state.last_route_update_s=float(plan.get("generated_at_s",0))
        self.state.last_route_sources={int(row.get("source",-1)) for row in plan.get("routes",[])}
        exits={}
        for row in plan.get("routes",[]):exits.setdefault(int(row.get("source",-1)),set()).add(int(row.get("feeder",-1)))
        self.state.source_exits=exits;self.state.source_exit_changed_at={source:float(plan.get("generated_at_s",0)) for source in exits}
        # Evaluate a newly installed fixed plan at its first simulation second.
        self.state.last_fast_epoch=-1
        self.state.last_demands={}
        self.state.last_feeder_capacities={}
        self.state.last_capacity_reroute_s=-math.inf

    def _evaluate(self,t,plan,pairs,total_demand,edges,current_traffic_satellite_count=None,positions=None):
        active=[];link_states=[];ideal=sum(x["rate_gbps"] for x in plan["links"]);actual=0
        for old in plan["links"]:
            elev=pairs.get((old["gateway"],old["satellite"]))
            if elev is None:
                link_states.append({"gateway":old["gateway"],"satellite":old["satellite"],"status":"不可见","rate_gbps":0.})
                continue
            grade=plan.get("rain_grade") if plan.get("offline") and self.config.rain_enabled else None
            rain=({"rain_rate_mmh":0.,"actual_attenuation_db":0.,"severity":grade} if grade else
                  link_rain_state(self.stations[int(old["gateway"])],elev,t,self.config.rain_enabled,self.config.rain_sense_sigma,self.config.rain_forecast_hours,self.config.rain_use_p90))
            # Offline scenarios and realtime execution must use the same rain
            # field. Otherwise a light-rain plan can start with dead links.
            grade=plan.get("rain_grade") if plan.get("offline") and self.config.rain_enabled else None
            fade=scenario_attenuation_db(self.stations[int(old["gateway"])],elev,grade,t) if grade else rain["sensed_attenuation_db"]
            altitude=float(positions[int(old["satellite"])][4])/1000 if positions is not None else self.config.reference_satellite_altitude_km
            snr,mcs,rate=link_state(elev,self.config,fade,_slant_distance_km(altitude,elev))
            if rate<=0:
                link_states.append({"gateway":old["gateway"],"satellite":old["satellite"],"status":"容量为零","elevation_deg":elev,"rate_gbps":0.,"rain_attenuation_db":fade})
                continue
            x=old.copy();x.update(elevation_deg=elev,snr_db=snr,modcod=mcs,rate_gbps=rate,gso_interference_db=round(gso_interference_penalty_db(self.config),3),rain_attenuation_db=fade,rain_actual_db=(fade if grade else rain["actual_attenuation_db"]),rain_rate_mmh=rain["rain_rate_mmh"],rain_severity=(grade if grade else rain["severity"]))
            if positions is not None:x["gsl_distance_km"]=_gsl_distance_km(positions[int(old["satellite"])][4],elev)
            active.append(x);actual+=rate
            status="容量下降" if rate<float(old.get("rate_gbps",0))*self.config.capacity_threshold_ratio else "可用"
            link_states.append({"gateway":old["gateway"],"satellite":old["satellite"],"status":status,"elevation_deg":elev,"rate_gbps":rate,"rain_attenuation_db":fade})
        active_feeders={x["satellite"] for x in active}
        feeder_left={int(x["satellite"]):float(x["rate_gbps"]) for x in active}
        gsl_by_feeder={int(x["satellite"]):float(x.get("gsl_distance_km",0)) for x in active}
        xyz_cache={}
        def xyz(satellite):
            satellite=int(satellite)
            if satellite not in xyz_cache:
                row=positions[satellite];xyz_cache[satellite]=_spherical_ecef(row[2],row[3],row[4])
            return xyz_cache[satellite]
        valid=[]
        for route in plan["routes"]:
            feeder=int(route.get("feeder",-1))
            if feeder not in active_feeders or not all((min(a,b),max(a,b)) in edges for a,b in zip(route["path"],route["path"][1:])):continue
            amount=min(float(route.get("demand_gbps",0)),feeder_left.get(feeder,0.))
            if amount<=1e-9:continue
            row=route.copy();row["demand_gbps"]=round(amount,6)
            if positions is not None:
                path=row.get("path",[])
                isl_km=sum(math.dist(xyz(a),xyz(b)) for a,b in zip(path,path[1:]))/1000
                gsl_km=gsl_by_feeder.get(feeder,0.)
                row.update(distance_km=round(isl_km,3),isl_propagation_delay_ms=round(isl_km/C_KM_PER_MS,4),
                           gsl_distance_km=round(gsl_km,3),gsl_propagation_delay_ms=round(gsl_km/C_KM_PER_MS,4),
                           propagation_delay_ms=round((isl_km+gsl_km)/C_KM_PER_MS,4))
            valid.append(row)
            feeder_left[feeder]-=amount
        reachable=sum(x["demand_gbps"] for x in valid)
        if plan["routes"]:
            demand_by_feeder={}
            for route in valid:
                feeder=int(route.get("feeder",-1));demand_by_feeder[feeder]=demand_by_feeder.get(feeder,0.)+float(route.get("demand_gbps",0))
            for x in active:
                x["scheduled_gbps"]=round(min(float(x["rate_gbps"]),demand_by_feeder.get(int(x["satellite"]),0.)),6)
                x["utilization"]=round(x["scheduled_gbps"]/max(x["rate_gbps"],1e-12),6)
            scheduled=min(float(total_demand),sum(float(x["scheduled_gbps"]) for x in active))
        else:
            scheduled=0.
            for x in active:
                x["scheduled_gbps"]=0.
                x["utilization"]=0.
        avg,bweighted,delay,p95_delay,max_delay=_route_metrics(valid)
        routed_sources={int(x.get("source",-1)) for x in valid};denominator=int(current_traffic_satellite_count if current_traffic_satellite_count is not None else plan.get("traffic_satellite_count",0))
        theoretical_deliverable=min(float(total_demand),actual)
        route_capacity_gap=max(0.,theoretical_deliverable-scheduled)
        route_satisfaction=1.0 if theoretical_deliverable<=1e-12 else min(1.,scheduled/theoretical_deliverable)
        return {"id":plan["id"],"name":plan["name"],"ideal_capacity_gbps":round(ideal,6),"actual_capacity_gbps":round(actual,6),
          "capacity_achievement_ratio":round(actual/max(ideal,1e-12),6),"scheduled_gbps":round(scheduled,6),
          "theoretical_deliverable_gbps":round(theoretical_deliverable,6),
          "business_route_satisfaction_ratio":round(route_satisfaction,6),
          "route_capacity_gap_gbps":round(route_capacity_gap,6),
          "unserved_gbps":round(max(0,total_demand-scheduled),6),"active_links":active,"link_states":link_states,"planned_link_count":len(plan["links"]),
          "active_link_count":len(active),"route_available_ratio":round(1.0 if denominator<=0 else min(1.,len(routed_sources)/denominator),6),
          "average_hops":avg,"business_weighted_hops":bweighted,"business_propagation_delay_ms":delay,
          "p95_propagation_delay_ms":p95_delay,"maximum_propagation_delay_ms":max_delay,"latency_scope":"服务卫星至信关站（ISL+馈电GSL）",
          "route_count":len(valid),"routed_source_count":len(routed_sources),"routes":valid}
    def update(self,t,positions,links,gsls,satellite_traffic):
        update_started=time.perf_counter();route_rebuild_ms=0.;adjustment_ms=0.;route_rebuild_mode="无"
        first_plan=not self.state.plans
        if self.config.offline_mode and first_plan:
            raise RuntimeError("尚未加载离线规划方案，请先执行离线规划")
        due=(not self.config.offline_mode) and (first_plan or t<self.state.generated_at_s or t-self.state.generated_at_s>=self.config.planning_period_s)
        planning_reason="首次生成慢周期资源方案" if first_plan else ("到达滚动规划周期" if due else "未到规划周期，沿用当前方案")
        if due:self.generate(t,positions,links,gsls,satellite_traffic)
        evaluation_due=due or self.state.last_result is None or t<self.state.last_evaluation_s or t-self.state.last_evaluation_s>=self.config.fast_evaluation_period_s
        if not evaluation_due:
            result=dict(self.state.last_result);result["planning_event"]=False;result["scheduling_event"]=False;result["planning_reason"]="未到规划周期，沿用当前方案";result["connection_change_count"]=0
            return result
        pairs={(int(g),int(s)):float(e) for g,s,e in gsls};edges={(min(int(a),int(b)),max(int(a),int(b))) for a,b,*_ in links};access_demand=sum(float(x[1]) for x in satellite_traffic);demand=access_demand*self.config.traffic_landing_ratio
        demands={int(x[0]):float(x[1])*self.config.traffic_landing_ratio for x in satellite_traffic if float(x[1])>0}
        current_source_count=len(demands);slow_plan=self.state.plans[0];plan=self.state.execution_plan or slow_plan
        if any(int(row.get("source",-1)) not in demands for row in plan.get("routes",[])):
            plan=plan.copy();plan["routes"]=[row for row in plan.get("routes",[]) if int(row.get("source",-1)) in demands]
            self.state.execution_plan=plan
        active=self._evaluate(t,plan,pairs,demand,edges,current_source_count,positions);action="按慢环规划执行"
        route_broken=bool(plan.get("routes")) and active["route_count"]<len(plan["routes"])
        new_sources=set(demands)-self.state.last_route_sources
        demand_changed=(set(demands)!=set(self.state.last_demands) or any(abs(float(value)-float(self.state.last_demands.get(source,0)))>1e-6 for source,value in demands.items()))
        route_capacity_missing=float(active.get("route_capacity_gap_gbps",0))>1e-3
        current_feeder_capacities={int(row["satellite"]):round(float(row.get("rate_gbps",0)),6)
                                   for row in active.get("active_links",[])}
        feeder_capacity_changed=current_feeder_capacities!=self.state.last_feeder_capacities
        route_needs_repair=route_broken or bool(new_sources) or demand_changed or route_capacity_missing
        graph=None
        fast_epoch=int(math.floor(t/max(self.config.fast_adjust_period_s,1e-6)))
        fast_tick=fast_epoch>self.state.last_fast_epoch
        if fast_tick:self.state.last_fast_epoch=fast_epoch
        if fast_tick and route_needs_repair:
            route_started=time.perf_counter()
            usable_links=list(active.get("active_links",[]))
            capacities={int(x["satellite"]):float(x["rate_gbps"]) for x in usable_links}
            available={feeder:value for feeder,value in capacities.items() if value>1e-12}
            gsl_distances={int(x["satellite"]):float(x.get("gsl_distance_km",0)) for x in usable_links}
            source_left=dict(demands);feeder_left=dict(available);retained=[];existing_exits={}
            # Active feeder satellites land their own traffic directly before
            # any retained ISL route can use that feeder's capacity.
            old_local={int(row["source"]):row for row in active.get("routes",[])
                       if int(row.get("source",-1))==int(row.get("feeder",-2))}
            for source in sorted(set(source_left)&set(feeder_left)):
                amount=min(source_left[source],feeder_left[source])
                if amount<=1e-9:continue
                row=old_local.get(source,{"source":source,"feeder":source,"path":[source],"hops":0,
                    "distance_km":0.,"isl_propagation_delay_ms":0.,"propagation_delay_ms":0.}).copy()
                row["demand_gbps"]=round(amount,6)
                retained.append(row);source_left[source]-=amount;feeder_left[source]-=amount
                existing_exits.setdefault(source,set()).add(source)
            for original in active.get("routes",[]):
                source=int(original.get("source",-1));feeder=int(original.get("feeder",-1))
                amount=min(float(original.get("demand_gbps",0)),source_left.get(source,0.),feeder_left.get(feeder,0.))
                if amount<=1e-9:continue
                row=original.copy();row["demand_gbps"]=round(amount,6);retained.append(row)
                source_left[source]-=amount;feeder_left[feeder]-=amount
                existing_exits.setdefault(source,set()).add(feeder)
            # Business changes every second. Fill the increment on an already
            # valid path before launching any new graph search.
            for row in retained:
                source=int(row["source"]);feeder=int(row["feeder"])
                increment=min(source_left.get(source,0.),feeder_left.get(feeder,0.))
                if increment<=1e-9:continue
                row["demand_gbps"]=round(float(row["demand_gbps"])+increment,6)
                source_left[source]-=increment;feeder_left[feeder]-=increment
            repair_demands={source:amount for source,amount in source_left.items() if amount>1e-6}
            search_capacities={feeder:amount for feeder,amount in feeder_left.items() if amount>1e-6}
            if repair_demands and search_capacities:
                graph=_graph(positions,links)
                components=_component_ids(graph)
                feeder_components={components[feeder] for feeder in search_capacities}
                repair_demands={source:amount for source,amount in repair_demands.items()
                                if components[source] in feeder_components}
                repaired=_repair_routes_local(graph,repair_demands,search_capacities,gsl_distances,
                    1.,0.,self.config.delay_reference_ms,excluded_feeders=existing_exits,
                    exit_limits={source:max(0,self.config.realtime_exit_limit_per_source-len(existing_exits.get(source,set()))) for source in repair_demands},
                    max_visited_per_source=500,max_exits_per_source=self.config.realtime_exit_limit_per_source)
            else:repaired=[]
            new_routes=_compact_routes(retained+repaired,demands,self.config.realtime_exit_limit_per_source)
            plan=plan.copy();plan["routes"]=new_routes
            _attach_feeder_delay(plan["routes"],usable_links)
            self.state.execution_plan=plan;active=self._evaluate(t,plan,pairs,demand,edges,current_source_count,positions)
            self.state.last_route_update_s=t
            self.state.last_route_sources=set(demands)
            self.state.last_demands=dict(demands)
            self.state.last_feeder_capacities=dict(current_feeder_capacities)
            exits={}
            for row in plan.get("routes",[]):exits.setdefault(int(row.get("source",-1)),set()).add(int(row.get("feeder",-1)))
            for source,current in exits.items():
                if current!=self.state.source_exits.get(source):self.state.source_exit_changed_at[source]=t
            self.state.source_exits=exits
            action=("实时寻址：优先补足当前业务容量" if demand_changed or route_capacity_missing
                    else "逐秒修复失效或新增业务路径")
            route_rebuild_ms=(time.perf_counter()-route_started)*1000
        elif route_needs_repair:
            action="检测到路径失效，等待下一秒修复"
        if not route_needs_repair:
            self.state.last_demands=dict(demands)
            self.state.last_feeder_capacities=dict(current_feeder_capacities)
        threshold=self.config.capacity_threshold_ratio
        demand_ratio=active["unserved_gbps"]/max(demand,1e-12) if active else 1.
        # Persistent demand above the installed radio/path capacity is a
        # reported constraint, not a topology emergency. Replanning the same
        # inputs every freeze window produced identical plans at t=30/60 s and
        # stalled the real-time frame. Early replanning is reserved for an
        # actual loss of links or reachability; demand is handled by fast repair
        # and the configured slow planning period.
        # Missing routes and transient capacity loss are repaired by the fast
        # loop.  They must not trigger a multi-second global plan between the
        # configured slow-planning epochs.
        severe=False
        if not due and severe and t-self.state.generated_at_s>=self.config.freeze_window_s:
            self.generate(t,positions,links,gsls,satellite_traffic);slow_plan=self.state.plans[0];plan=self.state.execution_plan or slow_plan
            active=self._evaluate(t,plan,pairs,demand,edges,current_source_count,positions);due=True;planning_reason="到达慢规划周期";action="执行慢周期全局重规划"
        pairs_now={(x["gateway"],x["satellite"]) for x in (active["active_links"] if active else [])}
        change_count=len(pairs_now.symmetric_difference(self.state.last_scheduled_pairs)) if self.state.last_scheduled_pairs else 0;changed=change_count>0
        if changed:self.state.switch_count+=change_count;self.state.last_switch_s=t
        self.state.last_scheduled_pairs=pairs_now
        loss=active["scheduled_gbps"]*self.config.switching_loss_ratio if active and changed else 0
        deployment=self._deployment(active,gsls)
        route_ratio=float(active.get("route_available_ratio",0)) if active else 0.
        business_route_ratio=float(active.get("business_route_satisfaction_ratio",0)) if active else 0.
        capacity_ratio=float(active.get("capacity_achievement_ratio",0)) if active else 0.
        unserved=float(active.get("unserved_gbps",demand)) if active else demand
        rain_affected=sum(float(x.get("rain_actual_db",0))>=3 for x in (active or {}).get("active_links",[]))
        final_routed_sources={int(x.get("source",-1)) for x in (active or {}).get("routes",[])}
        route_exits_by_source={}
        allocated_by_source={}
        for route in (active or {}).get("routes",[]):
            source=int(route.get("source",-1));route_exits_by_source.setdefault(source,set()).add(int(route.get("feeder",-1)))
            allocated_by_source[source]=allocated_by_source.get(source,0.)+float(route.get("demand_gbps",0))
        maximum_active_exits=max((len(value) for value in route_exits_by_source.values()),default=0)
        underallocated_sources=sorted(source for source,value in demands.items() if allocated_by_source.get(source,0.)+1e-6<value)
        underallocated_business=sum(max(0.,float(demands[source])-allocated_by_source.get(source,0.)) for source in underallocated_sources)
        unrouted_sources=sorted(set(demands)-final_routed_sources)
        unrouted_business=sum(float(demands.get(source,0)) for source in unrouted_sources)
        if unrouted_sources:
            if float((active or {}).get("actual_capacity_gbps",0))<demand:unrouted_reason="馈电链路有效容量不足"
            elif not (active or {}).get("active_links"):unrouted_reason="当前无可用馈电出口"
            else:unrouted_reason="候选出口容量碎片或当前ISL路径未完成分配"
        else:unrouted_reason="无"
        if route_ratio<.999:constraint_source="拓扑可达性"
        elif capacity_ratio<.999 and rain_affected:constraint_source="雨衰与星地链路容量"
        elif capacity_ratio<.999 and self.config.geo_interference_enabled:constraint_source="GEO/GSO同频干扰与星地链路容量"
        elif capacity_ratio<.999:constraint_source="星地链路容量"
        elif unserved>.001:constraint_source="业务需求与资源容量"
        else:constraint_source="无显著约束"
        decision_context={
          "access_traffic_gbps":round(access_demand,6),"landing_ratio":round(self.config.traffic_landing_ratio,6),
          "traffic_demand_gbps":round(demand,6),"traffic_satellite_count":len(satellite_traffic),
          "isl_count":len(links),"gsl_candidate_count":len(gsls),
          "visible_feeder_satellite_count":len({int(x[1]) for x in gsls}),
          "gateway_count":len(self.stations),"planned_link_count":len(plan.get("links",[])),
          "active_link_count":int(active.get("active_link_count",0)) if active else 0,
          "geo_interference_enabled":self.config.geo_interference_enabled,"geo_satellite_count":self.config.geo_satellite_count,
          "gso_i_over_n_db":self.config.gso_i_over_n_db,"gso_interference_penalty_db":round(gso_interference_penalty_db(self.config),3),
          "gso_protection_angle_deg":self.config.gso_protection_angle_deg,
          "capacity_weight":round(self.config.capacity_weight,6),"delay_weight":round(self.config.delay_weight,6),
          "adaptive_candidate_k":int(getattr(self,"_last_candidate_k",0)),"delay_reference_ms":round(self.config.delay_reference_ms,6),
          "route_available_ratio":round(route_ratio,6),"capacity_achievement_ratio":round(capacity_ratio,6),
          "business_route_satisfaction_ratio":round(business_route_ratio,6),
          "route_capacity_gap_gbps":round(float((active or {}).get("route_capacity_gap_gbps",0)),6),
          "routed_traffic_satellite_count":len(final_routed_sources),"unrouted_traffic_satellite_count":len(unrouted_sources),
          "maximum_active_exits_per_source":maximum_active_exits,"active_route_count":len((active or {}).get("routes",[])),
          "active_exit_limit_per_source":self.config.realtime_exit_limit_per_source,"underallocated_traffic_satellite_count":len(underallocated_sources),
          "underallocated_business_gbps":round(underallocated_business,6),"underallocated_satellite_ids":underallocated_sources[:100],
          "unrouted_business_gbps":round(unrouted_business,6),"unrouted_satellite_ids":unrouted_sources[:100],
          "unrouted_reason":unrouted_reason,
          "constraint_source":constraint_source,
          "route_rebuild_mode":route_rebuild_mode,"route_rebuild_ms":round(route_rebuild_ms,3),
          "feeder_adjustment_ms":round(adjustment_ms,3),
          "source_exit_hold_s":round(self.config.source_exit_hold_s,3),
          "planner_update_ms":round((time.perf_counter()-update_started)*1000,3)}
        result={"generation":self.state.generation,"generated_at_s":self.state.generated_at_s,"planning_period_s":self.config.planning_period_s,
          "planning_horizon_s":self.config.planning_horizon_s,"planning_slot_s":self.config.planning_slot_s,
          "fast_evaluation_period_s":self.config.fast_evaluation_period_s,"fast_adjust_period_s":self.config.fast_adjust_period_s,
          "route_update_period_s":self.config.route_update_period_s,"freeze_window_s":self.config.freeze_window_s,
          "capacity_threshold_ratio":threshold,"active_plan_id":self.state.active_plan_id,"last_switch_s":self.state.last_switch_s,
          "switch_count":self.state.switch_count,"switching_loss_gbps":round(loss,6),"delivered_gbps":round(min(demand,max(0,(active["scheduled_gbps"] if active else 0)-loss)),6),
          "action":action,"planning_reason":planning_reason,"connection_change_count":change_count,"planning_event":due,"scheduling_event":True,
          "rain_summary":self._rain_summary(active),"station_weather":self._station_weather(t,slow_plan),
          "next_planning_time_s":round(self.state.generated_at_s+self.config.planning_period_s,6),
          "plans":self.state.plans,"evaluations":[active],"active_plan":slow_plan,"execution_plan":plan,
          "active_evaluation":active,"gateway_deployment":deployment,"decision_context":decision_context}
        self.state.last_evaluation_s=t;self.state.last_result=result
        return result
    def _station_weather(self,t,plan):
        # One reference elevation per site makes the displayed weather proxy
        # independent of whichever feeder satellite happens to be selected.
        grade=plan.get("rain_grade") if plan.get("offline") and self.config.rain_enabled else None
        output=[]
        for gid,station in enumerate(self.stations):
            if not self.config.rain_enabled:
                fade=0.
            elif grade:
                fade=scenario_attenuation_db(station,45.,grade,t)
            else:
                rain=link_rain_state(station,45.,t,True,self.config.rain_sense_sigma,self.config.rain_forecast_hours,self.config.rain_use_p90)
                fade=float(rain["sensed_attenuation_db"])
            # Attenuation at one reference elevation is not an observed rain
            # rate.  Keep the selected scenario separate from the local fade.
            output.append({"gateway":gid,"reference_attenuation_db":round(fade,3),
                           "reference_elevation_deg":45.,"source":"模型推算","scenario":grade or "实时雨衰模型",
                           "observed_rain_rate_available":False})
        return output

    def _rain_summary(self,active):
        rows=(active or {}).get("active_links",[])
        values=[float(x.get("rain_actual_db",0)) for x in rows]
        return {"enabled":self.config.rain_enabled,"active_link_count":len(values),
          "affected_link_count":sum(x>=3 for x in values),
          "average_attenuation_db":round(sum(values)/len(values),3) if values else 0.,
          "maximum_attenuation_db":round(max(values),3) if values else 0.}
    def _deployment(self,active,gsls):
        rows=active["active_links"] if active else [];stations=[]
        for gid,s in enumerate(self.stations):
            channels=int(s.get("channel_count",s.get("antenna_count",20)));used=sum(x["gateway"]==gid for x in rows)
            stations.append({"gateway":gid,"name":s.get("name",f"站{gid+1}"),"visible_satellites":sum(int(x[0])==gid for x in gsls),
             "planned_channels":channels,"used_channels":used,"recommended_antennas":max(1,min(channels,used+(1 if used else 0))),"minimum_connection_satisfied":used>0})
        return {"status":"已规划信关站的当前资源状态","station_count":len(stations),
         "active_station_count":sum(x["used_channels"]>0 for x in stations),"minimum_connection_satisfied":all(x["minimum_connection_satisfied"] for x in stations) if stations else False,"stations":stations}
