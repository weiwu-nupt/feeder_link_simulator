"""文档第5.6、5.7和第6章对应的站星规划、实时调度与站点评估。"""
from __future__ import annotations
import csv, heapq, math, time
from collections import deque
from dataclasses import dataclass, field
from typing import Sequence
from .rain_model import link_rain_state, synthetic_rain_attenuation_db, synthetic_surface_rain_rate_mmh, forecast_rain_attenuation_db, forecast_rain_rate_mmh

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
 ModcodRow("32APSK 4/5",13.64,3.951571),
 ModcodRow("QPSK 8/9",6.20,1.766451),ModcodRow("QPSK 9/10",6.42,1.788612),
 ModcodRow("8PSK 8/9",10.69,2.646012),ModcodRow("8PSK 9/10",10.98,2.679207),
 ModcodRow("16APSK 8/9",12.89,3.523143),ModcodRow("16APSK 9/10",13.13,3.567342),
 ModcodRow("32APSK 5/6",14.28,4.119540),ModcodRow("32APSK 8/9",15.69,4.397854),
 ModcodRow("32APSK 9/10",16.05,4.453027))

# ETSI EN 302 307-2 V1.4.1 Table 20a: all normal FECFRAME S2X modes.
# These values describe an ideal linear AWGN channel, without pilot overhead.
_S2X_ROWS="""
QPSK 2/9|-2.85|.434841
QPSK 13/45|-2.03|.567805
QPSK 9/20|.22|.889135
QPSK 11/20|1.45|1.088581
8APSK 5/9-L|4.73|1.647211
8APSK 26/45-L|5.13|1.713601
8PSK 23/36|6.12|1.896173
8PSK 25/36|7.02|2.062148
8PSK 13/18|7.49|2.145136
16APSK 1/2-L|5.97|1.972253
16APSK 8/15-L|6.55|2.104850
16APSK 5/9-L|6.84|2.193247
16APSK 26/45|7.51|2.281645
16APSK 3/5|7.80|2.370043
16APSK 3/5-L|7.41|2.370043
16APSK 28/45|8.10|2.458441
16APSK 23/36|8.38|2.524739
16APSK 2/3-L|8.43|2.635236
16APSK 25/36|9.27|2.745734
16APSK 13/18|9.71|2.856231
16APSK 7/9|10.65|3.077225
16APSK 77/90|11.99|3.386618
32APSK 2/3-L|11.10|3.291954
32APSK 32/45|11.75|3.510192
32APSK 11/15|12.17|3.620536
32APSK 7/9|13.05|3.841226
64APSK 32/45-L|13.98|4.206428
64APSK 11/15|14.81|4.338659
64APSK 7/9|15.47|4.603122
64APSK 4/5|15.87|4.735354
64APSK 5/6|16.55|4.936639
128APSK 3/4|17.73|5.163248
128APSK 7/9|18.53|5.355556
256APSK 29/45-L|16.98|5.065690
256APSK 2/3-L|17.24|5.241514
256APSK 31/45-L|18.10|5.417338
256APSK 32/45|18.59|5.593162
256APSK 11/15-L|18.84|5.768987
256APSK 3/4|19.57|5.900855
"""
S2X_MODCOD_TABLE=tuple(ModcodRow(name,float(snr),float(eta)) for name,snr,eta in (line.split("|") for line in _S2X_ROWS.splitlines() if line))
S2_AND_S2X_MODCOD_TABLE=DEFAULT_MODCOD_TABLE+S2X_MODCOD_TABLE

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
    feeder_link_mode:str="single"
    rain_field:str="synthetic_field"
    # Zero uses antenna hardware limits without an arbitrary inter-slot angle cap.
    max_slot_pointing_change_deg:float=0.
    antenna_steering_type:str="mechanical"
    antenna_az_slew_deg_s:float=12.
    antenna_el_slew_deg_s:float=3.
    antenna_acquisition_s:float=1.
    electronic_switch_ms:float=20.
    electronic_scan_from_zenith_deg:float=60.
    antenna_switch_window_s:float=10.
    standby_rf_transfer_s:float=.02
    antenna_az_accel_deg_s2:float=3.
    antenna_el_accel_deg_s2:float=3.

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

def _station_pointing(station,position):
    ground=_spherical_ecef(station.get("lat",0),station.get("lon",0),station.get("alt",0))
    satellite=_spherical_ecef(position[2],position[3],position[4])
    vector=tuple(satellite[i]-ground[i] for i in range(3))
    norm=max(1.,math.sqrt(sum(value*value for value in vector)))
    latitude=math.radians(float(station.get("lat",0)))
    longitude=math.radians(float(station.get("lon",0)))
    east=-math.sin(longitude)*vector[0]+math.cos(longitude)*vector[1]
    north=-math.sin(latitude)*math.cos(longitude)*vector[0]-math.sin(latitude)*math.sin(longitude)*vector[1]+math.cos(latitude)*vector[2]
    up=math.cos(latitude)*math.cos(longitude)*vector[0]+math.cos(latitude)*math.sin(longitude)*vector[1]+math.sin(latitude)*vector[2]
    return {"los_unit":[round(value/norm,9) for value in vector],
            "azimuth_deg":round(math.degrees(math.atan2(east,north))%360.,3),
            "pointing_elevation_deg":round(math.degrees(math.atan2(up,math.hypot(east,north))),3)}

def _axis_slew_seconds(angle_deg,speed_deg_s,accel_deg_s2):
    angle=max(0.,float(angle_deg));speed=max(.01,float(speed_deg_s));accel=max(.01,float(accel_deg_s2))
    distance_to_accelerate_and_brake=speed*speed/accel
    return (2.*math.sqrt(angle/accel) if angle<=distance_to_accelerate_and_brake
            else 2.*speed/accel+(angle-distance_to_accelerate_and_brake)/speed)

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

def _component_delivery_bound(node_count,edges,demands,active_links):
    """Upper bound on deliverable traffic with disconnected ISL components."""
    parent=list(range(node_count))
    def root(node):
        while parent[node]!=node:
            parent[node]=parent[parent[node]];node=parent[node]
        return node
    for first,second in edges:
        a=root(int(first));b=root(int(second))
        if a!=b:parent[b]=a
    demand_by_component={};capacity_by_component={}
    for source,amount in demands.items():
        component=root(int(source))
        demand_by_component[component]=demand_by_component.get(component,0.)+max(0.,float(amount))
    for link in active_links:
        component=root(int(link["satellite"]))
        capacity_by_component[component]=capacity_by_component.get(component,0.)+max(0.,float(link.get("rate_gbps",0)))
    return sum(min(demand,capacity_by_component.get(component,0.))
               for component,demand in demand_by_component.items())

def _feeder_resources(links):
    """Aggregate parallel gateway links without losing their capacity."""
    capacities={};distances={}
    for link in links:
        satellite=int(link["satellite"])
        rate=max(0.,float(link.get("rate_gbps",0.)))
        capacities[satellite]=capacities.get(satellite,0.)+rate
        distance=float(link.get("gsl_distance_km",0.))
        distances[satellite]=min(distances.get(satellite,distance),distance)
    return capacities,distances

def _bind_routes_to_links(routes,links):
    """Assign each feeder-satellite route to real gateway links and their capacities."""
    by_satellite={};spare={};used={}
    for link in links:
        satellite=int(link["satellite"])
        key=(int(link["gateway"]),satellite,int(link.get("feeder_port",1)))
        by_satellite.setdefault(satellite,[]).append(link)
        spare[key]=max(0.,float(link.get("rate_gbps",0.)))
        used[key]=0.
    bound=[]
    for route in routes:
        satellite=int(route.get("feeder",-1))
        preferred=route.get("gateway")
        options=sorted(by_satellite.get(satellite,()),key=lambda link:(
            0 if preferred is not None and int(link["gateway"])==int(preferred) else 1,
            float(link.get("gsl_distance_km",0.))))
        left=max(0.,float(route.get("demand_gbps",0.)))
        priority_left=max(0.,float(route.get("priority_gbps",0.)))
        for link in options:
            key=(int(link["gateway"]),satellite,int(link.get("feeder_port",1)))
            amount=min(left,spare[key])
            if amount<=1e-9:continue
            item=route.copy();item["demand_gbps"]=round(amount,6)
            item["gateway"]=key[0];item["feeder_port"]=key[2]
            gsl_km=float(link.get("gsl_distance_km",0.))
            item["gsl_distance_km"]=round(gsl_km,3)
            item["gsl_propagation_delay_ms"]=round(gsl_km/C_KM_PER_MS,4)
            item["propagation_delay_ms"]=round((float(item.get("distance_km",0.))+gsl_km)/C_KM_PER_MS,4)
            if "priority_gbps" in item:
                item["priority_gbps"]=round(min(amount,priority_left),6)
                priority_left=max(0.,priority_left-amount)
            bound.append(item);spare[key]-=amount;used[key]+=amount;left-=amount
            if left<=1e-9:break
    return bound,used

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

def _priority_delivery(route_rows,priority_demands):
    carried={}
    for row in route_rows:
        source=int(row.get("source",-1));carried[source]=carried.get(source,0.)+float(row.get("demand_gbps",0.))
    return sum(min(float(amount),carried.get(int(source),0.)) for source,amount in (priority_demands or {}).items())

def _routes(g,feeders,demands,feeder_capacities=None,candidate_count=8,capacity_weight=1.,delay_weight=0.,delay_reference_ms=15.,feeder_gsl_distances=None,max_active_exits_per_source=2,source_exit_limits=None,source_excluded_feeders=None,source_priorities=None,priority_demands=None,max_isl_distance_km=None):
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
    priority_demands=priority_demands or {}
    def allocate(source,target,force_self=False,priority_mode=False):
        requested=max(0.,float(demands.get(source,0.)));target=min(max(0.,float(target)),residual.get(source,0.));available_labels=set(labels[source])
        exit_limit=max(0,min(max_active_exits_per_source,int(source_exit_limits.get(int(source),max_active_exits_per_source))))
        excluded={int(value) for value in source_excluded_feeders.get(int(source),())}
        while target>1e-12 and available_labels:
            options=[]
            usable_capacity={label:max(0.,remaining.get(feeders[label],0.)) for label in available_labels
                             if feeders[label] not in excluded}
            for label in available_labels:
                distance=labels[source][label];feeder=feeders[label];available=remaining.get(feeder,0.)
                if max_isl_distance_km is not None and (distance/1000-float(feeder_gsl_distances.get(feeder,0)))>max_isl_distance_km:continue
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
            elif priority_mode:
                # Fill nearby exits first, even when none can carry the whole
                # priority demand. The remaining demand can use another exit.
                choice=max(options,key=lambda row:(row[5],row[6]))
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
    # Reserve capacity for traffic originating in priority grid cells first.
    # A satellite may carry both priority and ordinary traffic.
    for source in sorted(residual,key=lambda value:float(priority_demands.get(int(value),0.)),reverse=True):
        allocate(source,min(residual[source],max(0.,float(priority_demands.get(int(source),0.)))),priority_mode=True)
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
                                     feeder_gsl_distances=None,max_active_exits_per_source=2,source_priorities=None,priority_demands=None,
                                     source_exit_limits=None):
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
                   max_active_exits_per_source=max_active_exits_per_source,source_exit_limits=source_exit_limits,
                   source_priorities=source_priorities,priority_demands=priority_demands)
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
                         len(exits_by_source.get(int(source),set()))<int((source_exit_limits or {}).get(int(source),max_active_exits_per_source))}
    # Expand only over feeders that still have capacity. Usually one pass is
    # sufficient; the bound of three prevents pathological work while allowing
    # an augmenting assignment to escape local capacity fragmentation.
    for _pass in range(3):
        if not residual_capacities or not residual_demands:break
        exit_limits={source:int((source_exit_limits or {}).get(source,max_active_exits_per_source))-len(exits_by_source.get(source,set()))
                     for source in residual_demands}
        extra=_routes(g,residual_capacities,residual_demands,residual_capacities,
                      candidate_count=min(12,len(residual_capacities)),
                      capacity_weight=capacity_weight,delay_weight=delay_weight,
                      delay_reference_ms=delay_reference_ms,feeder_gsl_distances=feeder_gsl_distances,
                      max_active_exits_per_source=max_active_exits_per_source,
                      source_exit_limits=exit_limits,source_excluded_feeders=exits_by_source,source_priorities=source_priorities,
                      priority_demands={source:max(0.,float(priority_demands.get(source,0.))-allocated_by_source.get(source,0.)) for source in residual_demands} if priority_demands else None)
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
                          if value>1e-9 and len(exits_by_source.get(source,set()))<int((source_exit_limits or {}).get(source,max_active_exits_per_source))}
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
                         source_exit_limits=source_exit_limits,
                         source_priorities=source_priorities,
                         priority_demands={source:max(0.,float((priority_demands or {}).get(source,0.))) for source in deficient}) if available else []
        old_amount=sum(float(route.get("demand_gbps",0)) for route in routes)
        new_routes=retained+rerouted
        new_amount=sum(float(route.get("demand_gbps",0)) for route in new_routes)
        if new_amount<=old_amount+1e-6 or _priority_delivery(new_routes,priority_demands)+1e-6<_priority_delivery(routes,priority_demands):break
        routes=new_routes
    return routes

def _repair_routes_local(g,demands,feeder_capacities,feeder_gsl_distances=None,
                         capacity_weight=1.,delay_weight=0.,delay_reference_ms=15.,
                         excluded_feeders=None,exit_limits=None,max_visited_per_source=2000,
                         max_exits_per_source=2,priority_demands=None,local_feeders_by_source=None,
                         component_ids=None,search_cache=None):
    """Repair affected sources with bounded source-local Dijkstra searches."""
    remaining={int(k):max(0.,float(v)) for k,v in feeder_capacities.items() if float(v)>1e-9}
    feeder_set=set(remaining);gsl=feeder_gsl_distances or {};excluded_feeders=excluded_feeders or {};exit_limits=exit_limits or {}
    # A depleted feeder must stop attracting searches.  With more demand than
    # capacity, the old loop searched 500 nodes for every remaining source.
    if component_ids is None:component_ids=_component_ids(g)
    component_spare={}
    component_feeder_count={}
    for feeder,capacity in remaining.items():
        component=component_ids[feeder]
        component_spare[component]=component_spare.get(component,0.)+capacity
        component_feeder_count[component]=component_feeder_count.get(component,0)+1
    total=max(sum(max(0.,float(v)) for v in demands.values()),1e-12);result=[]
    for source,requested in sorted(demands.items(),key=lambda item:(float((priority_demands or {}).get(int(item[0]),0.)),int(item[0]) in feeder_set,item[1]),reverse=True):
        source=int(source);left=max(0.,float(requested));limit=max(0,min(max_exits_per_source,int(exit_limits.get(source,max_exits_per_source))))
        if left<=1e-9 or limit<=0 or component_spare.get(component_ids[source],0.)<=1e-9:continue
        found=[];excluded=set(excluded_feeders.get(source,()))
        local_feeders=set((local_feeders_by_source or {}).get(source,set())) & remaining.keys()
        target_count=min(max(12,limit),component_feeder_count[component_ids[source]])
        # A complete route can end the search once no unseen feeder can have
        # lower total ISL+GSL delay. This preserves the selected exits while
        # avoiding twelve-feeder searches for every affected traffic source.
        # Bound retained searches: a full 500-node predecessor map for every
        # traffic source can consume hundreds of MB on a large constellation.
        use_cache=search_cache is not None and ((source,max_visited_per_source) in search_cache or len(search_cache)<256)
        if not use_cache:
            minimum_gsl_km=(min((float(gsl.get(feeder,0.)) for feeder in remaining
                if component_ids[feeder]==component_ids[source] and feeder not in excluded),default=0.)
                if not local_feeders else 0.)
            distances={source:0.};previous={source:-1};queue=[(0.,source)];visited=0
            best_full_delay_km=math.inf
            while queue and visited<max_visited_per_source and (len(found)<target_count or
                    (local_feeders and not any(feeder in local_feeders for _,feeder in found))):
                distance,node=heapq.heappop(queue)
                if distance!=distances.get(node):continue
                visited+=1
                if node in feeder_set and node not in excluded and remaining.get(node,0)>1e-9:
                    found.append((distance,node))
                    if not local_feeders and remaining[node]>=left-1e-9:
                        best_full_delay_km=min(best_full_delay_km,distance/1000+float(gsl.get(node,0.)))
                for neighbor,edge_distance in g[node]:
                    candidate=distance+edge_distance
                    if candidate<distances.get(neighbor,math.inf):distances[neighbor]=candidate;previous[neighbor]=node;heapq.heappush(queue,(candidate,neighbor))
                if best_full_delay_km<math.inf and (not queue or
                    queue[0][0]/1000+minimum_gsl_km>best_full_delay_km+1e-9):break
        else:
            cache_key=(source,max_visited_per_source)
            cached=search_cache.get(cache_key)
            if cached is None:
                distances={source:0.};previous={source:-1};queue=[(0.,source)];walked=[]
                while queue and len(walked)<max_visited_per_source:
                    distance,node=heapq.heappop(queue)
                    if distance!=distances.get(node):continue
                    walked.append((distance,node))
                    for neighbor,edge_distance in g[node]:
                        candidate=distance+edge_distance
                        if candidate<distances.get(neighbor,math.inf):
                            distances[neighbor]=candidate;previous[neighbor]=node
                            heapq.heappush(queue,(candidate,neighbor))
                cached=(walked,previous)
                search_cache[cache_key]=cached
            walked,previous=cached
            for distance,node in walked:
                if node in feeder_set and node not in excluded and remaining.get(node,0)>1e-9:
                    found.append((distance,node))
                if len(found)>=target_count and (not local_feeders or
                        any(feeder in local_feeders for _,feeder in found)):break
        options=[]
        for distance,feeder in found:
            carried=min(left,remaining.get(feeder,0.));delay=(distance/1000+float(gsl.get(feeder,0)))/C_KM_PER_MS
            options.append((1 if feeder in local_feeders else 0,1 if feeder==source else 0,-delay,carried,feeder,distance))
        if options:
            best_delay=min(-row[2] for row in options)
            # A nearby ground station is not useful when reaching its feeder
            # satellite requires a much longer ISL path.
            options=[(int(row[0] and -row[2]<=max(best_delay*1.5,best_delay+3.0)),*row[1:])
                     for row in options]
        for _regional_priority,_local_priority,_negative_delay,_initial_carried,feeder,distance in sorted(options,reverse=True)[:limit]:
            carried=min(left,remaining.get(feeder,0.))
            if carried<=1e-9:continue
            path=[];node=feeder
            while node>=0:path.append(node);node=previous.get(node,-1)
            path.reverse()
            if not path or path[0]!=source:continue
            delay=(distance/1000+float(gsl.get(feeder,0)))/C_KM_PER_MS
            utility,*_=_marginal_utility(carried,delay,total,capacity_weight,delay_weight,delay_reference_ms)
            result.append({"source":source,"feeder":feeder,"path":path,"hops":len(path)-1,
                "priority_local_exit":feeder in local_feeders,
                "distance_km":round(distance/1000,3),"isl_propagation_delay_ms":round(distance/1000/C_KM_PER_MS,4),
                "propagation_delay_ms":round(delay,4),"demand_gbps":round(carried,6),
                "requested_demand_gbps":round(float(requested),6),"route_objective_score":round(utility,9)})
            remaining[feeder]-=carried;left-=carried
            component_spare[component_ids[feeder]]-=carried
            if remaining[feeder]<=1e-9:
                del remaining[feeder]
                component_feeder_count[component_ids[feeder]]-=1
            if left<=1e-9:break
    return result

def _rebalance_priority_to_local(routes,demands,priority_demands,local_feeders_by_source,
                                 feeder_capacities,graph,gsl_distances,max_exits,component_ids=None):
    """Move priority traffic onto spare nearby exits without reducing delivery."""
    updated=[row.copy() for row in routes]
    used={}
    routes_by_source={}
    for row in updated:
        feeder=int(row["feeder"]);used[feeder]=used.get(feeder,0.)+float(row["demand_gbps"])
        routes_by_source.setdefault(int(row["source"]),[]).append(row)
    for source,priority in sorted(priority_demands.items(),key=lambda item:item[1],reverse=True):
        nearby=set(local_feeders_by_source.get(source,()))
        if not nearby or priority<=1e-9:continue
        own=routes_by_source.get(source,[])
        local_amount=sum(float(row["demand_gbps"]) for row in own if int(row["feeder"]) in nearby)
        deficit=max(0.,float(priority)-local_amount)
        if deficit<=1e-6:continue
        remote=[row for row in own if int(row["feeder"]) not in nearby]
        if not remote:continue
        spare={feeder:max(0.,float(feeder_capacities.get(feeder,0))-used.get(feeder,0.))
               for feeder in nearby}
        spare={feeder:amount for feeder,amount in spare.items() if amount>1e-6}
        if not spare:continue
        amount=min(deficit,sum(float(row["demand_gbps"]) for row in remote),sum(spare.values()))
        if amount<=1e-6:continue
        if component_ids is None:component_ids=_component_ids(graph)
        candidates=_repair_routes_local(graph,{source:amount},spare,gsl_distances,
            max_visited_per_source=500,max_exits_per_source=1,
            priority_demands={source:priority},local_feeders_by_source={source:nearby},
            component_ids=component_ids)
        if not candidates:continue
        candidate=candidates[0];moved=float(candidate["demand_gbps"])
        feeder=int(candidate["feeder"])
        if feeder not in nearby or moved<=1e-6:continue
        current_delay=min(float(row.get("propagation_delay_ms",math.inf)) for row in remote)
        candidate_delay=float(candidate.get("propagation_delay_ms",math.inf))
        if candidate_delay>max(current_delay*1.5,current_delay+3.0):continue
        exits={int(row["feeder"]) for row in own}
        if feeder not in exits and len(exits)>=max_exits:
            whole=next((row for row in remote if float(row["demand_gbps"])<=moved+1e-6),None)
            if whole is None:continue
            moved=float(whole["demand_gbps"])
            remote=[whole]
        left=moved
        for row in remote:
            take=min(left,float(row["demand_gbps"]))
            row["demand_gbps"]=round(float(row["demand_gbps"])-take,6)
            used[int(row["feeder"])]=max(0.,used[int(row["feeder"])]-take)
            left-=take
            if left<=1e-6:break
        candidate["demand_gbps"]=round(moved-left,6)
        updated.append(candidate);routes_by_source.setdefault(source,[]).append(candidate)
        used[feeder]=used.get(feeder,0.)+moved-left
    return _compact_routes(updated,demands,max_exits)

def _gsl_distance_km(altitude_m,elevation_deg):
    radius=EARTH_RADIUS_M;orbit_radius=radius+max(0.,float(altitude_m));elevation=math.radians(float(elevation_deg))
    return (math.sqrt(max(0.,orbit_radius*orbit_radius-(radius*math.cos(elevation))**2))-radius*math.sin(elevation))/1000

def _attach_feeder_delay(routes,links):
    feeder_links={int(x["satellite"]):x for x in links}
    pair_links={(int(x["gateway"]),int(x["satellite"]),int(x.get("feeder_port",1))):x for x in links}
    for route in routes:
        key=(int(route.get("gateway",-1)),int(route.get("feeder",-1)),int(route.get("feeder_port",1)))
        link=pair_links.get(key) or feeder_links.get(int(route.get("feeder",-1)))
        gsl_distance=float((link or {}).get("gsl_distance_km",0))
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
    grouped_by_source={}
    for (source,_feeder),row in grouped.items():
        grouped_by_source.setdefault(source,[]).append(row)
    result=[]
    for source in demands:
        remaining=max(0.,float(demands[source]))
        rows=sorted(grouped_by_source.get(source,()),
            key=lambda x:(not bool(x.get("priority_local_exit",False)),float(x.get("propagation_delay_ms",math.inf))))[:max_exits_per_source]
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
    last_transition_phase:int=-1
    last_invalid_route_count:int=0
    last_invalid_route_keys:set=field(default_factory=set)
    last_route_sources:set=field(default_factory=set)
    source_exits:dict=field(default_factory=dict)
    source_exit_changed_at:dict=field(default_factory=dict)
    last_fast_epoch:int=-1
    last_demands:dict=field(default_factory=dict)
    last_feeder_capacities:dict=field(default_factory=dict)
    last_capacity_reroute_s:float=-math.inf
    switched_pairs:set=field(default_factory=set)
    switch_effective_s:float=-math.inf
    switch_unavailable_until:dict=field(default_factory=dict)

class SlowFastLoopPlanner:
    """慢环生成一套规划骨架，快环逐帧执行和局部修复。"""
    def __init__(self,stations,config=None):
        self.stations=list(stations);self.config=config or PlannerConfig();self.state=PlanningState()
        self._offline_graph_key=None;self._offline_graph=None
    def _gateway_limit(self,gateway):
        station=self.stations[int(gateway)]
        channels=max(0,int(station.get("channel_count",station.get("antenna_count",20))))
        return min(channels,max(0,int(station.get("antenna_count",channels))))
    def _handover_window(self,old,carried_by_feeder,slot_seconds):
        normal=max(0.,float(self.config.antenna_switch_window_s))
        if old is None or carried_by_feeder.get(int(old["satellite"]),0.)>0.01:
            return normal
        # An unloaded antenna may spend part of this slot acquiring its next
        # target. Its physical slew time remains an actual unavailable period.
        return max(normal,min(30.,max(0.,float(slot_seconds))/2.))
    def _regional_feeders(self,links,plan=None):
        by_satellite={}
        for link in links:by_satellite.setdefault(int(link["satellite"]),[]).append(link)
        result={}
        preferred_by_region=(plan or {}).get("priority_feeders_by_region") or {}
        for region in self.config.priority_task_regions:
            name=str(region.get("name",""))
            preferred={int(satellite) for satellite in preferred_by_region.get(name,[])}
            usable=preferred.intersection(by_satellite)
            if not usable and by_satellite:
                distances={satellite:min(_surface_distance_km(
                    self.stations[int(link["gateway"])]["lat"],self.stations[int(link["gateway"])]["lon"],
                    region["lat"],region["lon"]) for link in rows)
                    for satellite,rows in by_satellite.items()}
                nearest=min(distances.values())
                usable={satellite for satellite,distance in distances.items() if distance<=nearest+100.}
            result[name]=usable
        return result
    def _standby_prepoint_seconds(self):
        if self.config.antenna_steering_type=="electronic":
            return max(0.,float(self.config.electronic_switch_ms))/1000.
        return max(_axis_slew_seconds(180.,self.config.antenna_az_slew_deg_s,self.config.antenna_az_accel_deg_s2),
                   _axis_slew_seconds(90.,self.config.antenna_el_slew_deg_s,self.config.antenna_el_accel_deg_s2))+max(0.,float(self.config.antenna_acquisition_s))
    def _standby_available(self,gateway,previous_plan,reference_time_s):
        if not previous_plan:return False
        antennas=max(0,int(self.stations[gateway].get("antenna_count",self._gateway_limit(gateway))))
        previous_count=sum(int(row["gateway"])==gateway for row in previous_plan.get("links",[]))
        available_s=max(0.,float(reference_time_s)-float(previous_plan.get("valid_from_s",previous_plan.get("generated_at_s",reference_time_s))))
        return antennas>self._gateway_limit(gateway) and previous_count<antennas and self._standby_prepoint_seconds()<=available_s+1e-9
    def _candidates(self,gsls,t,planning=False,rain_scenario_percentile=None,rain_grade=None,positions=None,window_rates=None):
        out=[]
        for gid,sid,elev in gsls:
            rain=({"rain_rate_mmh":0.,"actual_attenuation_db":0.,"sensed_attenuation_db":0.,"planning_attenuation_db":0.,"severity":rain_grade}
                  if self.config.rain_field in ("synthetic_field","forecast") else
                  link_rain_state(self.stations[int(gid)],float(elev),t,self.config.rain_enabled,self.config.rain_sense_sigma,self.config.rain_forecast_hours,self.config.rain_use_p90))
            statistical_fade=rain["planning_attenuation_db"] if planning else rain["sensed_attenuation_db"]
            if rain_grade and self.config.rain_enabled and self.config.rain_field=="forecast":
                fade=forecast_rain_attenuation_db(self.stations[int(gid)],float(elev),t)
            elif self.config.rain_enabled and self.config.rain_field=="synthetic_field":
                fade=synthetic_rain_attenuation_db(self.stations[int(gid)],float(elev),rain_grade or "中雨",t,list(positions[int(sid)]) if positions is not None else None)
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
            if self.config.rain_field in ("synthetic_field","forecast"):statistical_fade=fade
            if rate>0:out.append({"gateway":int(gid),"satellite":int(sid),"elevation_deg":float(elev),"snr_db":snr,
                                  "modcod":mcs,"rate_gbps":rate,"connection":1,"array_mode":1,"power_factor":1.0,
                                  "window_min_rate_gbps":rate if planning and self.config.selection_strategy=="window_margin" else None,
                                  "window_mean_rate_gbps":window_state.get("mean_rate_gbps") if planning and self.config.selection_strategy=="window_margin" and window_rates is not None else None,
                                  "window_worst_elevation_deg":window_state.get("worst_elevation_deg") if planning and self.config.selection_strategy=="window_margin" and window_rates is not None else None,
                                  "window_worst_time_s":window_state.get("worst_time_s") if planning and self.config.selection_strategy=="window_margin" and window_rates is not None else None,
                                  "gso_interference_db":round(gso_interference_penalty_db(self.config),3),
                                  "rain_attenuation_db":fade,"rain_statistical_db":statistical_fade,
                                  "rain_scenario_percentile":rain_scenario_percentile,"rain_actual_db":rain["actual_attenuation_db"],
                                  "rain_rate_mmh":forecast_rain_rate_mmh(self.stations[int(gid)],t) if rain_grade and self.config.rain_enabled and self.config.rain_field=="forecast" else synthetic_surface_rain_rate_mmh(self.stations[int(gid)],rain_grade or "中雨",t) if self.config.rain_enabled and self.config.rain_field=="synthetic_field" else rain["rain_rate_mmh"],"rain_severity":self.stations[int(gid)].get("rain_grade") or rain_grade or rain["severity"]})
        return out
    def _select(self,candidates,g,demands,capacity_only=False,priority_demands=None):
        component_demand=_component_demand(g,demands)
        component_by_node=_component_ids(g)
        if demands:candidates=[x for x in candidates if component_demand[x["satellite"]]>0]
        if not candidates:return []
        capacity_weight,delay_weight=_normalized_objective_weights(self.config.capacity_weight,self.config.delay_weight)
        total_demand=max(sum(demands.values()),1e-12);delay_reference=max(.001,float(self.config.delay_reference_ms))
        def select_with_k(k):
            options=_k_nearest_candidate_options(g,candidates,demands,k)
            remaining={i:self._gateway_limit(i) for i,_s in enumerate(self.stations)}
            unmet={int(source):float(value) for source,value in demands.items()}
            priority_unmet={int(source):min(float(value),unmet.get(int(source),0.)) for source,value in (priority_demands or {}).items()}
            selected=[];used={};available=set(range(len(candidates)))
            while available and sum(remaining.values())>0 and sum(unmet.values())>1e-12:
                evaluated=[]
                for index in tuple(available):
                    item=candidates[index];gateway=int(item["gateway"]);satellite=int(item["satellite"])
                    if remaining.get(gateway,0)<=0 or used.get(satellite,0)>=(2 if self.config.feeder_link_mode=="dual" else 1):continue
                    capacity=max(0.,float(item["rate_gbps"]));delivered=0.;weighted_delay=0.;allocation=[];assigned={};priority_gain=0.;priority_weighted_delay=0.
                    for priority_pass in (True,False):
                        for source,_demand,delay in options[index]:
                            left=max(0.,unmet.get(source,0.)-assigned.get(source,0.))
                            if priority_pass:left=min(left,priority_unmet.get(source,0.))
                            amount=min(left,capacity-delivered)
                            if amount<=0:continue
                            delivered+=amount;weighted_delay+=amount*delay;allocation.append((source,amount,delay))
                            assigned[source]=assigned.get(source,0.)+amount
                            if priority_pass:priority_gain+=amount;priority_weighted_delay+=amount*delay
                            if delivered>=capacity-1e-12:break
                        if delivered>=capacity-1e-12:break
                    if delivered<=0:continue
                    average_delay=weighted_delay/delivered
                    score,capacity_score,delay_gain,delay_score=_marginal_utility(
                        delivered,average_delay,total_demand,capacity_weight,delay_weight,delay_reference)
                    diversity,minimum_angle,plane_bonus=_candidate_diversity(item,selected,self.config.feeder_diversity_target_deg)
                    evaluated.append((delivered,score,diversity,minimum_angle,plane_bonus,index,allocation,average_delay,capacity_score,delay_gain,delay_score,priority_gain,priority_weighted_delay))
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
                if sum(priority_unmet.values())>1e-9:
                    eligible=evaluated
                elif self.config.selection_strategy in ("window_margin","slot_tracking") and not capacity_only:
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
                if sum(priority_unmet.values())>1e-9:
                    # Nearly equal priority throughput should land through the
                    # shorter gateway path. An exact max of priority gain made
                    # tiny rate differences dominate kilometres of detour.
                    best_priority_gain=max(row[11] for row in eligible)
                    if best_priority_gain>1e-9:
                        near_capacity=[row for row in eligible if row[11]+1e-9>=.95*best_priority_gain]
                        chosen_row=min(near_capacity,key=lambda row:(row[12]/max(row[11],1e-12)-
                            (.2 if candidates[row[5]].get("retained_feeder") and
                                candidates[row[5]].get("next_slot_visible") else 0.),-row[11],-row[0]))
                    else:chosen_row=max(eligible,key=lambda row:(row[0],-row[7]))
                elif self.config.selection_strategy in ("window_margin","slot_tracking") and not capacity_only:
                    margins=[max(0.,float(candidates[row[5]].get("window_mean_rate_gbps") or candidates[row[5]]["rate_gbps"])) for row in eligible]
                    qualities=[row[10] for row in eligible]
                    min_margin,max_margin=min(margins),max(margins)
                    min_quality,max_quality=min(qualities),max(qualities)
                    for row,margin,quality in zip(eligible,margins,qualities):
                        margin_norm=(margin-min_margin)/(max_margin-min_margin) if max_margin-min_margin>1e-9 else .5
                        delay_norm=(quality-min_quality)/(max_quality-min_quality) if max_quality-min_quality>1e-9 else .5
                        # Prefer a feeder that remains usable beyond this slot.
                        # Continuing a feeder that expires at the next boundary
                        # must not win solely because it avoids one handover.
                        utility=capacity_weight*margin_norm+delay_weight*delay_norm
                        if candidates[row[5]].get("next_slot_visible"):utility+=.12
                        if candidates[row[5]].get("retained_feeder"):
                            utility+=.12 if candidates[row[5]].get("next_slot_visible") else -.12
                        if self.config.selection_strategy=="slot_tracking":utility+=.15*row[2]
                        margin_scores[row[5]]=(utility,margin_norm,delay_norm,margin)
                    chosen_row=max(eligible,key=lambda row:(margin_scores[row[5]][0],row[0],-row[7]))
                elif self.config.selection_strategy=="legacy" and self.config.capacity_weight==0 and self.config.delay_weight==0:
                    chosen_row=(min(eligible,key=lambda row:(row[7],-row[0])) if feasible and not capacity_only
                                else max(eligible,key=lambda row:(row[0],-row[7])))
                else:
                    chosen_row=max(eligible,key=lambda row:(row[1]+diversity_weight*row[2],
                        row[3]>=self.config.feeder_min_separation_deg,row[1],-row[7]))
                delivered,score,diversity,minimum_angle,plane_bonus,index,allocation,average_delay,capacity_score,delay_gain,delay_score,priority_gain,_priority_weighted_delay=chosen_row
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
                item["feeder_port"]=used.get(int(item["satellite"]),0)+1
                if self.config.feeder_link_mode=="dual":
                    item["polarization"]="RHCP" if item["feeder_port"]==1 else "LHCP"
                item.update(selection_score=round(rank[0],9),capacity_score=round(capacity_score,9),delay_score=round(delay_score,9),
                            delay_gain=round(delay_gain,9),
                            estimated_weighted_delay_ms=round(average_delay,6),assigned_demand_gbps=round(sum(x[1] for x in allocation),6),
                            diversity_score=round(diversity,6),minimum_angular_separation_deg=round(minimum_angle,3),
                             orbital_plane_bonus=round(plane_bonus,3),priority_assigned_gbps=round(priority_gain,6),selection_reason=("重点业务优先、可行出口就近" if priority_gain>1e-9 else "五分钟平均容量与时延加权" if self.config.selection_strategy=="window_margin" else "容量可行集内优先时延"))
                if window_capacity is not None:item["estimated_window_capacity_gbps"]=round(window_capacity,6)
                selected.append(item);remaining[item["gateway"]]-=1
                used[item["satellite"]]=used.get(item["satellite"],0)+1
                for source,amount,_delay in allocation:
                    unmet[source]=max(0.,unmet.get(source,0.)-amount)
                    priority_unmet[source]=max(0.,priority_unmet.get(source,0.)-amount)
                if used[item["satellite"]]>=(2 if self.config.feeder_link_mode=="dual" else 1):
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
        k=min(8 if self.config.selection_strategy in ("window_margin","slot_tracking") else 6,feeder_count)
        best_selected,best_unmet,_can_expand,unmet=select_with_k(k);self._last_candidate_k=k
        # K-nearest pruning must never leave usable station resources idle while
        # landing demand remains. Fill residual quotas with the best remaining
        # capacity/latency candidates using one demand-to-network distance pass.
        if best_unmet>1e-9 and self.config.feeder_link_mode!="dual" and self.config.selection_strategy not in ("window_margin","slot_tracking"):
            used={int(item["satellite"]) for item in best_selected}
            remaining={i:max(0,self._gateway_limit(i)-sum(int(x["gateway"])==i for x in best_selected)) for i,_s in enumerate(self.stations)}
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
        used={satellite:sum(int(item["satellite"])==satellite for item in selected)
              for satellite in {int(item["satellite"]) for item in selected}}
        while total+1e-9<target:
            best=None
            for candidate in candidates:
                satellite=int(candidate["satellite"]);gateway=int(candidate["gateway"])
                if used.get(satellite,0)>=(2 if self.config.feeder_link_mode=="dual" else 1) or component_demand[satellite]<=0:continue
                same_gateway=[(index,item) for index,item in enumerate(selected) if int(item["gateway"])==gateway]
                channel_limit=self._gateway_limit(gateway)
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
            candidate["feeder_port"]=used.get(int(candidate["satellite"]),0)+1
            if self.config.feeder_link_mode=="dual":
                candidate["polarization"]="RHCP" if candidate["feeder_port"]==1 else "LHCP"
            if replace_index is None:selected.append(candidate)
            else:
                old_satellite=int(selected[replace_index]["satellite"])
                used[old_satellite]=max(0,used.get(old_satellite,0)-1)
                selected[replace_index]=candidate
            sat=int(candidate["satellite"]);used[sat]=used.get(sat,0)+1;total+=gain
        return selected,total,total+1e-9>=target
    def _retain_slot_links(self,chosen,candidates,previous_plan,graph,demands,capacity_target,reference_time_s,next_slot_visible_pairs=None):
        """Start each slot from usable feeders and limit voluntary handovers."""
        if not previous_plan:return chosen
        by_pair={(int(row["gateway"]),int(row["satellite"])):row for row in candidates}
        previous={}
        selected={}
        for row in previous_plan.get("links",[]):previous.setdefault(int(row["gateway"]),[]).append(row)
        for row in chosen:selected.setdefault(int(row["gateway"]),[]).append(row)
        nearest=_nearest_demand_distance(graph,demands)
        component_ids=_component_ids(graph)
        carried_by_feeder={};priority_by_feeder={}
        for route in previous_plan.get("routes",[]):
            feeder=int(route.get("feeder",-1));amount=max(0.,float(route.get("demand_gbps",0.)))
            carried_by_feeder[feeder]=carried_by_feeder.get(feeder,0.)+amount
            priority_by_feeder[feeder]=priority_by_feeder.get(feeder,0.)+min(amount,max(0.,float(route.get("priority_gbps",0.))))
        def regional(row):
            return any(_surface_distance_km(row.get("latitude_deg",0),row.get("longitude_deg",0),
                region["lat"],region["lon"])<=float(region["radius_km"])
                for region in self.config.priority_task_regions)
        used=set();result=[]
        for gateway in sorted(set(previous)|set(selected)):
            quota=self._gateway_limit(gateway)
            prior=previous.get(gateway,[])
            standby_ready=self._standby_available(gateway,previous_plan,reference_time_s)
            current=selected.get(gateway,[])
            selected_satellites={int(row["satellite"]) for row in current}
            desired_local={}
            for row in current:
                if regional(row):
                    component=component_ids[int(row["satellite"])]
                    desired_local[component]=min(2,desired_local.get(component,0)+1)
            # The optimizer has already compared demand, delay, visibility and
            # continuity. Rebuilding the slot from every visible old feeder
            # discards that result and creates a strong directional lock-in.
            retained=[]
            for row in current:
                satellite=int(row["satellite"])
                if satellite in used or len(retained)>=quota:continue
                retained.append(row)
                used.add(satellite)
            for old in sorted(prior,key=lambda row:float(row.get("rate_gbps",0)),reverse=True):
                satellite=int(old["satellite"])
                keep=by_pair.get((gateway,satellite))
                if keep is None or satellite in used or len(retained)>=quota:continue
                retained.append(dict(keep,selection_reason="continue tracking"))
                used.add(satellite)
            additions=[row for row in current if int(row["satellite"]) not in used]
            additions.sort(key=lambda row:(bool(row.get("next_slot_visible")),
                float(row.get("rate_gbps",0))),reverse=True)
            while additions and len(retained)<quota:
                row=additions.pop(0)
                if int(row["satellite"]) in used:continue
                retained.append(row);used.add(int(row["satellite"]))
            # The greedy selector may fill every channel with continuing links.
            # Still inspect a small set of feasible successors for idle antennas.
            seen_additions={int(row["satellite"]) for row in additions}
            successor_pool=[row for row in candidates if int(row["gateway"])==gateway
                and int(row["satellite"]) not in used and int(row["satellite"]) not in seen_additions]
            def successor_rank(row):
                return (bool(row.get("next_slot_visible")),
                    -(nearest[int(row["satellite"])]/1000+float(row.get("gsl_distance_km",0)))/C_KM_PER_MS,
                    float(row.get("rate_gbps",0)))
            successors=[]
            for sector in range(4):
                sector_rows=(row for row in successor_pool if int(float(row.get("azimuth_deg",0))%360//90)==sector)
                successors.extend(sorted(sector_rows,key=successor_rank,reverse=True)[:4])
            successors.extend(sorted(successor_pool,key=successor_rank,reverse=True)[:8])
            successors=list({int(row["satellite"]):row for row in successors}.values())
            additions.extend(successors)
            # A feeder that cannot cover the following slot must be replaced
            # before it reaches the current slot's visibility boundary.
            def direction_sector(row):
                return int(float(row.get("azimuth_deg", 0.)) % 360. // 90)
            def sector_load(rows):
                counts=[0, 0, 0, 0]
                for item in rows:
                    counts[direction_sector(item)]+=1
                return counts
            # Inspect underrepresented directions first. This only changes
            # replacement order; feasibility and traffic checks still apply.
            loads=sector_load(retained)
            additions.sort(key=lambda row:(loads[direction_sector(row)],
                not bool(row.get("next_slot_visible")),
                -float(row.get("rate_gbps", 0.))))
            expiring={int(row["satellite"]) for row in retained
                      if next_slot_visible_pairs is not None and
                      (gateway,int(row["satellite"])) not in next_slot_visible_pairs}
            for new in additions:
                if int(new["satellite"]) in used:continue
                options=[]
                for index,keep in enumerate(retained):
                    if (int(keep["gateway"]),int(keep["satellite"])) not in by_pair:continue
                    old=next((row for row in prior if int(row["satellite"])==int(keep["satellite"])),None)
                    if old is None:continue
                    local_old=any(_surface_distance_km(keep.get("latitude_deg",0),keep.get("longitude_deg",0),
                        region["lat"],region["lon"])<=float(region["radius_km"])
                        for region in self.config.priority_task_regions)
                    local_new=any(_surface_distance_km(new.get("latitude_deg",0),new.get("longitude_deg",0),
                        region["lat"],region["lon"])<=float(region["radius_km"])
                        for region in self.config.priority_task_regions)
                    new_component=component_ids[int(new["satellite"])]
                    old_component=component_ids[int(keep["satellite"])]
                    retained_local={}
                    for item in retained:
                        if regional(item):
                            component=component_ids[int(item["satellite"])]
                            retained_local[component]=retained_local.get(component,0)+1
                    regional_gap=(local_new and retained_local.get(new_component,0)<
                        desired_local.get(new_component,0))
                    if regional_gap and local_old and (old_component==new_component or
                        retained_local.get(old_component,0)<=desired_local.get(old_component,0)):
                        continue
                    if local_old and not local_new:continue
                    delta_az=abs((float(new.get("azimuth_deg",0))-float(old.get("azimuth_deg",0))+180.)%360.-180.)
                    delta_el=abs(float(new.get("pointing_elevation_deg",new.get("elevation_deg",0)))-
                                 float(old.get("pointing_elevation_deg",old.get("elevation_deg",0))))
                    switch_s=(max(0.,float(self.config.standby_rf_transfer_s)) if standby_ready else
                        max(0.,float(self.config.electronic_switch_ms))/1000. if self.config.antenna_steering_type=="electronic" else
                        max(_axis_slew_seconds(delta_az,self.config.antenna_az_slew_deg_s,self.config.antenna_az_accel_deg_s2),
                            _axis_slew_seconds(delta_el,self.config.antenna_el_slew_deg_s,self.config.antenna_el_accel_deg_s2))+
                        self.config.antenna_acquisition_s)
                    slot_seconds=float(previous_plan.get("valid_until_s",reference_time_s+60.))-float(previous_plan.get("valid_from_s",reference_time_s))
                    if switch_s>self._handover_window(old,carried_by_feeder,slot_seconds)+1e-9:continue
                    old_volume=float(keep["rate_gbps"])*slot_seconds
                    new_volume=float(new["rate_gbps"])*max(0.,slot_seconds-switch_s)
                    old_delay=(nearest[int(keep["satellite"])]/1000+float(keep.get("gsl_distance_km",0)))/C_KM_PER_MS
                    # Compare both exits with the same demand-to-feeder metric.
                    # A selection-stage weighted delay is not comparable with
                    # the old exit's nearest-demand delay.
                    new_delay=(nearest[int(new["satellite"])]/1000+
                        float(new.get("gsl_distance_km",0)))/C_KM_PER_MS
                    carried=carried_by_feeder.get(int(keep["satellite"]),0.)
                    priority_carried=priority_by_feeder.get(int(keep["satellite"]),0.)
                    future_gain=(next_slot_visible_pairs is not None and
                        (gateway,int(keep["satellite"])) not in next_slot_visible_pairs and
                        (gateway,int(new["satellite"])) in next_slot_visible_pairs)
                    early_handover=(int(keep["satellite"]) in expiring and
                        future_gain and float(new["rate_gbps"])>=carried and
                        new_volume>=old_volume*.75 and new_delay<=old_delay+(2. if priority_carried>0. else 3.))
                    idle_reposition=(carried<=0.01 and priority_carried<=1e-6 and bool(new.get("next_slot_visible")) and
                        float(new["rate_gbps"])>=float(keep["rate_gbps"])*.8 and
                        (future_gain and new_delay<=old_delay+1. or new_delay+1.<old_delay))
                    selected_upgrade=(carried<=0.01 and priority_carried<=1e-6 and int(new["satellite"]) in selected_satellites and
                        bool(new.get("next_slot_visible")) and
                        float(new["rate_gbps"])>=max(carried,float(keep["rate_gbps"])*.75) and
                        new_delay<=old_delay+1.5)
                    others=[row for j,row in enumerate(retained) if j!=index]
                    old_spread=min((_candidate_angle_deg(keep,row) for row in others),default=180.)
                    new_spread=min((_candidate_angle_deg(new,row) for row in others),default=180.)
                    spread_upgrade=(carried<=0.01 and priority_carried<=1e-6 and
                        bool(new.get("next_slot_visible")) and new_spread>old_spread+20. and
                        float(new["rate_gbps"])>=float(keep["rate_gbps"])*.8 and
                        new_delay<=old_delay+1.)
                    loads=sector_load(retained)
                    old_sector=direction_sector(keep)
                    new_sector=direction_sector(new)
                    direction_upgrade=(carried<=0.01 and priority_carried<=1e-6 and
                        old_sector!=new_sector and loads[old_sector]>=loads[new_sector]+3 and
                        bool(new.get("next_slot_visible")) and
                        float(new["rate_gbps"])>=float(keep["rate_gbps"])*.85 and
                        new_delay<=old_delay+1.5 and new_volume>=old_volume*.75)
                    worthwhile=(direction_upgrade or spread_upgrade or selected_upgrade or idle_reposition or early_handover or regional_gap or new_volume>old_volume*1.20 or
                        (new_delay+(2. if priority_carried>0. else 1.5)<old_delay and
                         new_volume>=old_volume*.95 and float(new["rate_gbps"])>=carried) or
                        (local_new and not local_old and new_volume>=old_volume*.90))
                    if worthwhile:options.append((int(early_handover or idle_reposition or selected_upgrade or spread_upgrade or direction_upgrade),
                        loads[old_sector]-loads[new_sector] if direction_upgrade else
                        new_spread-old_spread if spread_upgrade else -priority_carried*switch_s,
                        new_volume-old_volume+carried*(old_delay-new_delay),index,keep))
                if not options:continue
                _urgent,_priority_loss,_gain,index,old=max(options,key=lambda item:item[:3])
                used.discard(int(old["satellite"]));retained[index]=new
                used.add(int(new["satellite"]))
                expiring.discard(int(old["satellite"]))
            result.extend(retained)
        return result

    def _assign_slot_antennas(self,chosen,previous_plan,reference_time_s,fixed_assignments=None,graph=None):
        """Match each new link to a physical antenna with a bounded pointing step."""
        previous={}
        for row in (previous_plan or {}).get("links",[]):
            gateway=int(row["gateway"])
            previous.setdefault(gateway,{})[int(row.get("antenna_id",len(previous.get(gateway,{}))))]=row
        component_ids=_component_ids(graph) if graph is not None else None
        previous_routes=(previous_plan or {}).get("routes",[])
        carried_by_feeder={};priority_by_feeder={};used_by_feeder={}
        for route in previous_routes:
            feeder=int(route.get("feeder",-1));amount=max(0.,float(route.get("demand_gbps",0.)))
            carried_by_feeder[feeder]=carried_by_feeder.get(feeder,0.)+amount
            priority_by_feeder[feeder]=priority_by_feeder.get(feeder,0.)+min(amount,max(0.,float(route.get("priority_gbps",0.))))
            used_by_feeder[feeder]=used_by_feeder.get(feeder,0.)+amount
        def donor_risk(old,prior):
            satellite=int(old["satellite"]);carried=carried_by_feeder.get(satellite,0.)
            priority=priority_by_feeder.get(satellite,0.)
            if component_ids is None or satellite>=len(component_ids):
                return (priority,carried,carried)
            component=component_ids[satellite]
            spare=sum(max(0.,float(other.get("rate_gbps",0.))-used_by_feeder.get(int(other["satellite"]),0.))
                for other in prior.values() if int(other["satellite"])!=satellite
                and int(other["satellite"])<len(component_ids)
                and component_ids[int(other["satellite"])]==component)
            return (max(0.,priority-spare),max(0.,carried-spare),priority,carried)
        max_angle=max(0.,float(self.config.max_slot_pointing_change_deg))
        switch_window=max(0.,float(self.config.antenna_switch_window_s))
        slot_seconds=(float(previous_plan.get("valid_until_s",reference_time_s+60.))-
            float(previous_plan.get("valid_from_s",reference_time_s))) if previous_plan else 60.
        electronic=self.config.antenna_steering_type=="electronic"
        def steering(old,row):
            old_az=float(old.get("azimuth_deg",0));new_az=float(row.get("azimuth_deg",0))
            delta_az=abs((new_az-old_az+180.)%360.-180.)
            delta_el=abs(float(row.get("pointing_elevation_deg",row.get("elevation_deg",0)))-
                         float(old.get("pointing_elevation_deg",old.get("elevation_deg",0))))
            seconds=(max(0.,float(self.config.electronic_switch_ms))/1000. if electronic else
                max(_axis_slew_seconds(delta_az,self.config.antenna_az_slew_deg_s,self.config.antenna_az_accel_deg_s2),
                    _axis_slew_seconds(delta_el,self.config.antenna_el_slew_deg_s,self.config.antenna_el_accel_deg_s2))+
                max(0.,float(self.config.antenna_acquisition_s)))
            return delta_az,delta_el,seconds
        accepted=[];dropped=0;changes=0;largest=0.
        gateways=sorted({int(row["gateway"]) for row in chosen})
        for gateway in gateways:
            rows=[dict(row) for row in chosen if int(row["gateway"])==gateway]
            prior=previous.get(gateway,{})
            prior_antenna_by_satellite={(int(old["satellite"]),int(old.get("feeder_port",1))):antenna_id
                                        for antenna_id,old in prior.items()}
            active_limit=self._gateway_limit(gateway)
            antenna_count=max(0,int(self.stations[gateway].get("antenna_count",active_limit)))
            has_spare_antennas=antenna_count>active_limit
            # Without a stored parked attitude, use a full-range mechanical
            # traverse as the preparation bound. It must fit in the preceding
            # slot while the old feeder is still carrying traffic.
            prepoint_s=self._standby_prepoint_seconds()
            standby_ready=self._standby_available(gateway,previous_plan,reference_time_s)
            options=[]
            for row in rows:
                choices=[]
                if electronic and 90.-float(row.get("pointing_elevation_deg",row.get("elevation_deg",0)))>float(self.config.electronic_scan_from_zenith_deg)+1e-9:
                    options.append(choices);continue
                for antenna_id in range(antenna_count):
                    # A continuing link belongs to the same antenna throughout
                    # its pass; it cannot be reassigned as a fresh acquisition.
                    continuing_antenna=prior_antenna_by_satellite.get((int(row["satellite"]),int(row.get("feeder_port",1))))
                    if continuing_antenna is not None and antenna_id!=continuing_antenna:continue
                    old=prior.get(antenna_id)
                    if old is None:
                        if has_spare_antennas and previous_plan and not standby_ready:
                            continue
                        # An idle antenna can point during the preceding slot.
                        choices.append((1 if has_spare_antennas else 2,0.,antenna_id,None));continue
                    a=old.get("los_unit");b=row.get("los_unit")
                    if not a or not b:continue
                    cosine=max(-1.,min(1.,sum(float(x)*float(y) for x,y in zip(a,b))))
                    angle=math.degrees(math.acos(cosine))
                    changed=int(old["satellite"])!=int(row["satellite"])
                    delta_az,delta_el,seconds=steering(old,row)
                    if not changed and old.get("previous_slot_los_unit"):
                        prior_los=old["previous_slot_los_unit"]
                        tracking_dot=max(-1.,min(1.,sum(float(x)*float(y) for x,y in zip(prior_los,b))))
                        angle=math.degrees(math.acos(tracking_dot))
                        delta_az=abs((float(row["azimuth_deg"])-float(old["previous_slot_azimuth_deg"])+180.)%360.-180.)
                        delta_el=abs(float(row["pointing_elevation_deg"])-float(old["previous_slot_elevation_deg"]))
                    slot_seconds=float(previous_plan.get("valid_until_s",reference_time_s+60.))-float(previous_plan.get("valid_from_s",reference_time_s))
                    tracking_ok=(delta_az/max(.01,float(self.config.antenna_az_slew_deg_s))<=slot_seconds+1e-9 and
                                 delta_el/max(.01,float(self.config.antenna_el_slew_deg_s))<=slot_seconds+1e-9)
                    allowed_window=self._handover_window(old,carried_by_feeder,slot_seconds)
                    if (not max_angle or angle<=max_angle+1e-9) and ((not changed and (electronic or tracking_ok)) or
                                                                   (changed and seconds<=allowed_window+1e-9)):
                        choices.append((0 if int(old["satellite"])==int(row["satellite"])
                                        else 2 if has_spare_antennas else 1,angle,antenna_id,old))
                options.append(sorted(choices,key=lambda item:(item[0],item[1],item[2])))
            # A continuing satellite keeps its physical antenna.  A generic
            # bipartite matcher can otherwise permute identical links at every
            # slot boundary and report spurious handovers and outages.
            locked={}
            locked_antennas=set()
            for index,row in enumerate(rows):
                fixed_antenna=(fixed_assignments or {}).get((gateway,int(row["satellite"]),int(row.get("feeder_port",1))))
                if fixed_antenna is None and self.config.feeder_link_mode!="dual":
                    fixed_antenna=(fixed_assignments or {}).get((gateway,int(row["satellite"])))
                if fixed_antenna is not None and any(item[2]==fixed_antenna for item in options[index]):
                    locked[index]=fixed_antenna
                    locked_antennas.add(fixed_antenna)
                    continue
                for rank,_angle,antenna_id,_old in options[index]:
                    if rank==0 and antenna_id not in locked_antennas:
                        locked[index]=antenna_id
                        locked_antennas.add(antenna_id)
                        break
            owner={antenna_id:index for index,antenna_id in locked.items()}
            def match(index,seen):
                for _rank,_angle,antenna_id,_old in options[index]:
                    if antenna_id in locked_antennas:continue
                    if antenna_id in seen:continue
                    seen.add(antenna_id)
                    if antenna_id not in owner or match(owner[antenna_id],seen):
                        owner[antenna_id]=index;return True
                return False
            order=sorted(range(len(rows)),key=lambda index:(any(item[0]==0 for item in options[index]),float(rows[index].get("rate_gbps",0))))
            for index in order:
                if index not in locked:match(index,set())
            assignments={index:antenna_id for antenna_id,index in owner.items()}
            # Reserve the idle, prepointed antennas for the old feeder exits
            # whose traffic has the least room to move to another active exit.
            # Move a hard switch onto a lower-risk, otherwise unused old
            # antenna when that pointing move is physically feasible. The
            # high-risk old antenna is then free to keep its feeder alive
            # until a prepointed standby antenna takes over its channel.
            if has_spare_antennas and previous_plan and any(
                    antenna_id not in prior for antenna_id in assignments.values()):
                unused_old=set(prior)-set(assignments.values())
                hard_rows=[index for index,antenna_id in assignments.items() if antenna_id in prior
                    and int(prior[antenna_id]["satellite"])!=int(rows[index]["satellite"])
                    and index not in locked]
                hard_rows.sort(key=lambda index:donor_risk(prior[assignments[index]],prior),reverse=True)
                for hard_index in hard_rows:
                    old_antenna=assignments[hard_index]
                    old_risk=donor_risk(prior[old_antenna],prior)
                    eligible=[antenna_id for antenna_id in unused_old
                        if donor_risk(prior[antenna_id],prior)<old_risk
                        and any(item[2]==antenna_id for item in options[hard_index])]
                    if not eligible:continue
                    replacement=min(eligible,key=lambda antenna_id:donor_risk(prior[antenna_id],prior))
                    assignments[hard_index]=replacement
                    unused_old.remove(replacement)
                    unused_old.add(old_antenna)
            accepted_at_gateway=0
            for index,row in enumerate(rows):
                if index not in assignments or accepted_at_gateway>=active_limit:
                    dropped+=1;continue
                antenna_id=assignments[index];old=prior.get(antenna_id)
                if old is None:
                    row.update(antenna_id=antenna_id,pointing_change_deg=None,
                        handover_status="备用天线预指向完成" if has_spare_antennas and previous_plan else "备用天线接入" if previous_plan else "初始接入")
                    if has_spare_antennas and previous_plan:
                        row.update(prepoint_start_s=round(reference_time_s-prepoint_s,3),
                            prepoint_ready_s=round(reference_time_s,3),prepoint_duration_s=round(prepoint_s,3))
                else:
                    angle=next(item[1] for item in options[index] if item[2]==antenna_id)
                    changed=int(old["satellite"])!=int(row["satellite"])
                    changes+=int(changed);largest=max(largest,angle)
                    delta_az,delta_el,seconds=steering(old,row)
                    if not changed and old.get("previous_slot_azimuth_deg") is not None:
                        delta_az=abs((float(row["azimuth_deg"])-float(old["previous_slot_azimuth_deg"])+180.)%360.-180.)
                        delta_el=abs(float(row["pointing_elevation_deg"])-float(old["previous_slot_elevation_deg"]))
                    row.update(antenna_id=antenna_id,pointing_change_deg=round(angle,3),
                        previous_satellite=int(old["satellite"]),handover_status="切星" if changed else "持续跟踪",
                        previous_azimuth_deg=old.get("azimuth_deg") if changed else old.get("previous_slot_azimuth_deg",old.get("azimuth_deg")),
                        previous_elevation_deg=old.get("pointing_elevation_deg") if changed else old.get("previous_slot_elevation_deg",old.get("pointing_elevation_deg")),
                        pointing_reference_time_s=reference_time_s,
                        azimuth_change_deg=round(delta_az,3),elevation_change_deg=round(delta_el,3),
                        estimated_switch_s=round(seconds,3) if changed else None,
                        switch_window_s=self._handover_window(old,carried_by_feeder,slot_seconds),
                        switch_feasible=bool(not changed or seconds<=self._handover_window(old,carried_by_feeder,slot_seconds)+1e-9))
                accepted.append(row)
                accepted_at_gateway+=1
            if has_spare_antennas and previous_plan:
                gateway_rows=[row for row in accepted if int(row["gateway"])==gateway]
                current_satellites={int(row["satellite"]) for row in gateway_rows}
                assigned_antennas={int(row["antenna_id"]) for row in gateway_rows}
                departing=[old for old in prior.values() if int(old["satellite"]) not in current_satellites
                    and int(old["antenna_id"]) not in assigned_antennas]
                departing.sort(key=lambda old:donor_risk(old,prior),reverse=True)
                standby=[row for row in gateway_rows if row.get("prepoint_ready_s") is not None]
                standby.sort(key=lambda row:float(row.get("rate_gbps",0)),reverse=True)
                for row,old in zip(standby,departing):
                    row.update(handover_from_antenna_id=int(old["antenna_id"]),
                        previous_satellite=int(old["satellite"]),
                        standby_protected_demand_gbps=round(carried_by_feeder.get(int(old["satellite"]),0.),6),
                        standby_protected_priority_gbps=round(priority_by_feeder.get(int(old["satellite"]),0.),6),
                        estimated_switch_s=round(max(0.,float(self.config.standby_rf_transfer_s)),3),
                        switch_window_s=switch_window,switch_feasible=True)
                    changes+=1
        return accepted,{"handover_count":changes,"maximum_pointing_change_deg":round(largest,3),
            "unassignable_link_count":dropped,"pointing_limit_deg":max_angle,
            "antenna_steering_type":self.config.antenna_steering_type,"switch_window_s":switch_window,
            "antenna_az_slew_deg_s":self.config.antenna_az_slew_deg_s,
            "antenna_el_slew_deg_s":self.config.antenna_el_slew_deg_s,
            "antenna_az_accel_deg_s2":self.config.antenna_az_accel_deg_s2,
            "antenna_el_accel_deg_s2":self.config.antenna_el_accel_deg_s2,
            "antenna_acquisition_s":self.config.antenna_acquisition_s,
            "electronic_scan_from_zenith_deg":self.config.electronic_scan_from_zenith_deg}

    def generate(self,t,positions,links,gsls,satellite_traffic,rain_scenario_percentile=None,rain_grade=None,commit=True,window_rates=None,priority_traffic=None,priority_traffic_regions=None,previous_plan=None,next_slot_visible_pairs=None):
        selection_started=time.perf_counter()
        candidates=self._candidates(gsls,t,planning=True,rain_scenario_percentile=rain_scenario_percentile,rain_grade=rain_grade,positions=positions,window_rates=window_rates);demands={int(x[0]):float(x[1])*self.config.traffic_landing_ratio for x in satellite_traffic if float(x[1])>0}
        graph_key=(id(positions),id(links),len(positions),len(links)) if not commit else None
        if graph_key is not None and graph_key==self._offline_graph_key and self._offline_graph is not None:g=self._offline_graph
        else:
            g=_graph(positions,links)
            if graph_key is not None:self._offline_graph_key=graph_key;self._offline_graph=g
        position_by_id={int(row[0]):row for row in positions};priorities={};priority_sources=set()
        for candidate in candidates:
            row=position_by_id.get(int(candidate["satellite"]))
            if row is not None:
                candidate.update(_station_pointing(self.stations[int(candidate["gateway"])],row))
                candidate["plane"]=int(row[1]);candidate["latitude_deg"]=float(row[2]);candidate["longitude_deg"]=float(row[3])
            if self.config.selection_strategy=="slot_tracking":
                candidate["next_slot_visible"]=(int(candidate["gateway"]),int(candidate["satellite"])) in (next_slot_visible_pairs or ())
        if self.config.selection_strategy=="slot_tracking" and previous_plan:
            # Compare both targets at this slot boundary. The stored old
            # pointing belongs to the previous slot start (60 seconds ago).
            previous_plan=dict(previous_plan)
            previous_plan["links"]=[dict(old,previous_slot_los_unit=old.get("los_unit"),
                previous_slot_azimuth_deg=old.get("azimuth_deg"),
                previous_slot_elevation_deg=old.get("pointing_elevation_deg"),
                **_station_pointing(self.stations[int(old["gateway"])],position_by_id[int(old["satellite"])]))
                if int(old["satellite"]) in position_by_id else dict(old)
                for old in previous_plan.get("links",[])]
        if self.config.selection_strategy=="slot_tracking" and self.config.antenna_steering_type=="electronic":
            candidates=[candidate for candidate in candidates if
                90.-float(candidate.get("pointing_elevation_deg",candidate.get("elevation_deg",0)))<=
                float(self.config.electronic_scan_from_zenith_deg)+1e-9]
        if self.config.selection_strategy=="slot_tracking" and previous_plan:
            prior_by_gateway={}
            for row in previous_plan.get("links",[]):prior_by_gateway.setdefault(int(row["gateway"]),[]).append(row)
            carried_by_feeder={}
            for route in previous_plan.get("routes",[]):
                feeder=int(route.get("feeder",-1))
                carried_by_feeder[feeder]=carried_by_feeder.get(feeder,0.)+max(0.,float(route.get("demand_gbps",0.)))
            slot_seconds=float(previous_plan.get("valid_until_s",t+60.))-float(previous_plan.get("valid_from_s",t))
            standby_ready_by_gateway={gateway:self._standby_available(gateway,previous_plan,t)
                for gateway in prior_by_gateway}
            retained_pairs={(int(row["gateway"]),int(row["satellite"]))
                            for row in previous_plan.get("links",[])}
            for candidate in candidates:
                candidate["retained_feeder"]=(int(candidate["gateway"]),int(candidate["satellite"])) in retained_pairs
            def direction_counts(rows):
                counts={"east":0,"west":0,"other":0}
                for row in rows:
                    az=float(row.get("azimuth_deg",0.))%360.
                    counts["east" if 45.<=az<135. else "west" if 225.<=az<315. else "other"]+=1
                return counts
            direction_before=direction_counts(candidates)
            limit=max(0.,float(self.config.max_slot_pointing_change_deg))
            filtered=[]
            for candidate in candidates:
                gateway=int(candidate["gateway"]);prior=prior_by_gateway.get(gateway,[])
                if len(prior)<self._gateway_limit(gateway):filtered.append(candidate);continue
                if not candidate["retained_feeder"] and standby_ready_by_gateway.get(gateway,False):
                    filtered.append(candidate);continue
                for old in prior:
                    if not old.get("los_unit") or not candidate.get("los_unit"):continue
                    angle=math.degrees(math.acos(max(-1.,min(1.,sum(float(a)*float(b) for a,b in zip(old["los_unit"],candidate["los_unit"]))))))
                    if limit and angle>limit+1e-9:continue
                    if int(old["satellite"])==int(candidate["satellite"]):
                        if self.config.antenna_steering_type=="electronic":filtered.append(candidate);break
                        delta_az=abs((float(candidate.get("azimuth_deg",0))-float(old.get("azimuth_deg",0))+180.)%360.-180.)
                        delta_el=abs(float(candidate.get("pointing_elevation_deg",candidate.get("elevation_deg",0)))-
                                     float(old.get("pointing_elevation_deg",old.get("elevation_deg",0))))
                        slot_seconds=float(previous_plan.get("valid_until_s",t+60.))-float(previous_plan.get("valid_from_s",t))
                        if (delta_az/max(.01,self.config.antenna_az_slew_deg_s)<=slot_seconds+1e-9 and
                            delta_el/max(.01,self.config.antenna_el_slew_deg_s)<=slot_seconds+1e-9):filtered.append(candidate);break
                        continue
                    if self.config.antenna_steering_type=="electronic":seconds=max(0.,self.config.electronic_switch_ms)/1000.
                    else:
                        delta_az=abs((float(candidate.get("azimuth_deg",0))-float(old.get("azimuth_deg",0))+180.)%360.-180.)
                        delta_el=abs(float(candidate.get("pointing_elevation_deg",candidate.get("elevation_deg",0)))-
                                     float(old.get("pointing_elevation_deg",old.get("elevation_deg",0))))
                        seconds=max(_axis_slew_seconds(delta_az,self.config.antenna_az_slew_deg_s,self.config.antenna_az_accel_deg_s2),
                                    _axis_slew_seconds(delta_el,self.config.antenna_el_slew_deg_s,self.config.antenna_el_accel_deg_s2))+self.config.antenna_acquisition_s
                    if seconds<=self._handover_window(old,carried_by_feeder,slot_seconds)+1e-9:
                        filtered.append(candidate);break
            candidates=filtered
            direction_after=direction_counts(candidates)
        else:
            direction_before=direction_after=None
        priority_demands={source:min(demand,max(0.,float((priority_traffic or {}).get(source,0.))*self.config.traffic_landing_ratio))
                          for source,demand in demands.items() if float((priority_traffic or {}).get(source,0.))>0.}
        for source in priority_demands:
            priorities[source]=max(1.,float(self.config.priority_task_weight));priority_sources.add(source)
        for x in candidates:x["gsl_distance_km"]=_gsl_distance_km(positions[int(x["satellite"])][4],x["elevation_deg"])
        reserve=max(0.,float(self.config.capacity_reserve_ratio))
        planning_demands=({source:value*(1.+reserve) for source,value in demands.items()}
                          if self.config.selection_strategy not in ("window_margin","slot_tracking") else demands)
        chosen=self._select(candidates,g,planning_demands,priority_demands=priority_demands)
        capacity_target=sum(demands.values())*(1.+reserve)
        chosen,selected_capacity,target_satisfied=self._enforce_capacity_target(
            candidates,chosen,g,planning_demands,capacity_target)
        if self.config.selection_strategy=="slot_tracking" and priority_traffic_regions:
            region_component_ids=_component_ids(g)
            for region in self.config.priority_task_regions:
                name=str(region.get("name",""));loads=(priority_traffic_regions or {}).get(name,{})
                if sum(float(value) for value in loads.values())<=1e-9:continue
                required_components={region_component_ids[int(source)] for source,amount in loads.items()
                    if float(amount)>0 and 0<=int(source)<len(region_component_ids)}
                def inside(item):
                    return _surface_distance_km(item.get("latitude_deg",0),item.get("longitude_deg",0),
                        region["lat"],region["lon"])<=float(region["radius_km"])
                regional=[item for item in candidates if inside(item) and
                    int(item["satellite"]) not in {int(row["satellite"]) for row in chosen}]
                near=_nearest_demand_distance(g,{int(source):float(value) for source,value in loads.items()})
                regional.sort(key=lambda item:(near[int(item["satellite"])]/1000+
                    float(item.get("gsl_distance_km",0)),-float(item.get("rate_gbps",0))))
                source_positions=[position_by_id[int(source)] for source,amount in loads.items()
                                  if float(amount)>0 and int(source) in position_by_id]
                def local_coverage_ok(local):
                    if not local:return False
                    if sum(float(item.get("rate_gbps",0)) for item in local)<sum(float(value) for value in loads.values())*self.config.traffic_landing_ratio*1.05:return False
                    return all(min(_surface_distance_km(source[2],source[3],item["latitude_deg"],item["longitude_deg"])
                                   for item in local)<=float(region["radius_km"])/2 for source in source_positions)
                for item in regional:
                    local=[row for row in chosen if inside(row)]
                    covered_counts={component:sum(region_component_ids[int(row["satellite"])]==component
                        for row in local) for component in required_components}
                    missing_components={component for component in required_components
                        if covered_counts.get(component,0)<2}
                    if not missing_components and (local_coverage_ok(local) or
                        len(local)>=max(5,2*len(required_components))):break
                    if missing_components and region_component_ids[int(item["satellite"])] not in missing_components:continue
                    gateway=int(item["gateway"]);quota=self._gateway_limit(gateway)
                    at_gateway=[index for index,row in enumerate(chosen) if int(row["gateway"])==gateway]
                    if len(at_gateway)<quota:
                        chosen.append(item);continue
                    removable=[index for index in at_gateway if not any(
                        _surface_distance_km(chosen[index].get("latitude_deg",0),chosen[index].get("longitude_deg",0),
                            other["lat"],other["lon"])<=float(other["radius_km"])
                        for other in self.config.priority_task_regions)]
                    if removable:
                        chosen[min(removable,key=lambda index:float(chosen[index].get("rate_gbps",0)))]=item
            selected_capacity=sum(float(item.get("rate_gbps",0)) for item in chosen)
            target_satisfied=selected_capacity+1e-9>=capacity_target
        selected_before_retention=(direction_counts(chosen) if direction_before is not None else None)
        selected_after_retention=None
        selected_after_primary_match=None
        handover={}
        if self.config.selection_strategy=="slot_tracking":
            if self.config.feeder_link_mode!="dual":
                chosen=self._retain_slot_links(chosen,candidates,previous_plan,g,demands,capacity_target,t,next_slot_visible_pairs)
            selected_after_retention=(direction_counts(chosen) if direction_before is not None else None)
            primary,handover=self._assign_slot_antennas(chosen,previous_plan,t,graph=g)
            selected_after_primary_match=(direction_counts(primary) if direction_before is not None else None)
            # Give the antenna matcher fallback satellites before it drops an
            # infeasible first choice and leaves an otherwise usable antenna idle.
            selected_satellites={int(row["satellite"]) for row in primary}
            alternatives_by_gateway={}
            for candidate in candidates:
                satellite=int(candidate["satellite"])
                if satellite not in selected_satellites:
                    alternatives_by_gateway.setdefault(int(candidate["gateway"]),[]).append(candidate)
            fallback=[]
            for gateway,alternatives in alternatives_by_gateway.items():
                alternatives.sort(key=lambda row:float(row.get("rate_gbps",0)),reverse=True)
                for candidate in alternatives:
                    satellite=int(candidate["satellite"])
                    if satellite not in selected_satellites:
                        fallback.append(candidate);selected_satellites.add(satellite)
            fixed={(int(row["gateway"]),int(row["satellite"]),int(row.get("feeder_port",1))):int(row["antenna_id"])
                   for row in primary}
            chosen,handover=self._assign_slot_antennas(primary+fallback,previous_plan,t,fixed,graph=g)
            # The generic matcher can still omit the last feeder in a
            # priority-bearing ISL component.  Try a physically feasible
            # one-for-one replacement while locking every other antenna.
            regional_rescues=0
            if self.config.feeder_link_mode!="dual" and priority_traffic_regions and self.config.priority_task_regions:
                component_ids=_component_ids(g)
                for region in self.config.priority_task_regions:
                    loads=(priority_traffic_regions or {}).get(str(region.get("name","")),{})
                    required={component_ids[int(source)] for source,amount in loads.items()
                        if float(amount)>0 and 0<=int(source)<len(component_ids)}
                    if not required:continue
                    def in_region(row):
                        return _surface_distance_km(row.get("latitude_deg",0),row.get("longitude_deg",0),
                            region["lat"],region["lon"])<=float(region["radius_km"])
                    covered={component_ids[int(row["satellite"])] for row in chosen if in_region(row)}
                    for missing in sorted(required-covered):
                        used_satellites={int(row["satellite"]) for row in chosen}
                        replacements=sorted((row for row in candidates
                            if component_ids[int(row["satellite"])]==missing and in_region(row)
                            and int(row["satellite"]) not in used_satellites),
                            key=lambda row:(-float(row.get("rate_gbps",0)),float(row.get("gsl_distance_km",0))))
                        repaired=False
                        for candidate in replacements:
                            gateway=int(candidate["gateway"])
                            current_counts={component:sum(component_ids[int(row["satellite"])]==component
                                for row in chosen if in_region(row)) for component in required}
                            victims=sorted((row for row in chosen if int(row["gateway"])==gateway and
                                (not in_region(row) or current_counts.get(component_ids[int(row["satellite"])],0)>1)),
                                key=lambda row:(in_region(row),float(row.get("rate_gbps",0))))
                            for victim in victims:
                                trial=[row for row in chosen if row is not victim]+[candidate]
                                locks={(int(row["gateway"]),int(row["satellite"])):int(row["antenna_id"])
                                    for row in chosen if row is not victim}
                                assigned,trial_handover=self._assign_slot_antennas(trial,previous_plan,t,locks,graph=g)
                                if len(assigned)==len(chosen) and any(
                                    int(row["gateway"])==gateway and int(row["satellite"])==int(candidate["satellite"])
                                    for row in assigned):
                                    chosen,handover=assigned,trial_handover
                                    regional_rescues+=1;repaired=True;break
                            if repaired:break
            handover["regional_component_rescue_count"]=regional_rescues
            # Feeder capacity in one disconnected ISL component cannot serve
            # traffic in another. Replace surplus links at the same gateway
            # when a physically feasible target can reduce a component gap.
            component_ids=_component_ids(g)
            component_demands={}
            for source,amount in demands.items():
                component=component_ids[int(source)]
                component_demands[component]=component_demands.get(component,0.)+float(amount)
            component_rescues=0
            component_trial_budget=48
            for _attempt in range(0 if self.config.feeder_link_mode=="dual" else min(8,len(chosen))):
                component_capacity={}
                for link in chosen:
                    component=component_ids[int(link["satellite"])]
                    component_capacity[component]=component_capacity.get(component,0.)+float(link["rate_gbps"])
                deficits=sorted(((amount-component_capacity.get(component,0.),component)
                    for component,amount in component_demands.items()),reverse=True)
                if not deficits or deficits[0][0]<=1e-6:break
                changed=False
                used={int(link["satellite"]) for link in chosen}
                for deficit,component in deficits:
                    if deficit<=1e-6:break
                    alternatives=sorted((row for row in candidates
                        if component_ids[int(row["satellite"])]==component and int(row["satellite"]) not in used),
                        key=lambda row:float(row["rate_gbps"]),reverse=True)
                    for candidate in alternatives[:8]:
                        gateway=int(candidate["gateway"])
                        victims=sorted((row for row in chosen if int(row["gateway"])==gateway
                            and component_ids[int(row["satellite"])]!=component
                            and component_capacity[component_ids[int(row["satellite"])]]-
                                float(row["rate_gbps"])>=component_demands.get(component_ids[int(row["satellite"])],0.)-1e-6
                            and not any(_surface_distance_km(row.get("latitude_deg",0),row.get("longitude_deg",0),
                                region["lat"],region["lon"])<=float(region["radius_km"])
                                for region in self.config.priority_task_regions)),
                            key=lambda row:float(row["rate_gbps"]))
                        for victim in victims[:4]:
                            if component_trial_budget<=0:break
                            component_trial_budget-=1
                            trial=[row for row in chosen if row is not victim]+[candidate]
                            locks={(int(row["gateway"]),int(row["satellite"])):int(row["antenna_id"])
                                for row in chosen if row is not victim}
                            assigned,trial_handover=self._assign_slot_antennas(trial,previous_plan,t,locks,graph=g)
                            if len(assigned)!=len(chosen) or not any(int(row["satellite"])==int(candidate["satellite"])
                                and int(row["gateway"])==gateway for row in assigned):continue
                            chosen,handover=assigned,trial_handover
                            component_rescues+=1;changed=True;break
                        if changed:break
                    if changed:break
                if not changed or component_trial_budget<=0:break
            handover["regional_component_rescue_count"]=regional_rescues
            handover["capacity_component_rescue_count"]=component_rescues
            handover["unassignable_link_count"]=max(0,sum(self._gateway_limit(gateway)
                for gateway in {int(row["gateway"]) for row in candidates})-len(chosen))
            selected_capacity=sum(float(item.get("rate_gbps",0)) for item in chosen)
            target_satisfied=selected_capacity+1e-9>=capacity_target
        selection_compute_ms=round((time.perf_counter()-selection_started)*1000,3)
        capacities,gsl_distances=_feeder_resources(chosen)
        routes=[]
        # Offline planning uses one complete bounded search.  The former
        # 8/16/24 sequence rebuilt almost the same multi-source labels three
        # times for every scenario and dominated total planning time.
        routes=_routes_with_capacity_completion(g,capacities,demands,capacities,candidate_count=24,
                       capacity_weight=self.config.capacity_weight,delay_weight=self.config.delay_weight,
                       delay_reference_ms=self.config.delay_reference_ms,
                       feeder_gsl_distances=gsl_distances,
                       max_active_exits_per_source=self.config.planning_exit_limit_per_source,
                       source_priorities=priorities,priority_demands=priority_demands) if chosen else []
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
        for _joint_repair in range(0 if self.config.feeder_link_mode=="dual" or self.config.selection_strategy=="window_margin" else 1):
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
            replacements.sort(reverse=True,key=lambda row:row[:4]);trial=[dict(item) for item in chosen];protected=set();swap_limit=min(4 if self.config.selection_strategy=="slot_tracking" else 12,max(2,len(deficient)*2));swaps=0
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
            if self.config.selection_strategy=="slot_tracking":
                fixed={(int(row["gateway"]),int(row["satellite"])):int(row["antenna_id"])
                       for row in chosen if int(row["satellite"]) in selected_satellites}
                assigned,trial_handover=self._assign_slot_antennas(trial,previous_plan,t,fixed,graph=g)
                if len(assigned)!=len(chosen):break
                trial=assigned
            trial_capacities,trial_gsl=_feeder_resources(trial)
            trial_routes=_routes_with_capacity_completion(g,trial_capacities,demands,trial_capacities,candidate_count=24,
                         capacity_weight=self.config.capacity_weight,delay_weight=self.config.delay_weight,
                         delay_reference_ms=self.config.delay_reference_ms,feeder_gsl_distances=trial_gsl,
                         max_active_exits_per_source=self.config.planning_exit_limit_per_source,
                         source_priorities=priorities,priority_demands=priority_demands)
            old_score,old_delivery,_old_average,_old_p95=repair_objective(routes,gsl_distances)
            new_score,new_delivery,_new_average,_new_p95=repair_objective(trial_routes,trial_gsl)
            joint_repair_score_before=old_score;joint_repair_score_after=new_score
            # Only accept a replacement that delivers more traffic. Path
            # latency ranks feasible exits during routing, not this repair.
            if (new_delivery<=old_delivery+1e-6 or new_score<=old_score+1e-9 or
                    _priority_delivery(trial_routes,priority_demands)+1e-6<_priority_delivery(routes,priority_demands)):break
            chosen,routes,capacities,gsl_distances=trial,trial_routes,trial_capacities,trial_gsl
            if self.config.selection_strategy=="slot_tracking":handover=trial_handover
            joint_repair_applied=True
        capacity_fallback_used=False
        capacity_fallback_ms=0.
        total_business=sum(demands.values())
        if (self.config.selection_strategy=="legacy" and
                sum(float(row.get("demand_gbps",0)) for row in routes)<.99*total_business):
            fallback_started=time.perf_counter()
            conservative=self._select(candidates,g,planning_demands,capacity_only=True,priority_demands=priority_demands)
            conservative,conservative_capacity,conservative_target=self._enforce_capacity_target(
                candidates,conservative,g,planning_demands,capacity_target)
            conservative_rates,conservative_gsl=_feeder_resources(conservative)
            conservative_routes=_routes_with_capacity_completion(
                g,conservative_rates,demands,conservative_rates,candidate_count=24,
                capacity_weight=self.config.capacity_weight,delay_weight=self.config.delay_weight,
                delay_reference_ms=self.config.delay_reference_ms,
                feeder_gsl_distances=conservative_gsl,
                max_active_exits_per_source=self.config.planning_exit_limit_per_source,
                source_priorities=priorities,priority_demands=priority_demands) if conservative else []
            if (sum(float(row.get("demand_gbps",0)) for row in conservative_routes)>
                    sum(float(row.get("demand_gbps",0)) for row in routes)+1e-6 and
                    _priority_delivery(conservative_routes,priority_demands)+1e-6>=_priority_delivery(routes,priority_demands)):
                chosen,routes,capacities,gsl_distances=conservative,conservative_routes,conservative_rates,conservative_gsl
                selected_capacity,target_satisfied=conservative_capacity,conservative_target
                capacity_fallback_used=True
            capacity_fallback_ms=round((time.perf_counter()-fallback_started)*1000,3)
        priority_feeder_ids=set();priority_feeders_by_region={};priority_region_delivery={}
        if self.config.selection_strategy=="slot_tracking" and priority_demands and chosen:
            # Reserve 1–5 nearby exits for each region's ground-origin traffic.
            # Other traffic is allocated only after these reservations.
            regions=priority_traffic_regions or {"重点区域":priority_traffic or {}}
            priority_remaining=dict(priority_demands);used={};reserved=[];region_unmet={}
            for region_name,region_loads in regions.items():
                regional={source:min(priority_remaining[source],max(0.,float(region_loads.get(source,0.))*self.config.traffic_landing_ratio))
                          for source in priority_remaining if priority_remaining[source]>1e-9 and float(region_loads.get(source,0.))>0.}
                if not regional:continue
                geometry=next((item for item in self.config.priority_task_regions if str(item.get("name"))==str(region_name)),None)
                nearest_priority=_nearest_demand_distance(g,regional)
                # The region circle defines where demand originates. A feeder
                # satellite's subpoint need not be inside that circle: the
                # nearby *gateway* is the landing site, and ISLs may be needed.
                regional_candidates=list(chosen)
                ranked=sorted(regional_candidates,key=lambda row:(
                    _surface_distance_km(self.stations[int(row["gateway"])]["lat"],
                        self.stations[int(row["gateway"])]["lon"],geometry["lat"],geometry["lon"])
                    if geometry is not None else 0.,
                    nearest_priority[int(row["satellite"])]/1000+
                    float(row.get("gsl_distance_km",0)),-float(row.get("rate_gbps",0))))
                exits=set();available_total=0.
                if geometry is not None:
                    # Pick the five exits by traffic that can actually reach
                    # them through the current ISL graph, not surface distance
                    # alone.  The later reserved-route pass uses only these.
                    probe_capacity={int(item["satellite"]):max(0.,capacities[int(item["satellite"])]-
                        used.get(int(item["satellite"]),0.)) for item in ranked}
                    probe_routes=_routes(g,probe_capacity,regional,probe_capacity,
                        candidate_count=len(probe_capacity),feeder_gsl_distances=gsl_distances,
                        max_active_exits_per_source=5,priority_demands=regional) if probe_capacity else []
                    probe_load={}
                    for route in probe_routes:
                        feeder=int(route["feeder"])
                        probe_load[feeder]=probe_load.get(feeder,0.)+float(route["demand_gbps"])
                    component_ids=_component_ids(g)
                    component_load={}
                    for source,amount in regional.items():
                        component=component_ids[int(source)]
                        component_load[component]=component_load.get(component,0.)+float(amount)
                    for component in sorted(component_load,key=lambda key:component_load[key],reverse=True):
                        choices=[int(item["satellite"]) for item in ranked
                            if component_ids[int(item["satellite"])]==component]
                        if choices and len(exits)<5:
                            exits.add(max(choices,key=lambda feeder:(probe_load.get(feeder,0.),
                                probe_capacity.get(feeder,0.))))
                    for feeder in sorted(probe_load,key=lambda key:probe_load[key],reverse=True):
                        if len(exits)>=5:break
                        exits.add(feeder)
                    # Capacity at one exit does not imply that it is near every
                    # priority source. Spread at most five exits over the area.
                    radius=float(geometry["radius_km"])
                    source_coords={source:(position_by_id[source][2],position_by_id[source][3])
                                   for source in regional if source in position_by_id}
                    nearest_by_source={source:2.*radius for source in source_coords}
                    for item in ranked:
                        satellite=int(item["satellite"])
                        if satellite not in exits:continue
                        available_total+=max(0.,capacities[satellite]-used.get(satellite,0.))
                        for source,(lat,lon) in source_coords.items():
                            nearest_by_source[source]=min(nearest_by_source[source],
                                _surface_distance_km(lat,lon,item["latitude_deg"],item["longitude_deg"]))
                    remaining_candidates=list(ranked)
                    while remaining_candidates and len(exits)<5:
                        def coverage_gain(item):
                            improvement=sum(regional[source]*max(0.,nearest_by_source[source]-
                                _surface_distance_km(lat,lon,item["latitude_deg"],item["longitude_deg"]))
                                for source,(lat,lon) in source_coords.items())
                            return (improvement,min(capacities[int(item["satellite"])],sum(regional.values())),
                                    -float(item.get("gsl_distance_km",0)))
                        item=max(remaining_candidates,key=coverage_gain);remaining_candidates.remove(item)
                        satellite=int(item["satellite"])
                        if satellite in exits:continue
                        exits.add(satellite)
                        available_total+=max(0.,capacities[satellite]-used.get(satellite,0.))
                        for source,(lat,lon) in source_coords.items():
                            nearest_by_source[source]=min(nearest_by_source[source],
                                _surface_distance_km(lat,lon,item["latitude_deg"],item["longitude_deg"]))
                        if (available_total>=sum(regional.values())*1.05 and
                            all(distance<=radius/2 for distance in nearest_by_source.values())):break
                else:
                    for item in ranked:
                        satellite=int(item["satellite"])
                        if satellite in exits:continue
                        exits.add(satellite);available_total+=max(0.,capacities[satellite]-used.get(satellite,0.))
                        if len(exits)>=5 or available_total>=sum(regional.values())*1.05:break
                available={satellite:max(0.,capacities[satellite]-used.get(satellite,0.)) for satellite in exits}
                regional_routes=_routes(g,available,regional,available,candidate_count=len(available),
                    feeder_gsl_distances={satellite:gsl_distances[satellite] for satellite in exits},
                    max_active_exits_per_source=5,priority_demands=regional,source_priorities=priorities)
                baseline_delay={}
                for row in probe_routes if geometry is not None else ():
                    source=int(row["source"])
                    delay=(float(row["distance_km"])+float(gsl_distances.get(int(row["feeder"]),0.)))/C_KM_PER_MS
                    baseline_delay[source]=min(delay,baseline_delay.get(source,math.inf))
                regional_routes=[row for row in regional_routes if
                    int(row["source"]) not in baseline_delay or
                    (float(row["distance_km"])+float(gsl_distances.get(int(row["feeder"]),0.)))/C_KM_PER_MS
                    <=max(baseline_delay[int(row["source"])]*1.5,baseline_delay[int(row["source"])]+3.0)]
                used_exits=set()
                for row in regional_routes:
                    amount=float(row["demand_gbps"]);row["priority_gbps"]=amount;row["priority_region"]=region_name
                    used[int(row["feeder"])]=used.get(int(row["feeder"]),0.)+amount
                    priority_remaining[int(row["source"])]=max(0.,priority_remaining[int(row["source"]) ]-amount)
                    used_exits.add(int(row["feeder"]))
                reserved.extend(regional_routes)
                priority_feeders_by_region[str(region_name)]=sorted(used_exits)
                delivered_region=sum(float(row["demand_gbps"]) for row in regional_routes)
                demanded_region=sum(regional.values())
                priority_region_delivery[str(region_name)]={"demand_gbps":round(demanded_region,6),
                    "delivered_gbps":round(delivered_region,6),"feeder_count":len(used_exits),
                    "target_met":bool(1<=len(used_exits)<=5 and delivered_region>=.99*demanded_region)}
                delivered_sources={}
                for row in regional_routes:
                    source=int(row["source"])
                    delivered_sources[source]=delivered_sources.get(source,0.)+float(row["demand_gbps"])
                region_unmet[str(region_name)]={source:max(0.,amount-delivered_sources.get(source,0.))
                    for source,amount in regional.items()}
                priority_feeder_ids.update(used_exits)
            remaining_caps={satellite:max(0.,value-used.get(satellite,0.)) for satellite,value in capacities.items()}
            # Only the priority traffic actually carried by the reserved exits
            # is removed. Unserved priority traffic competes before ordinary
            # traffic for every remaining usable exit.
            reserved_by_source={}
            for row in reserved:
                source=int(row["source"])
                reserved_by_source[source]=reserved_by_source.get(source,0.)+float(row["demand_gbps"])
            remaining_demands={source:max(0.,value-reserved_by_source.get(source,0.)) for source,value in demands.items()}
            reserved_exits_by_source={}
            for row in reserved:
                reserved_exits_by_source.setdefault(int(row["source"]),set()).add(int(row["feeder"]))
            ordinary_exit_limits={source:max(1,self.config.planning_exit_limit_per_source-len(reserved_exits_by_source.get(source,())))
                                  for source in remaining_demands}
            ordinary=_routes_with_capacity_completion(g,remaining_caps,remaining_demands,remaining_caps,candidate_count=24,
                capacity_weight=self.config.capacity_weight,delay_weight=self.config.delay_weight,
                delay_reference_ms=self.config.delay_reference_ms,feeder_gsl_distances=gsl_distances,
                max_active_exits_per_source=self.config.planning_exit_limit_per_source,
                source_exit_limits=ordinary_exit_limits,
                priority_demands=priority_remaining)
            region_geometry={str(item.get("name")):item for item in self.config.priority_task_regions}
            chosen_by_satellite={int(item["satellite"]):item for item in chosen}
            for row in ordinary:
                source=int(row["source"])
                amount=min(float(row["demand_gbps"]),priority_remaining.get(source,0.))
                if amount<=1e-9:continue
                # A distant ordinary route may still carry business, but it is
                # not a nearby exit fulfilling the regional guarantee.
                local_amount=0.
                for region_name,unmet in region_unmet.items():
                    geometry=region_geometry.get(region_name)
                    feeder_row=chosen_by_satellite.get(int(row["feeder"]))
                    if geometry is not None and (int(row["feeder"]) not in priority_feeders_by_region.get(region_name,[]) or feeder_row is None):continue
                    region_amount=min(amount,unmet.get(source,0.))
                    if region_amount<=1e-9:continue
                    unmet[source]-=region_amount;amount-=region_amount;local_amount+=region_amount
                    region_result=priority_region_delivery[region_name]
                    region_result["delivered_gbps"]=round(region_result["delivered_gbps"]+region_amount,6)
                    feeder=int(row["feeder"])
                    feeders=set(priority_feeders_by_region.get(region_name,[]));feeders.add(feeder)
                    priority_feeders_by_region[region_name]=sorted(feeders)
                    region_result["feeder_count"]=len(feeders)
                    region_result["target_met"]=bool(1<=len(feeders)<=5 and
                        region_result["delivered_gbps"]>=.99*region_result["demand_gbps"])
                    priority_feeder_ids.add(feeder)
                    if amount<=1e-9:break
                if local_amount>1e-9:
                    row["priority_gbps"]=round(local_amount,6)
                    priority_remaining[source]=max(0.,priority_remaining[source]-local_amount)
            routes=reserved+ordinary
        # Regional exit reservation can send a feeder satellite's own access
        # traffic over ISLs even while its GSL has unused capacity. Move such
        # a route back to the source feeder without changing delivered volume
        # or consuming an additional source exit.
        used_by_feeder={}
        for row in routes:
            feeder=int(row["feeder"])
            used_by_feeder[feeder]=used_by_feeder.get(feeder,0.)+float(row["demand_gbps"])
        chosen_by_feeder={int(row["satellite"]):row for row in chosen}
        direct_corrections=0
        for row in routes:
            source=int(row["source"])
            if source==int(row["feeder"]) or source not in chosen_by_feeder:continue
            amount=float(row.get("demand_gbps",0.))
            if amount<=1e-9 or capacities.get(source,0.)-used_by_feeder.get(source,0.)+1e-9<amount:continue
            # Keep a reserved regional route on one of its selected landing exits.
            region_name=row.get("priority_region")
            if region_name:
                region=next((item for item in self.config.priority_task_regions
                    if str(item.get("name"))==str(region_name)),None)
                if region is not None and source not in priority_feeders_by_region.get(region_name,[]):continue
            if any(other is not row and int(other["source"])==source and
                   int(other["feeder"])==source for other in routes):continue
            old_feeder=int(row["feeder"])
            used_by_feeder[old_feeder]-=amount
            used_by_feeder[source]=used_by_feeder.get(source,0.)+amount
            gsl_km=float(gsl_distances.get(source,0.))
            utility,capacity_gain,delay_gain,delay_quality=_marginal_utility(
                amount,gsl_km/C_KM_PER_MS,max(sum(demands.values()),1e-12),
                self.config.capacity_weight,self.config.delay_weight,self.config.delay_reference_ms)
            row.update(feeder=source,path=[source],hops=0,distance_km=0.,
                isl_propagation_delay_ms=0.,route_objective_score=round(utility,9),
                route_capacity_gain=round(capacity_gain,9),route_delay_gain=round(delay_gain,9),
                route_delay_quality=round(delay_quality,9))
            direct_corrections+=1
        # The regional five-exit reservation can leave a priority source on a
        # very long route while another selected feeder inside the same region
        # is one ISL hop away and still has spare capacity. Recheck long routes
        # against all selected in-region exits after ordinary allocation.
        region_components=None
        for row in routes:
            region_name=row.get("priority_region")
            if not region_name or int(row.get("hops",0))<=6:continue
            region=next((item for item in self.config.priority_task_regions
                if str(item.get("name"))==str(region_name)),None)
            if region is None:continue
            amount=float(row.get("demand_gbps",0.))
            if amount<=1e-9:continue
            old_feeder=int(row["feeder"])
            nearby_capacity={satellite:capacity-used_by_feeder.get(satellite,0.)
                for satellite,capacity in capacities.items()
                if satellite!=old_feeder and capacity-used_by_feeder.get(satellite,0.)+1e-9>=amount
                and satellite in chosen_by_feeder and satellite in priority_feeders_by_region.get(region_name,[])}
            if not nearby_capacity:continue
            if region_components is None:region_components=_component_ids(g)
            alternative=_repair_routes_local(g,{int(row["source"]):amount},nearby_capacity,
                gsl_distances,max_visited_per_source=2000,max_exits_per_source=1,
                local_feeders_by_source={int(row["source"]):set(nearby_capacity)},
                component_ids=region_components)
            if not alternative:continue
            replacement=alternative[0]
            if float(replacement["demand_gbps"])+1e-6<amount:continue
            old_distance=float(row.get("distance_km",0))+float(gsl_distances.get(old_feeder,0.))
            new_feeder=int(replacement["feeder"])
            new_distance=float(replacement["distance_km"])+float(gsl_distances.get(new_feeder,0.))
            if new_distance+1e-3>=old_distance:continue
            used_by_feeder[old_feeder]-=amount
            used_by_feeder[new_feeder]=used_by_feeder.get(new_feeder,0.)+amount
            utility,*_=_marginal_utility(amount,new_distance/C_KM_PER_MS,
                max(sum(demands.values()),1e-12),self.config.capacity_weight,
                self.config.delay_weight,self.config.delay_reference_ms)
            row.update(feeder=new_feeder,path=replacement["path"],hops=replacement["hops"],
                distance_km=replacement["distance_km"],
                isl_propagation_delay_ms=replacement["isl_propagation_delay_ms"],
                route_objective_score=round(utility,9))
            direct_corrections+=1
        if direct_corrections and priority_region_delivery:
            for region_name,result in priority_region_delivery.items():
                exits={int(row["feeder"]) for row in routes if row.get("priority_region")==region_name
                    and float(row.get("priority_gbps",0.))>1e-9}
                priority_feeders_by_region[str(region_name)]=sorted(exits)
                result["feeder_count"]=len(exits)
                result["target_met"]=bool(1<=len(exits)<=5 and
                    result["delivered_gbps"]>=.99*result["demand_gbps"])
            priority_feeder_ids={feeder for exits in priority_feeders_by_region.values() for feeder in exits}
        if self.config.feeder_link_mode=="dual":
            routes,_planned_link_usage=_bind_routes_to_links(routes,chosen)
        _attach_feeder_delay(routes,chosen);avg,bweighted,delay,p95_delay,max_delay=_route_metrics(routes)
        usage=[{"gateway":i,"used":sum(x["gateway"]==i for x in chosen),"available":self._gateway_limit(i)} for i,_s in enumerate(self.stations)]
        routed_business=sum(float(x.get("demand_gbps",0)) for x in routes);total_business=sum(demands.values())
        priority_demand=sum(priority_demands.values())
        delivered_by_source={}
        for row in routes:
            source=int(row.get("source",-1));delivered_by_source[source]=delivered_by_source.get(source,0.)+float(row.get("demand_gbps",0))
        priority_delivered=(sum(float(row.get("priority_gbps",0.)) for row in routes) if self.config.selection_strategy=="slot_tracking"
                            else sum(min(amount,delivered_by_source.get(source,0.)) for source,amount in priority_demands.items()))
        delivery_ratio=1.0 if total_business<=1e-12 else min(1.,routed_business/total_business)
        if self.config.selection_strategy=="window_margin":
            load_by_feeder={}
            for row in routes:
                feeder=int(row.get("feeder",-1))
                load_by_feeder[feeder]=load_by_feeder.get(feeder,0.)+float(row.get("demand_gbps",0.))
            for link in chosen:
                key=(int(link["gateway"]),int(link["satellite"]),int(link.get("feeder_port",1)))
                load=(_planned_link_usage.get(key,0.) if self.config.feeder_link_mode=="dual"
                      else load_by_feeder.get(int(link["satellite"]),0.))
                link["planned_load_gbps"]=round(load,6)
                link["planned_window_spare_gbps"]=round(max(0.,float(link["rate_gbps"])-load),6)
        plan={"id":"offline_plan","name":"离线站星连接规划","generated_at_s":t,
          "valid_from_s":t,"valid_until_s":t+self.config.planning_horizon_s,
          "slot_s":self.config.planning_slot_s,"slot_count":max(1,math.ceil(self.config.planning_horizon_s/self.config.planning_slot_s)),
          "freeze_until_s":t+self.config.freeze_window_s,"links":chosen,"routes":routes,
          "total_capacity_gbps":round(sum(x["rate_gbps"] for x in chosen),6),"total_demand_gbps":round(sum(demands.values()),6),
          "capacity_reserve_ratio":round(reserve,6),"capacity_target_gbps":round(capacity_target,6),
          "selection_strategy":self.config.selection_strategy,"feeder_link_mode":self.config.feeder_link_mode,
          "rain_grade":rain_grade or "中雨","rain_scenario_percentile":None,
          "capacity_target_satisfied":bool(target_satisfied),"capacity_target_gap_gbps":round(max(0.,capacity_target-selected_capacity),6),
          "traffic_satellite_count":len(demands),
          "demand_by_source_gbps":{str(source):round(amount,6) for source,amount in demands.items()},
          "candidate_link_count":len(candidates),"selection_compute_ms":selection_compute_ms,
          "reachable_demand_gbps":round(routed_business,6),"planned_delivery_ratio":round(delivery_ratio,6),
          "delivery_target_ratio":.99,"delivery_target_satisfied":bool(delivery_ratio>=.99-1e-9),
          "delivery_gap_gbps":round(max(0.,total_business-routed_business),6),
          "active_exit_limit_per_source":self.config.planning_exit_limit_per_source,
          "priority_region_count":len(self.config.priority_task_regions),"priority_satellite_count":len(priority_sources),
          "priority_demand_gbps":round(priority_demand,6),"priority_delivered_gbps":round(priority_delivered,6),
          "priority_demands_by_satellite_gbps":{str(source):round(amount,6) for source,amount in priority_demands.items()},
          "priority_delivery_ratio":round(1. if priority_demand<=1e-12 else min(1.,priority_delivered/priority_demand),6),
          "priority_feeder_satellites":sorted(priority_feeder_ids),
          "priority_feeder_count":len({int(row["feeder"]) for row in routes if float(row.get("priority_gbps",0.))>0.}),
          "priority_feeders_by_region":priority_feeders_by_region,
          "priority_region_delivery":priority_region_delivery,
          "gateway_count":len({x["gateway"] for x in chosen}),
          "average_hops":avg,"business_weighted_hops":bweighted,"business_propagation_delay_ms":delay,
          "p95_propagation_delay_ms":p95_delay,"maximum_propagation_delay_ms":max_delay,"latency_scope":"服务卫星至信关站（ISL+馈电GSL）","station_usage":usage}
        plan.update(handover)
        if direction_before is not None:
            plan["candidate_direction_diagnostics"]={"before_switch_filter":direction_before,
                "after_switch_filter":direction_after,
                "optimizer_selected":selected_before_retention,
                "after_retention":selected_after_retention,
                "after_primary_match":selected_after_primary_match,
                "selected":direction_counts(chosen)}
        plan.update(capacity_weight=round(self.config.capacity_weight,6),delay_weight=round(self.config.delay_weight,6),
                    adaptive_candidate_k=int(getattr(self,"_last_candidate_k",0)),delay_reference_ms=round(self.config.delay_reference_ms,6),
                    selection_method="有界多源最短路 + 容量时延双目标边际优化",
                    objective_formula="wc*(新增落地业务/总需求)+wd*参考时延/(参考时延+路径时延)",
                    objective_capacity_is_hard_constraint=self.config.selection_strategy not in ("window_margin","slot_tracking"),
                    objective_gateway_channels_are_hard_constraints=True)
        plan.update(joint_repair_applied=joint_repair_applied,
                    capacity_fallback_used=capacity_fallback_used,
                    capacity_fallback_ms=capacity_fallback_ms,
                    joint_repair_score_before=round(joint_repair_score_before,9),
                    joint_repair_score_after=round(joint_repair_score_after,9))
        if self.config.selection_strategy=="window_margin":
            plan["selection_method"]="five-minute visible links; minimum rate for feasibility, mean rate and path delay for scoring"
            plan["objective_formula"]="wc*normalized(window_mean_rate)+wd*normalized(path_delay_quality); minimum-rate capacity target repair"
        elif self.config.selection_strategy=="slot_tracking":
            plan["selection_method"]="five-minute rolling plan with 60-second link review, retained feeders and hard-handover outage"
            plan["objective_formula"]="capacity and delay utility with next-slot visibility; retained feeders compete with feasible successors"
            plan["rain_field"]=self.config.rain_field
        elif self.config.capacity_weight==0 and self.config.delay_weight==0:
            plan["selection_method"]="capacity feasibility, then shortest feasible path"
            plan["objective_formula"]="lexicographic: satisfy capacity and gateway quotas; minimize path delay"
        objective_rows=[x for x in chosen if "selection_score" in x]
        plan["objective_diagnostics"]={
          "selected_signature":[[int(x["gateway"]),int(x["satellite"]),int(x.get("feeder_port",1))]
              if self.config.feeder_link_mode=="dual" else [int(x["gateway"]),int(x["satellite"])] for x in chosen],
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

    def _slot_switch_starts(self,plan,previous):
        slot_start=float(plan.get("valid_from_s",plan.get("generated_at_s",0)))
        slot_end=float(plan.get("valid_until_s",slot_start+60.))
        standby_starts={(int(row["gateway"]),int(row["antenna_id"])):slot_start
            for row in plan.get("links",[]) if row.get("antenna_id") is not None
            and row.get("handover_from_antenna_id") is not None}
        previous_by_antenna={(int(row["gateway"]),int(row["antenna_id"])):row
            for row in (previous or {}).get("links",[]) if row.get("antenna_id") is not None}
        if self.config.antenna_steering_type=="electronic":
            return standby_starts | {(int(row["gateway"]),int(row["antenna_id"])):slot_start
                for row in plan.get("links",[]) if row.get("antenna_id") is not None
                and (old:=previous_by_antenna.get((int(row["gateway"]),int(row["antenna_id"])))) is not None
                and int(old["satellite"])!=int(row["satellite"])}
        carrying_feeders={int(row.get("feeder",-1)) for row in (previous or {}).get("routes",[])
                          if float(row.get("demand_gbps",0))>1e-9}
        changes_by_gateway={}
        for row in plan.get("links",[]):
            antenna=row.get("antenna_id");gateway=int(row["gateway"])
            old=previous_by_antenna.get((gateway,int(antenna))) if antenna is not None else None
            if old is not None and int(old["satellite"])!=int(row["satellite"]):
                if row.get("handover_from_antenna_id") is None:
                    changes_by_gateway.setdefault(gateway,[]).append(row)
        starts=dict(standby_starts)
        visibility_deadlines=plan.get("old_feeder_visibility_deadline_s") or {}
        for gateway,rows in changes_by_gateway.items():
            rows.sort(key=lambda row:(
                float(visibility_deadlines.get(f'{gateway}:{int(row["antenna_id"])}',slot_end)),
                int(previous_by_antenna[(gateway,int(row["antenna_id"]))]["satellite"]) in carrying_feeders,
                int(row["antenna_id"])))
            longest=max(float(row.get("estimated_switch_s") or 0.) for row in rows)
            span=max(0.,slot_end-slot_start-longest)
            for index,row in enumerate(rows):
                antenna=int(row["antenna_id"])
                deadline=float(visibility_deadlines.get(f'{gateway}:{antenna}',slot_end))
                duration=float(row.get("estimated_switch_s") or 0.)
                preferred=slot_start+span*index/max(1,len(rows)-1) if len(rows)>1 else slot_start
                starts[(gateway,antenna)]=max(slot_start,min(preferred,deadline-duration,slot_end-duration))
        return starts

    def prepare_transition_routes(self,plan,previous,positions,links,priority_traffic_regions=None):
        """Preplan routes for every five-second handover phase."""
        if not previous:return plan
        starts=self._slot_switch_starts(plan,previous)
        if not starts:return plan
        preparation_started=time.perf_counter()
        previous_by_antenna={(int(row["gateway"]),int(row["antenna_id"])):row
            for row in previous.get("links",[]) if row.get("antenna_id") is not None}
        slot_start=float(plan.get("valid_from_s",plan.get("generated_at_s",0)))
        slot_end=float(plan.get("valid_until_s",slot_start+60.))
        demands={int(source):float(amount) for source,amount in plan.get("demand_by_source_gbps",{}).items()}
        priority={int(source):float(amount) for source,amount in plan.get("priority_demands_by_satellite_gbps",{}).items()}
        if not demands:return plan
        graph=_graph(positions,links)
        component_ids=_component_ids(graph)
        # The graph is fixed throughout this slot. Reuse each source's bounded
        # shortest-path exploration across the twelve five-second phases.
        search_cache={}
        prepared=[]
        for index in range(int(math.ceil((slot_end-slot_start)/5.))):
            begin=slot_start+5.*index;end=min(slot_end,begin+5.)
            usable={};waiting_old_feeders=set()
            for row in plan.get("links",[]):
                gateway=int(row["gateway"]);antenna=row.get("antenna_id")
                key=(gateway,int(antenna)) if antenna is not None else None
                start=starts.get(key)
                if start is None or begin+1e-9>=start+float(row.get("estimated_switch_s") or 0.) or (
                    (self.config.antenna_steering_type=="electronic" or row.get("handover_from_antenna_id") is not None)
                    and end+1e-9>=start+float(row.get("estimated_switch_s") or 0.)):
                    usable[(gateway,int(antenna) if antenna is not None else -1,int(row["satellite"]))]=row
                elif begin<start+float(row.get("estimated_switch_s") or 0.) and (
                    old:=previous_by_antenna.get((gateway,int(row["handover_from_antenna_id"]))
                        if row.get("handover_from_antenna_id") is not None else key)) is not None:
                    deadline=float((plan.get("old_feeder_visibility_deadline_s") or {}).get(
                        f'{gateway}:{int(old.get("antenna_id",-1))}',slot_end))
                    if end<=deadline+1e-9:
                        usable[(gateway,int(old.get("antenna_id",-1)),int(old["satellite"]))]=old
                        waiting_old_feeders.add(int(old["satellite"]))
            capacities,gsl=_feeder_resources(usable.values())
            local={}
            for region_name,region_loads in (priority_traffic_regions or {}).items():
                region=next((item for item in self.config.priority_task_regions
                    if str(item.get("name"))==str(region_name)),None)
                if region is None:continue
                nearby={int(row["satellite"]) for row in usable.values() if
                    _surface_distance_km(row.get("latitude_deg",0),row.get("longitude_deg",0),
                        region["lat"],region["lon"])<=float(region["radius_km"])}
                for source,amount in region_loads.items():
                    if float(amount)>0:local.setdefault(int(source),set()).update(nearby)
            remaining_demands=dict(demands);remaining_capacities=dict(capacities);held=[];held_exits={}
            # A waiting antenna is still pointed at its old feeder. Rebuild its
            # routes on the current ISL topology instead of copying stale paths
            # from the previous slot or routing its traffic via new feeders.
            old_capacities={feeder:remaining_capacities[feeder] for feeder in waiting_old_feeders
                            if feeder in remaining_capacities}
            if old_capacities:
                held=_repair_routes_local(graph,remaining_demands,old_capacities,gsl,
                    max_visited_per_source=500,
                    max_exits_per_source=self.config.realtime_exit_limit_per_source,
                    priority_demands=priority,local_feeders_by_source=local,
                    component_ids=component_ids,search_cache=search_cache)
                for row in held:
                    source=int(row["source"]);feeder=int(row["feeder"])
                    amount=float(row["demand_gbps"])
                    row["handover_hold"]=True
                    held_exits.setdefault(source,set()).add(feeder)
                    remaining_demands[source]=max(0.,remaining_demands[source]-amount)
                    remaining_capacities[feeder]=max(0.,remaining_capacities[feeder]-amount)
            repaired=_repair_routes_local(graph,remaining_demands,remaining_capacities,gsl,
                max_visited_per_source=500,max_exits_per_source=self.config.realtime_exit_limit_per_source,
                excluded_feeders=held_exits,
                exit_limits={source:max(0,self.config.realtime_exit_limit_per_source-len(held_exits.get(source,()))) for source in remaining_demands},
                priority_demands=priority,local_feeders_by_source=local,
                component_ids=component_ids,search_cache=search_cache) if remaining_capacities else []
            routes=held+repaired
            for route in routes:
                source=int(route["source"])
                route["priority_gbps"]=round(min(float(route["demand_gbps"]),priority.get(source,0.)),6)
            prepared.append({"start_s":round(begin,3),"routes":routes,
                "feeder_count":len(capacities),"routed_gbps":round(sum(float(row["demand_gbps"]) for row in routes),6)})
        plan["transition_routes"]=prepared
        plan["transition_preparation_ms"]=round((time.perf_counter()-preparation_started)*1000,3)
        plan["transition_search_cache_sources"]=len(search_cache)
        return plan

    def install_offline_plan(self,plan):
        """Install a precomputed plan without running global optimization."""
        plan=dict(plan);self.state.generation=int(plan.get("generation",self.state.generation+1))
        slot_start=float(plan.get("valid_from_s",plan.get("generated_at_s",0)))
        slot_end=float(plan.get("valid_until_s",slot_start+60.))
        previous=self.state.plans[0] if self.state.plans else None
        previous_execution=self.state.execution_plan
        if plan.get("transition_routes"):
            plan["transition_fallback_routes"]=list((previous or {}).get("routes",[]))
        previous_by_antenna={(int(row["gateway"]),int(row["antenna_id"])):row
            for row in (previous or {}).get("links",[]) if row.get("antenna_id") is not None}
        switch_start_by_antenna=self._slot_switch_starts(plan,previous)
        execution_links=[];old_links=[];switch_until={}
        for row in plan.get("links",[]):
            new=dict(row);gateway=int(new["gateway"]);satellite=int(new["satellite"])
            antenna=new.get("antenna_id")
            key=(gateway,int(antenna)) if antenna is not None else None
            switch_start=switch_start_by_antenna.get(key)
            if switch_start is not None:
                until=switch_start+float(new.get("estimated_switch_s") or 0.)
                new["scheduled_switch_start_s"]=round(switch_start,3)
                switch_until[(gateway,satellite,int(new.get("feeder_port",1)))]=until
                old_key=(gateway,int(new["handover_from_antenna_id"])) if new.get("handover_from_antenna_id") is not None else key
                if old_key in previous_by_antenna:
                    old=dict(previous_by_antenna[old_key])
                    old_deadline=float((plan.get("old_feeder_visibility_deadline_s") or {}).get(
                        f'{old_key[0]}:{old_key[1]}',slot_end))
                    old["temporary_until_s"]=min(old_deadline,until if new.get("handover_from_antenna_id") is not None else switch_start)
                    if new.get("handover_from_antenna_id") is not None:
                        old["standby_transfer_start_s"]=switch_start
                        old["standby_transfer_duration_s"]=float(new.get("estimated_switch_s") or 0.)
                    old_links.append(old)
            execution_links.append(new)
        self.state.switch_unavailable_until=switch_until
        # Minimum dwell is a limit on how often an antenna may change targets,
        # not a five-second outage for every newly selected feeder link.
        # Offline slot plans already contain the antenna handover assignment.
        self.state.switched_pairs=set()
        self.state.switch_effective_s=float(plan.get("valid_from_s",plan.get("generated_at_s",0)))
        execution_plan=plan.copy()
        if old_links:
            execution_plan["links"]=execution_links+old_links
            execution_plan["routes"]=list(plan.get("routes",[]))+list((previous_execution or previous or {}).get("routes",[]))
            execution_plan["planned_link_count_override"]=len(execution_links)
            execution_plan["ideal_capacity_override_gbps"]=sum(float(row["rate_gbps"]) for row in execution_links)
        self.state.plans=[plan];self.state.execution_plan=execution_plan;self.state.generated_at_s=float(plan.get("generated_at_s",0))
        self.state.last_transition_phase=-1
        self.state.last_invalid_route_count=0
        self.state.last_invalid_route_keys=set()
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

    def _evaluate(self,t,plan,pairs,total_demand,edges,current_traffic_satellite_count=None,positions=None,demand_by_source=None,link_cache=None):
        # A fast route repair changes routes, but not this frame's feeder links.
        # Keep the radio evaluation local to update(); never reuse it across frames.
        link_key=(id(plan["links"]),plan.get("rain_grade"),bool(plan.get("offline")),
                  plan.get("ideal_capacity_override_gbps"))
        cached=link_cache.get(link_key) if link_cache is not None else None
        if cached is not None:
            base_links,base_states,ideal,actual=cached
            active=[row.copy() for row in base_links]
            link_states=[row.copy() for row in base_states]
        else:
            active=[];link_states=[];ideal=float(plan.get("ideal_capacity_override_gbps",
                sum(x["rate_gbps"] for x in plan["links"])));actual=0
            for old in plan["links"]:
                if t+1e-9>=float(old.get("temporary_until_s",math.inf)):
                    continue
                unavailable_until=self.state.switch_unavailable_until.get((int(old["gateway"]),int(old["satellite"]),int(old.get("feeder_port",1))),0.)
                if t+1e-9<unavailable_until:
                    switch_start=old.get("scheduled_switch_start_s")
                    waiting=switch_start is not None and t+1e-9<float(switch_start)
                    link_states.append({"gateway":old["gateway"],"satellite":old["satellite"],"feeder_port":old.get("feeder_port",1),
                        "status":"等待切换" if waiting else "备用天线信道转接中" if old.get("handover_from_antenna_id") is not None else "切星捕获中","rate_gbps":0.,
                        "scheduled_switch_start_s":switch_start,"unavailable_until_s":round(unavailable_until,3)})
                    continue
                elev=pairs.get((old["gateway"],old["satellite"]))
                if elev is None:
                    link_states.append({"gateway":old["gateway"],"satellite":old["satellite"],"feeder_port":old.get("feeder_port",1),"status":"不可见","rate_gbps":0.})
                    continue
                grade=plan.get("rain_grade") if plan.get("offline") and self.config.rain_enabled else None
                rain=({"rain_rate_mmh":0.,"actual_attenuation_db":0.,"severity":grade or "中雨"} if self.config.rain_field=="synthetic_field" else
                      link_rain_state(self.stations[int(old["gateway"])],elev,t,self.config.rain_enabled,self.config.rain_sense_sigma,self.config.rain_forecast_hours,self.config.rain_use_p90))
                # Offline scenarios and realtime execution must use the same rain
                # field. Otherwise a light-rain plan can start with dead links.
                if grade and self.config.rain_field=="forecast":
                    fade=forecast_rain_attenuation_db(self.stations[int(old["gateway"])],elev,t)
                elif self.config.rain_enabled and self.config.rain_field=="synthetic_field":
                    fade=synthetic_rain_attenuation_db(self.stations[int(old["gateway"])],elev,grade or "中雨",t,list(positions[int(old["satellite"])]) if positions is not None else None)
                else:fade=rain["sensed_attenuation_db"]
                altitude=float(positions[int(old["satellite"])][4])/1000 if positions is not None else self.config.reference_satellite_altitude_km
                snr,mcs,rate=link_state(elev,self.config,fade,_slant_distance_km(altitude,elev))
                transfer_start=old.get("standby_transfer_start_s")
                if transfer_start is not None and t+1e-9>=float(transfer_start):
                    frame_s=max(1e-6,float(self.config.fast_evaluation_period_s))
                    transfer_s=max(0.,float(old.get("standby_transfer_duration_s",0.)))
                    rate*=1.-min(1.,transfer_s/frame_s)
                if rate<=0:
                    link_states.append({"gateway":old["gateway"],"satellite":old["satellite"],"feeder_port":old.get("feeder_port",1),"status":"容量为零","elevation_deg":elev,"rate_gbps":0.,"rain_attenuation_db":fade})
                    continue
                x=old.copy();x.update(elevation_deg=elev,snr_db=snr,modcod=mcs,rate_gbps=rate,gso_interference_db=round(gso_interference_penalty_db(self.config),3),rain_attenuation_db=fade,rain_actual_db=(fade if self.config.rain_field=="synthetic_field" or grade else rain["actual_attenuation_db"]),rain_rate_mmh=(forecast_rain_rate_mmh(self.stations[int(old["gateway"])],t) if grade and self.config.rain_field=="forecast" else synthetic_surface_rain_rate_mmh(self.stations[int(old["gateway"])],grade or "中雨",t) if self.config.rain_enabled and self.config.rain_field=="synthetic_field" else rain["rain_rate_mmh"]),rain_severity=(self.stations[int(old["gateway"])].get("rain_grade") or grade or rain["severity"]))
                if positions is not None:x["gsl_distance_km"]=_gsl_distance_km(positions[int(old["satellite"])][4],elev)
                active.append(x);actual+=rate
                status="容量下降" if rate<float(old.get("rate_gbps",0))*self.config.capacity_threshold_ratio else "可用"
                link_states.append({"gateway":old["gateway"],"satellite":old["satellite"],"feeder_port":old.get("feeder_port",1),"status":status,"elevation_deg":elev,"rate_gbps":rate,"rain_attenuation_db":fade})
            if link_cache is not None:
                link_cache[link_key]=([row.copy() for row in active],
                                      [row.copy() for row in link_states],ideal,actual)
        active_feeders={x["satellite"] for x in active}
        feeder_left,gsl_by_feeder=_feeder_resources(active)
        xyz_cache=link_cache.setdefault("xyz_cache",{}) if link_cache is not None else {}
        path_geometry=link_cache.setdefault("path_geometry",{}) if link_cache is not None else {}
        edge_distance=link_cache.setdefault("edge_distance",{}) if link_cache is not None else {}
        def xyz(satellite):
            satellite=int(satellite)
            if satellite not in xyz_cache:
                row=positions[satellite];xyz_cache[satellite]=_spherical_ecef(row[2],row[3],row[4])
            return xyz_cache[satellite]
        def edge_m(a,b):
            key=(min(a,b),max(a,b))
            distance=edge_distance.get(key)
            if distance is None:
                distance=math.dist(xyz(a),xyz(b))
                edge_distance[key]=distance
            return distance
        valid=[];source_left=dict(demand_by_source or {})
        for route in sorted(plan["routes"],key=lambda row:(bool(row.get("handover_hold",False)),float(row.get("priority_gbps",0)),int(row.get("route_continuity_rank",0))),reverse=True):
            feeder=int(route.get("feeder",-1))
            if feeder not in active_feeders or not all((min(a,b),max(a,b)) in edges for a,b in zip(route["path"],route["path"][1:])):continue
            source=int(route.get("source",-1))
            amount=min(float(route.get("demand_gbps",0)),feeder_left.get(feeder,0.),source_left.get(source,math.inf))
            if amount<=1e-9:continue
            row=route.copy();row["demand_gbps"]=round(amount,6)
            if "priority_gbps" in row:row["priority_gbps"]=round(min(amount,float(row["priority_gbps"])),6)
            if positions is not None:
                path=row.get("path",[])
                geometry_key=(link_key,feeder,tuple(path))
                geometry=path_geometry.get(geometry_key)
                if geometry is None:
                    isl_km=sum(edge_m(a,b) for a,b in zip(path,path[1:]))/1000
                    gsl_km=gsl_by_feeder.get(feeder,0.)
                    path_geometry[geometry_key]=(isl_km,gsl_km)
                else:isl_km,gsl_km=geometry
                row.update(distance_km=round(isl_km,3),isl_propagation_delay_ms=round(isl_km/C_KM_PER_MS,4),
                           gsl_distance_km=round(gsl_km,3),gsl_propagation_delay_ms=round(gsl_km/C_KM_PER_MS,4),
                           propagation_delay_ms=round((isl_km+gsl_km)/C_KM_PER_MS,4))
            valid.append(row)
            feeder_left[feeder]-=amount
            if source in source_left:source_left[source]-=amount
        # Demand may drift between cached routing snapshots. Use spare capacity
        # on already validated paths before considering a costly graph search.
        for row in valid:
            source=int(row["source"]);feeder=int(row["feeder"])
            extra=min(max(0.,source_left.get(source,0.)),max(0.,feeder_left.get(feeder,0.)))
            if extra<=1e-9:continue
            row["demand_gbps"]=round(float(row["demand_gbps"])+extra,6)
            source_left[source]-=extra;feeder_left[feeder]-=extra
        if self.config.feeder_link_mode=="dual":
            valid,link_usage=_bind_routes_to_links(valid,active)
        else:
            link_usage={}
        reachable=sum(x["demand_gbps"] for x in valid)
        if plan["routes"]:
            demand_by_feeder={}
            for route in valid:
                feeder=int(route.get("feeder",-1));demand_by_feeder[feeder]=demand_by_feeder.get(feeder,0.)+float(route.get("demand_gbps",0))
            for x in active:
                key=(int(x["gateway"]),int(x["satellite"]),int(x.get("feeder_port",1)))
                load=(link_usage.get(key,0.) if self.config.feeder_link_mode=="dual"
                      else demand_by_feeder.get(int(x["satellite"]),0.))
                x["scheduled_gbps"]=round(min(float(x["rate_gbps"]),load),6)
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
        if scheduled+1e-3<theoretical_deliverable and demand_by_source and positions is not None:
            bound_key=("component_bound",link_key)
            component_bound=link_cache.get(bound_key) if link_cache is not None else None
            if component_bound is None:
                component_bound=_component_delivery_bound(len(positions),edges,demand_by_source,active)
                if link_cache is not None:link_cache[bound_key]=component_bound
            theoretical_deliverable=min(theoretical_deliverable,
                component_bound)
        route_capacity_gap=max(0.,theoretical_deliverable-scheduled)
        route_satisfaction=1.0 if theoretical_deliverable<=1e-12 else min(1.,scheduled/theoretical_deliverable)
        return {"id":plan["id"],"name":plan["name"],"ideal_capacity_gbps":round(ideal,6),"actual_capacity_gbps":round(actual,6),
          "capacity_achievement_ratio":round(actual/max(ideal,1e-12),6),"scheduled_gbps":round(scheduled,6),
          "theoretical_deliverable_gbps":round(theoretical_deliverable,6),
          "component_capacity_shortfall_gbps":round(max(0.,float(total_demand)-theoretical_deliverable),6),
          "business_route_satisfaction_ratio":round(route_satisfaction,6),
          "route_capacity_gap_gbps":round(route_capacity_gap,6),
          "unserved_gbps":round(max(0,total_demand-scheduled),6),"active_links":active,"link_states":link_states,"planned_link_count":int(plan.get("planned_link_count_override",len(plan["links"]))),
          "active_link_count":len(active),"route_available_ratio":round(1.0 if denominator<=0 else min(1.,len(routed_sources)/denominator),6),
          "average_hops":avg,"business_weighted_hops":bweighted,"business_propagation_delay_ms":delay,
          "p95_propagation_delay_ms":p95_delay,"maximum_propagation_delay_ms":max_delay,"latency_scope":"服务卫星至信关站（ISL+馈电GSL）",
          "route_count":len(valid),"routed_source_count":len(routed_sources),"routes":valid}
    def update(self,t,positions,links,gsls,satellite_traffic,priority_traffic=None,priority_traffic_regions=None):
        update_started=time.perf_counter();route_rebuild_ms=0.;adjustment_ms=0.;route_rebuild_mode="无"
        initial_evaluation_ms=0.;final_evaluation_ms=0.;route_search_ms=0.
        graph_build_ms=0.;local_route_search_ms=0.;repair_source_count=0
        first_plan=not self.state.plans
        if self.config.offline_mode and first_plan:
            raise RuntimeError("尚未加载离线规划方案，请先执行离线规划")
        due=(not self.config.offline_mode) and (first_plan or t<self.state.generated_at_s or t-self.state.generated_at_s>=self.config.planning_period_s)
        planning_reason="首次生成慢周期资源方案" if first_plan else ("到达滚动规划周期" if due else "未到规划周期，沿用当前方案")
        if due:self.generate(t,positions,links,gsls,satellite_traffic,priority_traffic=priority_traffic,
                             priority_traffic_regions=priority_traffic_regions)
        evaluation_due=due or self.state.last_result is None or t<self.state.last_evaluation_s or t-self.state.last_evaluation_s>=self.config.fast_evaluation_period_s
        if not evaluation_due:
            result=dict(self.state.last_result);result["planning_event"]=False;result["scheduling_event"]=False;result["planning_reason"]="未到规划周期，沿用当前方案";result["connection_change_count"]=0
            return result
        pairs={(int(g),int(s)):float(e) for g,s,e in gsls};edges={(min(int(a),int(b)),max(int(a),int(b))) for a,b,*_ in links};access_demand=sum(float(x[1]) for x in satellite_traffic);demand=access_demand*self.config.traffic_landing_ratio
        demands={int(x[0]):float(x[1])*self.config.traffic_landing_ratio for x in satellite_traffic if float(x[1])>0}
        priority_demands={source:min(value,max(0.,float((priority_traffic or {}).get(source,0.))*self.config.traffic_landing_ratio))
                          for source,value in demands.items() if float((priority_traffic or {}).get(source,0.))>0.}
        current_source_count=len(demands);slow_plan=self.state.plans[0];plan=self.state.execution_plan or slow_plan
        phases=slow_plan.get("transition_routes") or []
        phase_changed=False
        if phases:
            phase=min(len(phases)-1,max(0,int((t-float(slow_plan.get("valid_from_s",t)))//5.)))
            if phase!=self.state.last_transition_phase:
                phase_changed=True
                planned_priority={int(source):float(amount) for source,amount in
                    slow_plan.get("priority_demands_by_satellite_gbps",{}).items()}
                # Preserve paths repaired during the previous phase if their
                # feeder and every ISL edge are still usable in this frame.
                # The prepared phase routes then act as successors, rather
                # than resetting all sources to a stale slot snapshot.
                usable_feeders={int(row["satellite"]) for row in plan.get("links",[])
                    if (int(row["gateway"]),int(row["satellite"])) in pairs
                    and t+1e-9<float(row.get("temporary_until_s",math.inf))
                    and t+1e-9>=self.state.switch_unavailable_until.get(
                        (int(row["gateway"]),int(row["satellite"]),int(row.get("feeder_port",1))),0.)}
                retained=[];seen=set()
                for row in plan.get("routes",[])[:int(plan.get("primary_route_count",len(plan.get("routes",[]))))]:
                    source=int(row.get("source",-1));feeder=int(row.get("feeder",-1))
                    path=row.get("path",[])
                    if source not in demands or feeder not in usable_feeders or not all(
                        (min(a,b),max(a,b)) in edges for a,b in zip(path,path[1:])):continue
                    key=(source,feeder,tuple(path))
                    if key in seen:continue
                    seen.add(key)
                    retained.append(dict(row,route_continuity_rank=2))
                retained_amount={};retained_expires=set()
                phase_end=min(float(slow_plan.get("valid_until_s",t+5.)),t+5.)
                feeder_expiry={}
                for link in plan.get("links",[]):
                    feeder=int(link["satellite"])
                    feeder_expiry[feeder]=max(feeder_expiry.get(feeder,0.),
                        float(link.get("temporary_until_s",math.inf)))
                for row in retained:
                    source=int(row["source"])
                    retained_amount[source]=retained_amount.get(source,0.)+float(row.get("demand_gbps",0.))
                    if feeder_expiry.get(int(row["feeder"]),0.)<=phase_end+1e-9:
                        retained_expires.add(source)
                prepared=[]
                for row in phases[phase]["routes"]:
                    source=int(row.get("source",-1));feeder=int(row.get("feeder",-1))
                    if source not in demands:continue
                    if retained_amount.get(source,0.)>=demands[source]-1e-6 and source not in retained_expires:
                        continue
                    key=(source,feeder,tuple(row.get("path",[])))
                    if key in seen:continue
                    seen.add(key)
                    prepared.append(dict(row,priority_gbps=round(min(float(row.get("demand_gbps",0)),
                        planned_priority.get(source,0.)),6),route_continuity_rank=1))
                primary=retained+prepared
                covered={int(row["source"]) for row in primary}
                fallback=[row for row in slow_plan.get("routes",[])+slow_plan.get("transition_fallback_routes",[])
                    if int(row.get("source",-1)) in demands and int(row.get("source",-1)) not in covered]
                plan=plan.copy();plan["routes"]=primary+fallback
                plan["primary_route_count"]=len(primary)
                self.state.execution_plan=plan
                self.state.last_transition_phase=phase
        if any(int(row.get("source",-1)) not in demands for row in plan.get("routes",[])):
            plan=plan.copy();plan["routes"]=[row for row in plan.get("routes",[]) if int(row.get("source",-1)) in demands]
            self.state.execution_plan=plan
        link_cache={}
        evaluation_started=time.perf_counter()
        active=self._evaluate(t,plan,pairs,demand,edges,current_source_count,positions,demands,link_cache);action="按慢环规划执行"
        initial_evaluation_ms=(time.perf_counter()-evaluation_started)*1000
        active_feeders={int(row["satellite"]) for row in active.get("active_links",[])}
        primary_routes=plan.get("routes",[])[:int(plan.get("primary_route_count",len(plan.get("routes",[]))))]
        invalid_route_keys={
            (int(row.get("source",-1)),int(row.get("feeder",-1)),tuple(row.get("path",[])))
            for row in primary_routes if
            int(row.get("feeder",-1)) not in active_feeders or
            any((min(a,b),max(a,b)) not in edges for a,b in zip(row.get("path",[]),row.get("path",[])[1:]))}
        invalid_route_count=len(invalid_route_keys)
        route_broken=invalid_route_count>0
        new_sources=set(demands)-self.state.last_route_sources
        demand_changed=(set(demands)!=set(self.state.last_demands) or any(abs(float(value)-float(self.state.last_demands.get(source,0)))>1e-6 for source,value in demands.items()))
        priority_changed=priority_demands!=getattr(self,"_last_priority_demands",{})
        route_capacity_missing=float(active.get("route_capacity_gap_gbps",0))>1e-3
        current_feeder_capacities={satellite:round(rate,6) for satellite,rate in
                                   _feeder_resources(active.get("active_links",[]))[0].items()}
        lost_active_feeder=bool(set(self.state.last_feeder_capacities)-set(current_feeder_capacities))
        newly_active_feeder=bool(set(current_feeder_capacities)-set(self.state.last_feeder_capacities))
        route_needs_repair=route_broken or bool(new_sources) or demand_changed or priority_changed or route_capacity_missing
        if self.config.selection_strategy=="slot_tracking":
            # Keep the regional 1–5 exit allocation intact until the next
            # scheduled slot; the evaluator still applies current rain and ACM.
            repair_interval_elapsed=t-self.state.last_route_update_s>=max(
                float(self.config.route_update_period_s),float(self.config.fast_adjust_period_s),1e-6)-1e-9
            slot_transition=bool(plan.get("offline")) and abs(t-float(plan.get("valid_from_s",-math.inf)))<1e-6
            newly_broken=bool(invalid_route_keys-self.state.last_invalid_route_keys)
            expected_delivery=(min(demand,float(phases[phase].get("routed_gbps",demand)))
                if phases else demand)
            # Small rounding/capacity differences must not launch a full
            # source-local graph search on every one-second frame.
            material_gap=max(0.01,0.001*demand)
            unexpected_gap=float(active.get("scheduled_gbps",0))+material_gap<expected_delivery
            recoverable_gap=float(active.get("route_capacity_gap_gbps",0))>material_gap
            previous_active=(self.state.last_result or {}).get("active_evaluation") or {}
            previous_unserved=float(previous_active.get("unserved_gbps",0)) if previous_active else 0.
            gap_worsened=(bool(previous_active) and
                float(active.get("unserved_gbps",0))>previous_unserved+material_gap)
            region_geometry={str(row.get("name")):row for row in self.config.priority_task_regions}
            regional_feeders=self._regional_feeders(active.get("active_links",[]),plan)
            priority_sources={int(source) for loads in (priority_traffic_regions or {}).values()
                              for source,amount in loads.items() if float(amount)>0}
            local_spare=False
            if priority_sources:
                for link in active.get("active_links",[]):
                    if float(link.get("rate_gbps",0))-float(link.get("scheduled_gbps",0))<=1e-3:continue
                    feeder=int(link["satellite"])
                    if any(feeder in feeders for feeders in regional_feeders.values()):
                        local_spare=True;break
            local_carried={};routed_priority_sources=set()
            if local_spare:
                for row in active.get("routes",[]):
                    source=int(row.get("source",-1))
                    if source not in priority_sources:continue
                    routed_priority_sources.add(source)
                    if any(source in loads and float(loads[source])>0 and
                           int(row["feeder"]) in regional_feeders.get(str(name),set())
                           for name,loads in (priority_traffic_regions or {}).items()):
                        local_carried[source]=local_carried.get(source,0.)+float(row["demand_gbps"])
            local_rebalance=bool(local_spare and any(
                priority_demands.get(source,0)-local_carried.get(source,0)>1e-3
                for source in routed_priority_sources))
            if phases:
                urgent_repair=(gap_worsened or newly_broken or lost_active_feeder or newly_active_feeder) and (unexpected_gap or recoverable_gap)
                routine_repair=(repair_interval_elapsed or slot_transition) and (unexpected_gap or recoverable_gap or local_rebalance)
                route_needs_repair=urgent_repair or routine_repair
            else:
                route_needs_repair=(repair_interval_elapsed or slot_transition or newly_broken or lost_active_feeder or newly_active_feeder) and (
                    route_capacity_missing or recoverable_gap or local_rebalance)
        graph=None
        fast_epoch=int(math.floor(t/max(self.config.fast_adjust_period_s,1e-6)))
        fast_tick=fast_epoch>self.state.last_fast_epoch
        if fast_tick:self.state.last_fast_epoch=fast_epoch
        if fast_tick and route_needs_repair:
            route_started=time.perf_counter()
            route_rebuild_mode=("紧急修补" if self.config.selection_strategy=="slot_tracking" and
                (gap_worsened or newly_broken or lost_active_feeder or newly_active_feeder)
                else "周期修补" if self.config.selection_strategy=="slot_tracking" else "局部修补")
            usable_links=list(active.get("active_links",[]))
            capacities,gsl_distances=_feeder_resources(usable_links)
            available={feeder:value for feeder,value in capacities.items() if value>1e-12}
            source_left=dict(demands);feeder_left=dict(available);retained=[];existing_exits={}
            # Active feeder satellites land their own traffic directly before
            # any retained ISL route can use that feeder's capacity.
            old_local={int(row["source"]):row for row in active.get("routes",[])
                       if int(row.get("source",-1))==int(row.get("feeder",-2))}
            for source in sorted(set(source_left)&set(feeder_left),key=lambda source:priority_demands.get(source,0.),reverse=True):
                amount=min(source_left[source],feeder_left[source])
                if amount<=1e-9:continue
                row=old_local.get(source,{"source":source,"feeder":source,"path":[source],"hops":0,
                    "distance_km":0.,"isl_propagation_delay_ms":0.,"propagation_delay_ms":0.}).copy()
                row["demand_gbps"]=round(amount,6)
                retained.append(row);source_left[source]-=amount;feeder_left[source]-=amount
                existing_exits.setdefault(source,set()).add(source)
            for original in sorted(active.get("routes",[]),key=lambda row:priority_demands.get(int(row.get("source",-1)),0.),reverse=True):
                source=int(original.get("source",-1));feeder=int(original.get("feeder",-1))
                amount=min(float(original.get("demand_gbps",0)),source_left.get(source,0.),feeder_left.get(feeder,0.))
                if amount<=1e-9:continue
                row=original.copy();row["demand_gbps"]=round(amount,6);retained.append(row)
                source_left[source]-=amount;feeder_left[feeder]-=amount
                existing_exits.setdefault(source,set()).add(feeder)
            # Business changes every second. Fill the increment on an already
            # valid path before launching any new graph search.
            for row in sorted(retained,key=lambda row:priority_demands.get(int(row.get("source",-1)),0.),reverse=True):
                source=int(row["source"]);feeder=int(row["feeder"])
                increment=min(source_left.get(source,0.),feeder_left.get(feeder,0.))
                if increment<=1e-9:continue
                row["demand_gbps"]=round(float(row["demand_gbps"])+increment,6)
                source_left[source]-=increment;feeder_left[feeder]-=increment
            repair_demands={source:amount for source,amount in source_left.items() if amount>1e-6}
            repair_source_count=len(repair_demands)
            search_capacities={feeder:amount for feeder,amount in feeder_left.items() if amount>1e-6}
            region_geometry={str(row.get("name")):row for row in self.config.priority_task_regions}
            regional_feeders=self._regional_feeders(active.get("active_links",[]),plan)
            local_feeders_by_source={}
            for region_name,region_loads in (priority_traffic_regions or {}).items():
                region=region_geometry.get(str(region_name))
                if region is None:continue
                nearby=set(available).intersection(regional_feeders.get(str(region_name),set()))
                for source,amount in region_loads.items():
                    if float(amount)>0:local_feeders_by_source.setdefault(int(source),set()).update(nearby)
            search_started=time.perf_counter()
            if repair_demands and search_capacities:
                graph=_graph(positions,links)
                components=_component_ids(graph)
                graph_build_ms=(time.perf_counter()-search_started)*1000
                feeder_components={components[feeder] for feeder in search_capacities}
                repair_demands={source:amount for source,amount in repair_demands.items()
                                if components[source] in feeder_components}
                repaired=_repair_routes_local(graph,dict(sorted(repair_demands.items(),key=lambda item:priority_demands.get(item[0],0.),reverse=True)),search_capacities,gsl_distances,
                    1.,0.,self.config.delay_reference_ms,excluded_feeders=existing_exits,
                    exit_limits={source:max(0,self.config.realtime_exit_limit_per_source-len(existing_exits.get(source,set()))) for source in repair_demands},
                    max_visited_per_source=500,max_exits_per_source=self.config.realtime_exit_limit_per_source,
                    priority_demands=priority_demands,local_feeders_by_source=local_feeders_by_source,
                    component_ids=components)
            else:repaired=[]
            route_search_ms=(time.perf_counter()-search_started)*1000
            local_route_search_ms=max(0.,route_search_ms-graph_build_ms)
            new_routes=_compact_routes(retained+repaired,demands,self.config.realtime_exit_limit_per_source)
            if self.config.selection_strategy=="slot_tracking" and priority_demands and local_feeders_by_source:
                if graph is None:graph=_graph(positions,links)
                new_routes=_rebalance_priority_to_local(new_routes,demands,priority_demands,
                    local_feeders_by_source,available,graph,gsl_distances,
                    self.config.realtime_exit_limit_per_source,
                    component_ids=components if repair_demands and search_capacities else None)
            plan=plan.copy();plan["routes"]=new_routes
            plan["primary_route_count"]=len(new_routes)
            _attach_feeder_delay(plan["routes"],usable_links)
            self.state.execution_plan=plan
            evaluation_started=time.perf_counter()
            active=self._evaluate(t,plan,pairs,demand,edges,current_source_count,positions,demands,link_cache)
            final_evaluation_ms=(time.perf_counter()-evaluation_started)*1000
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
        self._last_priority_demands=dict(priority_demands)
        # Record the paths actually installed after a fast repair. Keeping the
        # pre-repair set would mistake a repaired path for a fresh failure on
        # the next frame.
        if fast_tick and route_needs_repair:
            active_feeders={int(row["satellite"]) for row in active.get("active_links",[])}
            primary_routes=plan.get("routes",[])[:int(plan.get("primary_route_count",len(plan.get("routes",[]))))]
            invalid_route_keys={
                (int(row.get("source",-1)),int(row.get("feeder",-1)),tuple(row.get("path",[])))
                for row in primary_routes if
                int(row.get("feeder",-1)) not in active_feeders or
                any((min(a,b),max(a,b)) not in edges for a,b in zip(row.get("path",[]),row.get("path",[])[1:]))}
            invalid_route_count=len(invalid_route_keys)
        self.state.last_invalid_route_count=invalid_route_count
        self.state.last_invalid_route_keys=invalid_route_keys
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
            self.generate(t,positions,links,gsls,satellite_traffic,priority_traffic=priority_traffic,
                          priority_traffic_regions=priority_traffic_regions);slow_plan=self.state.plans[0];plan=self.state.execution_plan or slow_plan
            active=self._evaluate(t,plan,pairs,demand,edges,current_source_count,positions,demands,link_cache);due=True;planning_reason="到达慢规划周期";action="执行慢周期全局重规划"
        pairs_now={(x["gateway"],x["satellite"],x.get("feeder_port",1))
                   if self.config.feeder_link_mode=="dual" else (x["gateway"],x["satellite"])
                   for x in (active["active_links"] if active else [])}
        change_count=len(pairs_now.symmetric_difference(self.state.last_scheduled_pairs)) if self.state.last_scheduled_pairs else 0;changed=change_count>0
        if changed:self.state.switch_count+=change_count;self.state.last_switch_s=t
        self.state.last_scheduled_pairs=pairs_now
        loss=0.
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
        priority_demand=sum(priority_demands.values())
        priority_delay_amount=0.;priority_delay_total=0.;priority_left=dict(priority_demands)
        priority_total_left=dict(priority_demands);priority_total_delivered=0.
        region_geometry={str(item.get("name")):item for item in self.config.priority_task_regions}
        current_positions={int(row[0]):row for row in positions}
        regional_local_feeders=self._regional_feeders((active or {}).get("active_links",[]),plan)
        for route in (active or {}).get("routes",[]):
            source=int(route.get("source",-1))
            total_amount=min(priority_total_left.get(source,0.),float(route.get("demand_gbps",0.)))
            priority_total_delivered+=total_amount
            priority_total_left[source]=max(0.,priority_total_left.get(source,0.)-total_amount)
            if self.config.selection_strategy=="slot_tracking":
                local=any(int(route.get("feeder",-1)) in regional_local_feeders.get(str(name),set())
                    and source in region_loads and float(region_loads[source])>0
                    for name,region_loads in (priority_traffic_regions or {}).items()
                    for geometry in [region_geometry.get(str(name))] if geometry is not None)
                amount=min(priority_left.get(source,0.),float(route.get("demand_gbps",0))) if local else 0.
                route["priority_gbps"]=round(amount,6)
            else:
                amount=min(priority_left.get(source,0.),float(route.get("demand_gbps",0.)))
            priority_delay_total+=amount*float(route.get("propagation_delay_ms",0.))
            priority_delay_amount+=amount;priority_left[source]=max(0.,priority_left.get(source,0.)-amount)
        priority_delivered=priority_delay_amount
        priority_region_status={}
        for name,loads in (priority_traffic_regions or {}).items():
            region_demand=0.;region_total=0.;region_total_delay_sum=0.
            routes_by_source={}
            for route in (active or {}).get("routes",[]):
                source=int(route.get("source",-1))
                if float(loads.get(source,0.))<=0:continue
                routes_by_source.setdefault(source,[]).append(route)
            for source,raw_amount in loads.items():
                source=int(source)
                source_demand=max(0.,float(demands.get(source,0.)))
                amount=min(source_demand,max(0.,float(raw_amount))*self.config.traffic_landing_ratio)
                if amount<=0 or source_demand<=0:continue
                region_demand+=amount
                region_total+=amount*min(1.,allocated_by_source.get(source,0.)/source_demand)
                region_total_delay_sum+=sum(
                    amount*max(0.,float(route.get("demand_gbps",0.)))/source_demand*
                    float(route.get("propagation_delay_ms",0.))
                    for route in routes_by_source.get(source,()))
            priority_region_status[str(name)]={
                "demand_gbps":round(region_demand,6),"total_delivered_gbps":round(region_total,6),
                "average_delay_ms":round(region_total_delay_sum/region_total,4) if region_total else 0.}
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
          "priority_demand_gbps":round(priority_demand,6),"priority_delivered_gbps":round(priority_delivered,6),
          "priority_total_delivered_gbps":round(priority_total_delivered,6),
          "priority_nonlocal_delivered_gbps":round(max(0.,priority_total_delivered-priority_delivered),6),
          "priority_delivery_ratio":round(priority_delivered/max(priority_demand,1e-12),6) if priority_demand else 1.,
          "priority_average_delay_ms":round(priority_delay_total/max(priority_delay_amount,1e-12),4) if priority_delay_amount else 0.,
          "priority_region_status":priority_region_status,
          "isl_count":len(links),"gsl_candidate_count":len(gsls),
          "visible_feeder_satellite_count":len({int(x[1]) for x in gsls}),
          "gateway_count":len(self.stations),"planned_link_count":int(plan.get("planned_link_count_override",len(plan.get("links",[])))),
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
          "initial_evaluation_ms":round(initial_evaluation_ms,3),
          "final_evaluation_ms":round(final_evaluation_ms,3),
          "route_search_ms":round(route_search_ms,3),
          "graph_build_ms":round(graph_build_ms,3),
          "local_route_search_ms":round(local_route_search_ms,3),
          "repair_source_count":repair_source_count,
          "feeder_adjustment_ms":round(adjustment_ms,3),
          "source_exit_hold_s":round(self.config.source_exit_hold_s,3),
          "planner_update_ms":round((time.perf_counter()-update_started)*1000,3)}
        # Transition route sets belong to the offline catalog. Copying them
        # into every displayed/stored simulation frame multiplies memory and I/O.
        frame_plan_keys={"transition_routes","transition_fallback_routes","demand_by_source_gbps","pointing_traces"}
        def frame_plan(source):
            return {key:value for key,value in source.items() if key not in frame_plan_keys}
        result={"generation":self.state.generation,"generated_at_s":self.state.generated_at_s,"planning_period_s":self.config.planning_period_s,
          "planning_horizon_s":self.config.planning_horizon_s,"planning_slot_s":self.config.planning_slot_s,
          "fast_evaluation_period_s":self.config.fast_evaluation_period_s,"fast_adjust_period_s":self.config.fast_adjust_period_s,
          "route_update_period_s":self.config.route_update_period_s,"freeze_window_s":self.config.freeze_window_s,
          "capacity_threshold_ratio":threshold,"active_plan_id":self.state.active_plan_id,"last_switch_s":self.state.last_switch_s,
          "switch_count":self.state.switch_count,"switching_loss_gbps":round(loss,6),"delivered_gbps":round(min(demand,max(0,(active["scheduled_gbps"] if active else 0)-loss)),6),
          "action":action,"planning_reason":planning_reason,"connection_change_count":change_count,"planning_event":due,"scheduling_event":True,
          "rain_summary":self._rain_summary(active),"station_weather":self._station_weather(t,slow_plan),
          "next_planning_time_s":round(self.state.generated_at_s+self.config.planning_period_s,6),
          "plans":[frame_plan(item) for item in self.state.plans],"evaluations":[active],"active_plan":frame_plan(slow_plan),"execution_plan":frame_plan(plan),
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
            elif grade and self.config.rain_field=="forecast":
                fade=forecast_rain_attenuation_db(station,45.,t)
            elif self.config.rain_field=="synthetic_field":
                fade=synthetic_rain_attenuation_db(station,45.,grade or "中雨",t)
            else:
                rain=link_rain_state(station,45.,t,True,self.config.rain_sense_sigma,self.config.rain_forecast_hours,self.config.rain_use_p90)
                fade=float(rain["sensed_attenuation_db"])
            # Attenuation at one reference elevation is not an observed rain
            # rate.  Keep the selected scenario separate from the local fade.
            output.append({"gateway":gid,"reference_attenuation_db":round(fade,3),
                           "reference_elevation_deg":45.,"source":"空间雨强场（未标定）" if self.config.rain_field=="synthetic_field" else "IMERG 历史半小时降水" if grade and self.config.rain_field=="forecast" else "模型推算","scenario":grade or "中度雨衰",
                           "observed_rain_rate_available":self.config.rain_field=="forecast",
                           "rain_rate_mmh":forecast_rain_rate_mmh(station,t) if grade and self.config.rain_field=="forecast" else None})
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
