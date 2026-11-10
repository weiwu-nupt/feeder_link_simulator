"""Offline resource planning for preconfigured gateway stations."""
from __future__ import annotations

import math
from dataclasses import replace
from .rain_model import link_rain_state

EARTH_RADIUS_M = 6_378_135.0


def _distance_km(a, b):
    lat1, lon1, lat2, lon2 = map(math.radians, (a["lat"], a["lon"], b["lat"], b["lon"]))
    value = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    return 2*6371.0*math.asin(min(1.0, math.sqrt(value)))


def plan_gateway_resources(stations, orbit_cache, min_elevation_deg=20.0,
                           station_capacity_gbps=80.0, demand_gbps=1000.0,
                           max_gsl_km=0.0, traffic_cells=None,
                           rain_enabled=True,rain_sense_sigma=.20,rain_forecast_hours=2.,
                           rain_use_p90=True,link_config=None):
    """Plan resources for every preconfigured station without selecting sites."""
    sites=[dict(x) for x in stations]
    if not sites:
        return {"status":"未配置信关站","algorithm":"固定站点资源规划","stations":[],"planned_stations":[]}
    import numpy as np
    frame_count=len(orbit_cache) if orbit_cache is not None else 0
    sample_count=min(61,frame_count)
    sample_indices=sorted({round(i*(frame_count-1)/max(1,sample_count-1)) for i in range(sample_count)}) if frame_count else []
    weighted_cells=[x for x in (traffic_cells or []) if float(getattr(x,"base_weight",0))>0]
    weight_total=sum(float(x.base_weight) for x in weighted_cells)
    metrics=[]
    for index,site in enumerate(sites):
        counts=[]; planning_fades=[]; frame_nominal_rates=[]; frame_interference_rates=[]; frame_combined_rates=[]
        channel_limit=max(1,int(site.get("channel_count",site.get("antenna_count",20))))
        no_gso_config=replace(link_config,geo_interference_enabled=False) if link_config is not None else None
        for frame_index in sample_indices:
            frame=orbit_cache[frame_index];lat=math.radians(float(site["lat"]));lon=math.radians(float(site["lon"]))
            slat=np.radians(frame[:,0]);slon=np.radians(frame[:,1]);radius=EARTH_RADIUS_M+np.maximum(0.0,frame[:,2])
            cosine=np.clip(math.sin(lat)*np.sin(slat)+math.cos(lat)*np.cos(slat)*np.cos(slon-lon),-1.0,1.0);central=np.arccos(cosine)
            elevation=np.degrees(np.arctan2(radius*np.cos(central)-EARTH_RADIUS_M,radius*np.sin(central)))
            slant=np.sqrt(radius*radius+EARTH_RADIUS_M*EARTH_RADIUS_M-2*radius*EARTH_RADIUS_M*np.cos(central))/1000.0
            valid=elevation>=min_elevation_deg
            if max_gsl_km and max_gsl_km>0:valid&=slant<=max_gsl_km
            counts.append(int(np.count_nonzero(valid)))
            if np.any(valid):
                # Aggregate concurrent station channels.  One marginal, low
                # elevation link must not represent or disable the whole site.
                visible_elevations=np.sort(elevation[valid])[-channel_limit:]
                nominal_sum=interference_sum=combined_sum=0.0
                for visible_elevation in visible_elevations:
                    rain=link_rain_state(site,float(visible_elevation),float(frame_index),rain_enabled,
                                         rain_sense_sigma,rain_forecast_hours,rain_use_p90)
                    raw_fade=float(rain["planning_attenuation_db"])
                    planning_fades.append(raw_fade)
                    # The offline catalog explicitly evaluates light/medium/
                    # heavy rain at 1/3/6 dB.  Site sizing must use the same
                    # design envelope; an uncalibrated synthetic two-hour
                    # forecast previously produced 50--70 dB and reduced most
                    # candidate sites to zero capacity before those scenarios
                    # were evaluated.
                    fade=min(raw_fade,6.0)
                    if link_config is not None:
                        from .planning import link_state
                        nominal_sum+=float(link_state(float(visible_elevation),no_gso_config,0.)[2])
                        interference_sum+=float(link_state(float(visible_elevation),link_config,0.)[2])
                        combined_sum+=float(link_state(float(visible_elevation),link_config,fade)[2])
                if link_config is not None:
                    frame_nominal_rates.append(nominal_sum)
                    frame_interference_rates.append(interference_sum)
                    frame_combined_rates.append(combined_sum)
        availability=sum(value>0 for value in counts)/max(1,len(counts))
        affinity=(sum(float(x.base_weight)*math.exp(-_distance_km(site,{"lat":x.latitude_deg,"lon":x.longitude_deg})/1800.0) for x in weighted_cells)/weight_total) if weight_total else 1.0
        sorted_fades=sorted(planning_fades);p90_fade=sorted_fades[min(len(sorted_fades)-1,int(.9*len(sorted_fades)))] if sorted_fades else 0.
        if link_config is not None and frame_nominal_rates:
            from .planning import gso_interference_penalty_db
            # P10 station capacity is exceeded in about 90% of sampled instants.
            # Calculate it after summing channels, rather than applying one
            # adverse link's fade to every channel in the station.
            frame_gso_factors=[i/max(n,1e-12) for n,i in zip(frame_nominal_rates,frame_interference_rates)]
            frame_rain_factors=[c/max(i,1e-12) for i,c in zip(frame_interference_rates,frame_combined_rates)]
            frame_combined_factors=[c/max(n,1e-12) for n,c in zip(frame_nominal_rates,frame_combined_rates)]
            gso_capacity_factor=max(0.,min(1.,float(np.percentile(frame_gso_factors,10))))
            rain_capacity_factor=max(0.,min(1.,float(np.percentile(frame_rain_factors,10))))
            combined_capacity_factor=max(0.,min(1.,float(np.percentile(frame_combined_factors,10))))
            robust_station_capacity_gbps=max(0.,float(np.percentile(frame_combined_rates,10)))
            gso_penalty_db=float(gso_interference_penalty_db(link_config))
        else:
            gso_capacity_factor=1.;gso_penalty_db=0.
            rain_capacity_factor=max(.35,min(1.,1.-p90_fade/20.));combined_capacity_factor=rain_capacity_factor
            robust_station_capacity_gbps=station_capacity_gbps*combined_capacity_factor
        rain_suitability=rain_capacity_factor
        metrics.append({"gateway":index,"name":site.get("name",f"信关站{index+1}"),"lat":float(site["lat"]),"lon":float(site["lon"]),
                        "availability_ratio":round(availability,6),"average_visible_satellites":round(sum(counts)/max(1,len(counts)),2),
                        "minimum_visible_satellites":min(counts) if counts else 0,"maximum_visible_satellites":max(counts) if counts else 0,
                        "business_affinity":round(affinity,6),"planning_rain_p90_db":round(p90_fade,3),
                        "rain_suitability":round(rain_suitability,6),"rain_capacity_factor":round(rain_capacity_factor,6),
                        "gso_interference_db":round(gso_penalty_db,3),"gso_capacity_factor":round(gso_capacity_factor,6),
                        "combined_capacity_factor":round(combined_capacity_factor,6),
                        "robust_station_capacity_gbps":round(robust_station_capacity_gbps,6)})
    raw=[max(1e-6,m["availability_ratio"])*max(.05,m["business_affinity"])*max(.1,m["rain_suitability"]) for m in metrics];raw_total=sum(raw);planned=[]
    for index,(site,metric) in enumerate(zip(sites,metrics)):
        share=raw[index]/raw_total if raw_total else 1/len(sites);assigned=demand_gbps*share
        antenna_limit=max(1,int(site.get("antenna_count",20)));channel_limit=max(1,int(site.get("channel_count",antenna_limit)))
        impaired_capacity=float(metric.get("robust_station_capacity_gbps",
                                            station_capacity_gbps*float(metric.get("combined_capacity_factor",1.))))
        # Site construction fixes the available antenna/channel pool.  Traffic
        # assignment is an operating target, not a reason to permanently remove
        # idle channels from the deployed station.  The scheduler may activate
        # any configured channel when routes, rain or interference change.
        channel_capacity=impaired_capacity/max(1,channel_limit)
        planned_channels=channel_limit;planned_antennas=antenna_limit;designed=impaired_capacity
        alternatives=[j for j in range(len(sites)) if j!=index]
        backup=max(alternatives,key=lambda j:(_distance_km(site,sites[j]),metrics[j]["availability_ratio"])) if alternatives else None
        station=dict(site);station.update(configured_antenna_count=antenna_limit,configured_channel_count=channel_limit,
            antenna_count=planned_antennas,channel_count=planned_channels,planning_role="固定部署站")
        planned.append(station);metric.update(planned=True,traffic_share=round(share,6),assigned_demand_gbps=round(assigned,3),
            planned_antennas=planned_antennas,planned_channels=planned_channels,designed_capacity_gbps=round(designed,3),
            backup_gateway=backup,backup_name=sites[backup].get("name",f"信关站{backup+1}") if backup is not None else "--")
    designed_total=sum(x["designed_capacity_gbps"] for x in metrics)
    return {"status":"已生成固定信关站资源规划" if designed_total>=demand_gbps else "已生成资源规划，但配置容量存在缺口",
            "algorithm":"固定站点的时域可见性、业务分担与资源容量规划","sampled_frames":len(sample_indices),
            "station_count":len(sites),"target_demand_gbps":round(float(demand_gbps),3),"designed_capacity_gbps":round(designed_total,3),
            "capacity_gap_gbps":round(max(0.0,demand_gbps-designed_total),3),"capacity_satisfied":designed_total>=demand_gbps,
            "stations":metrics,"planned_stations":planned}


def plan_gateway_automatic(orbit_cache, min_elevation_deg=20.0, station_capacity_gbps=80.0,
                           demand_gbps=1000.0, max_gsl_km=0.0, traffic_cells=None,
                           maximum_sites=12, redundancy=2, antenna_limit=20,
                           rain_enabled=True,rain_sense_sigma=.20,rain_forecast_hours=2.,
                           rain_use_p90=True,link_config=None,minimum_sites=5):
    """Create an internal China search grid and select a deployment automatically."""
    # These coordinates represent city-level planning anchors rather than exact
    # construction parcels.  Traffic-cell centroids are deliberately excluded:
    # without a DEM/water/land-use mask they could fall on rivers or steep terrain.
    anchors=(("佳木斯",46.80,130.32),("北京",39.90,116.40),("喀什",39.47,75.99),("拉萨",29.65,91.17),("三亚",18.25,109.51),
             ("上海",31.23,121.47),("广州",23.13,113.26),("成都",30.57,104.07),("西安",34.34,108.94),
             ("乌鲁木齐",43.83,87.62),("哈尔滨",45.80,126.53),("沈阳",41.81,123.43),("呼和浩特",40.84,111.75),
             ("济南",36.65,117.12),("南京",32.06,118.80),("武汉",30.59,114.30),("重庆",29.56,106.55),
             ("昆明",25.04,102.71),("兰州",36.06,103.83),("郑州",34.75,113.62),("南宁",22.82,108.32),
             ("福州",26.08,119.30),("长沙",28.23,112.94),("合肥",31.82,117.23),("银川",38.49,106.23))
    candidates=[{"name":name,"lat":lat,"lon":lon,"alt":0,"antenna_count":int(antenna_limit),"channel_count":int(antenna_limit),"site_basis":"城市级基础设施规划锚点"} for name,lat,lon in anchors]
    rain_args=(rain_enabled,rain_sense_sigma,rain_forecast_hours,rain_use_p90,link_config)
    evaluated=plan_gateway_resources(candidates,orbit_cache,min_elevation_deg,station_capacity_gbps,demand_gbps,max_gsl_km,traffic_cells,*rain_args)
    metrics=evaluated["stations"];selected=[];remaining=set(range(len(candidates)))
    weighted_cells=[x for x in (traffic_cells or []) if float(getattr(x,"base_weight",0))>0]
    business_weights=[float(x.base_weight) for x in weighted_cells];business_total=max(sum(business_weights),1e-12)
    coverage=[[math.exp(-_distance_km(site,{"lat":cell.latitude_deg,"lon":cell.longitude_deg})/1800.0) for cell in weighted_cells] for site in candidates]
    covered=[0.]*len(weighted_cells);selected_marginal_gain={}
    minimum_site_separation_km=600.0
    def marginal_business_gain(index):
        return sum(weight*(max(old,new)-old) for weight,old,new in zip(business_weights,covered,coverage[index]))/business_total if weighted_cells else 1.
    def choose_site():
        best=None
        for i in remaining:
            nearest_selected=math.inf if not selected else min(_distance_km(candidates[i],candidates[j]) for j in selected)
            # 相邻站点具有高度相关的可见星、雨区和业务覆盖，不能把两个
            # 独立站位名额消耗在同一局部区域，因此站间距采用硬约束。
            if nearest_selected<minimum_site_separation_km:continue
            diversity=1.0 if not selected else min(1.0,nearest_selected/1800.0)
            score=(.40*marginal_business_gain(i)+.25*metrics[i].get("combined_capacity_factor",1.)+
                   .20*metrics[i]["availability_ratio"]+.15*diversity)
            value=(score,metrics[i]["minimum_visible_satellites"],-i)
            if best is None or value>best[0]:best=(value,i)
        # The minimum site count and geographic redundancy are hard planning
        # constraints.  A site with a poor P10 weather/interference factor must
        # remain selectable as a diversity site; rejecting it here previously
        # stopped automatic planning after the first positive-capacity site.
        if best is None:
            return False
        chosen=best[1];selected_marginal_gain[chosen]=marginal_business_gain(chosen);selected.append(chosen);remaining.remove(chosen)
        for cell_index,value in enumerate(coverage[chosen]):covered[cell_index]=max(covered[cell_index],value)
        return True
    positive_capacities=sorted((float(x.get("robust_station_capacity_gbps",0.)) for x in metrics
                                if float(x.get("robust_station_capacity_gbps",0.))>0.01),reverse=True)
    accumulated=0.; capacity_required=0
    for capacity in positive_capacities:
        accumulated+=capacity; capacity_required+=1
        if accumulated>=demand_gbps: break
    if accumulated<demand_gbps: capacity_required=int(maximum_sites)
    required=max(int(minimum_sites),int(redundancy),capacity_required)
    target=min(max(1,int(maximum_sites)),max(1,required),len(candidates))
    while remaining and len(selected)<target:
        if not choose_site(): break
    selected_sites=[candidates[i] for i in selected]
    result=plan_gateway_resources(selected_sites,orbit_cache,min_elevation_deg,station_capacity_gbps,demand_gbps,max_gsl_km,traffic_cells,*rain_args)
    # The coarse ceil(demand / nominal station capacity) estimate can still
    # under-provision the selected set when traffic affinity overloads one site.
    # Expand until actual planned channel capacity covers demand or the site
    # limit is reached.
    while not result.get("capacity_satisfied",False) and remaining and len(selected)<min(int(maximum_sites),len(candidates)):
        if not choose_site(): break
        selected_sites=[candidates[i] for i in selected]
        result=plan_gateway_resources(selected_sites,orbit_cache,min_elevation_deg,station_capacity_gbps,demand_gbps,max_gsl_km,traffic_cells,*rain_args)
    result.update(mode="automatic",algorithm="业务覆盖、时域可见性、雨衰与干扰联合规划",
                  search_point_count=len(candidates),required_station_count=required,maximum_sites=int(maximum_sites),redundancy=int(redundancy),
                  minimum_sites=int(minimum_sites),minimum_site_separation_km=minimum_site_separation_km,
                  station_count_rule=f"按可靠链路容量覆盖{demand_gbps:.2f} Gb/s规划目标，至少{int(minimum_sites)}站，站间距不少于{minimum_site_separation_km:.0f} km，受上限{int(maximum_sites)}站约束",
                  limitations="当前输出为城市级规划位置，并非精确建设坐标；精确选址仍需接入DEM坡度、水体、土地利用、实测雨强、电力和骨干网数据")
    selected_original=[selected_sites[i].get("name","") for i in range(len(selected_sites))]
    for order,station in enumerate(result["planned_stations"]):
        original=selected_original[order];station["name"]=f"GW-A{order+1:02d} {original}"
    for order,row in enumerate(result["stations"]):
        original=selected_original[order];row["name"]=f"GW-A{order+1:02d} {original}"
        original_index=selected[order] if order<len(selected) else -1
        row["marginal_business_coverage"]=round(float(selected_marginal_gain.get(original_index,0.)),6)
        row["selection_reason"]=(f"新增业务覆盖贡献{100*row['marginal_business_coverage']:.1f}%；"
                                 f"可用率{100*row.get('availability_ratio',0):.1f}%；"
                                 f"可靠容量{row.get('designed_capacity_gbps',0):.1f} Gb/s")
        if row.get("backup_gateway") is not None:row["backup_name"]=f"GW-A{int(row['backup_gateway'])+1:02d}"
    return result
