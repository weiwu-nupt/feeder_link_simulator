"""ITU-R based rain attenuation for planning and real-time scheduling.

P.618 supplies the slant-path procedure, P.838 the specific attenuation law,
and P.837 the station R0.01 input. Static coefficients and geometry are cached.
"""
from __future__ import annotations
import math
import bisect
import hashlib
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np

_P838_CURVES={
 "kh":((-5.33980,-.10008,1.13098),(-.35351,1.26970,.45400),(-.23789,.86036,.15354),(-.94158,.64552,.16817),-.18961,.71147),
 "kv":((-3.80595,.56934,.81061),(-3.44965,-.22911,.51059),(-.39902,.73042,.11899),(.50167,1.07319,.27195),-.16398,.63297),
 "ah":((-.14318,1.82442,-.55187),(.29591,.77564,.19822),(.32177,.63773,.13164),(-5.37610,-.96230,1.47828),(16.1721,-3.29980,3.43990),.67849,-1.95537),
 "av":((-.07771,2.33840,-.76284),(.56727,.95545,.54039),(-.20238,1.14520,.26809),(-48.2991,.791669,.116226),(48.5833,.791459,.116479),-.053739,.83433)}

def _curve(frequency,name):
    x=math.log10(max(1.,min(1000.,float(frequency))));parts=_P838_CURVES[name];m,c=parts[-2:]
    value=sum(a*math.exp(-((x-b)/c0)**2) for a,b,c0 in parts[:-2])+m*x+c
    return 10**value if name in ("kh","kv") else value

@lru_cache(maxsize=256)
def _p838(f_mghz,tau_tenths,elev_tenths):
    f=f_mghz/1000.;tau=math.radians(tau_tenths/10.);elev=math.radians(elev_tenths/10.)
    kh,ah,kv,av=_curve(f,"kh"),_curve(f,"ah"),_curve(f,"kv"),_curve(f,"av")
    c=math.cos(elev)**2*math.cos(2*tau);k=(kh+kv+(kh-kv)*c)/2
    alpha=(kh*ah+kv*av+(kh*ah-kv*av)*c)/(2*k)
    return k,alpha

_ITU_DATA_ROOT=Path(__file__).resolve().parents[2]/"link_budget"/"algorithm"/"data"

@lru_cache(maxsize=2)
def _itu_grid(name):
    import numpy as np
    if name=="r001":
        path=_ITU_DATA_ROOT/"P.837_R001_Maps"/"R001.TXT"
        shape=(1441,2881)
    else:
        path=_ITU_DATA_ROOT/"P.839-4 rain_height"/"h0.txt"
        shape=(121,241)
    grid=np.loadtxt(path,dtype=np.float32)
    if grid.shape!=shape:
        raise ValueError(f"Unexpected ITU map shape {path}: {grid.shape}, expected {shape}")
    return grid

def _bilinear(grid,row,col):
    i=min(max(int(math.floor(row)),0),grid.shape[0]-2)
    j=min(max(int(math.floor(col)),0),grid.shape[1]-2)
    a=max(0.,min(1.,row-i));b=max(0.,min(1.,col-j))
    return float(grid[i,j]*(1-a)*(1-b)+grid[i+1,j]*a*(1-b)
                 +grid[i,j+1]*(1-a)*b+grid[i+1,j+1]*a*b)

def _p837_r001(station):
    supplied=station.get("rain_rate_001_mmh")
    if supplied not in (None,""):return max(.1,float(supplied))
    lat=float(station.get("lat",0));lon=float(station.get("lon",0))
    return max(0.,_bilinear(_itu_grid("r001"),(lat+90.)/.125,((lon+180.)%360.)/.125))

@lru_cache(maxsize=1)
def _sanya_r001():
    return _p837_r001({"lat":18.2528,"lon":109.5119})

def _rain_height_km(station):
    supplied=station.get("rain_height_km")
    if supplied not in (None,""):return max(0.,float(supplied))
    lat=float(station.get("lat",0));lon=float(station.get("lon",0))
    return _bilinear(_itu_grid("h0"),(90.-lat)/1.5,(lon%360.)/1.5)+.36

def _rain_rate(station,time_s):
    """Return the current rain rate without inventing a temporal trend.

    P.837 R0.01 is a climate statistic, not a weather time series.  When a
    scenario has no measured rain series, use a stable representative rate.
    A caller may supply ``rain_rate_mmh`` (or
    ``representative_rain_rate_mmh``) in the station data.  Time variation is
    introduced only by an explicit external series in a future data adapter.
    """
    supplied=station.get("rain_rate_mmh",station.get("representative_rain_rate_mmh"))
    if supplied not in (None,""):return max(0.,float(supplied))
    return .08*_p837_r001(station)


@lru_cache(maxsize=8192)
def _synthetic_storm_cell(y,x):
    """Static parameters of one reproducible, globally shared rain cell."""
    fraction=lambda value:value-math.floor(value)
    seed=fraction(math.sin(y*127.1+x*311.7)*43758.5453)
    if seed<.42:return None
    return (3*(y+.2+.6*fraction(seed*37.719)),
            3*(x+.2+.6*fraction(seed*91.173)),
            7.+5*seed,10.-6*seed,70.+50.*fraction(seed*53.37))

@lru_cache(maxsize=128)
def _synthetic_site(lat,lon,alt_m,r001_override,height_override):
    station={"lat":lat,"lon":lon,"alt":alt_m}
    if r001_override is not None:station["rain_rate_001_mmh"]=r001_override
    if height_override is not None:station["rain_height_km"]=height_override
    return _rain_height_km(station),min(2.5,max(.45,_p837_r001(station)/45.))

@lru_cache(maxsize=8192)
def _synthetic_storm_offset(epoch,time_s):
    """Continuous rain-field phase shared by every station and path segment."""
    elapsed=(datetime.fromisoformat(epoch).timestamp() if epoch else 0.)+float(time_s)
    hours=elapsed/3600.
    daily=math.sin(2*math.pi*hours/24.)
    weekly=math.sin(2*math.pi*hours/(24.*7.))
    return 6*daily+2*weekly

def _synthetic_storm_displacement(cell,station,time_s):
    """Bounded storm movement keyed to absolute simulation time."""
    offset=_synthetic_storm_offset(station.get("simulation_start_utc"),float(time_s))
    return cell[2]*offset,cell[3]*offset

@lru_cache(maxsize=8192)
def _random_storm_peak(epoch,lat,lon):
    key=f"{epoch}|{lat:.6f}|{lon:.6f}".encode("utf-8")
    return (3.,14.,45.)[hashlib.sha256(key).digest()[0]%3]

def _synthetic_storm_peak(rain_grade,cell,station):
    """One stable intensity per rain cell for the random scenario."""
    rain_grade=station.get("rain_grade") or rain_grade
    fixed={"轻雨":3.,"中雨":14.,"重雨":45.}
    if rain_grade!="随机":return fixed.get(str(rain_grade),14.)
    return _random_storm_peak(str(station.get("simulation_start_utc","")),cell[0],cell[1])

@lru_cache(maxsize=65536)
def _synthetic_daily_event(lat_mdeg,lon_mdeg,day):
    """A reproducible synthetic event, keyed by station and UTC date."""
    digest=hashlib.sha256(f"{lat_mdeg}|{lon_mdeg}|{day}".encode("ascii")).digest()
    values=[int.from_bytes(digest[i:i+4],"big")/4294967296. for i in range(0,20,4)]
    station={"lat":lat_mdeg/1000.,"lon":lon_mdeg/1000.}
    climate=min(1.5,max(.2,_p837_r001(station)/max(.1,_sanya_r001())))
    day_of_year=datetime.fromtimestamp(day*86400,timezone.utc).timetuple().tm_yday
    # Seasonal weighting and wet-day chance are uncalibrated scenario inputs.
    season=.65+.35*math.cos(2*math.pi*(day_of_year-196)/365.25)
    wet_day_threshold=min(.7,.42*climate*season)
    centre=day*86400+(2.+20.*values[1])*3600.
    width=(1.+3.*values[2])*3600.
    return values[0],wet_day_threshold,centre,width,.35+.65*values[3]

def _synthetic_event_strength(instant,lat_mdeg,lon_mdeg,grade):
    day=math.floor(instant/86400.)
    strength=0.
    # A grade changes event occurrence as well as intensity. Shared daily
    # hashes keep the three scenarios correlated and reproducible.
    occurrence_scale={"轻雨":.7,"中雨":1.,"重雨":1.6}.get(grade,1.)
    for event_day in (day-1,day,day+1):
        draw,threshold,centre,width,peak=_synthetic_daily_event(lat_mdeg,lon_mdeg,event_day)
        if draw>=min(.85,threshold*occurrence_scale):continue
        strength=max(strength,peak*math.exp(-.5*((instant-centre)/width)**2))
    return strength

@lru_cache(maxsize=16384)
def _synthetic_time_factor_at(epoch,lat_mdeg,lon_mdeg,time_s,grade):
    start=datetime.fromisoformat(epoch)
    if start.tzinfo is None:start=start.replace(tzinfo=timezone.utc)
    return _synthetic_event_strength(start.timestamp()+float(time_s),lat_mdeg,lon_mdeg,grade)

def _synthetic_time_factor(station,time_s,grade):
    epoch=str(station.get("simulation_start_utc") or "2000-01-01T00:00:00+00:00")
    lat_mdeg=round(float(station.get("lat",0))*1000)
    lon_mdeg=round(float(station.get("lon",0))*1000)
    return _synthetic_time_factor_at(epoch,lat_mdeg,lon_mdeg,round(float(time_s),3),str(grade))

def _uncertainty_rain_floor(station,grade,time_s):
    """Small, nonzero missed-rain allowance for an uncertainty scenario."""
    base={"轻雨":.12,"中雨":.4,"重雨":1.2}.get(str(grade),0.)
    if not base:return 0.
    climate=min(1.3,max(.5,_p837_r001(station)/max(.1,_sanya_r001())))
    phase=math.radians(float(station.get("lat",0))*7.+float(station.get("lon",0))*3.)
    epoch=str(station.get("simulation_start_utc") or "2000-01-01T00:00:00+00:00")
    start=datetime.fromisoformat(epoch)
    if start.tzinfo is None:start=start.replace(tzinfo=timezone.utc)
    instant=start.timestamp()+float(time_s or 0.)
    return base*climate*(.8+.2*math.cos(2.*math.pi*instant/86400.+phase))

def _configured_station_rain_rate(station,distance_km=0.,rain_grade=None,time_s=None):
    """Perturb a date-driven synthetic rain field's occurrence and intensity.

    P.837 R0.01 sets relative site intensity, not a daily forecast. The
    baseline calibration, perturbation and 3 km cell scale are scenario inputs.
    """
    grade=station.get("rain_grade") or rain_grade
    # Calibrate the heavy Sanya scenario to a 45 mm/h upper bound. Actual
    # half-hour samples vary with the generated event and may stay below it.
    baseline_peak=(45./(1.2*max(.1,_sanya_r001())))*_p837_r001(station)
    grade_scale={"轻雨":.8,"中雨":1.,"重雨":1.2}.get(str(grade),1.)
    radius=max(.1,float(station.get("rain_cell_radius_km",3.)))
    time_factor=1. if time_s is None else _synthetic_time_factor(station,time_s,grade)
    event_rate=baseline_peak*time_factor*grade_scale
    scenario_floor=_uncertainty_rain_floor(station,grade,time_s)
    return max(event_rate,scenario_floor)*math.exp(-.5*(float(distance_km)/radius)**2)

def synthetic_surface_rain_rate_mmh(station,rain_grade,time_s):
    """Rain rate at a gateway for visualising the simulated moving field."""
    if (station.get("rain_grade") or rain_grade)!="随机":
        return round(_configured_station_rain_rate(station,rain_grade=rain_grade,time_s=time_s),3)
    lat=float(station.get("lat",0));lon=float(station.get("lon",0))
    _,climate=_synthetic_site(lat,lon,float(station.get("alt",0)),
        station.get("rain_rate_001_mmh"),station.get("rain_height_km"))
    tile_y=math.floor(lat/3.);tile_x=math.floor(lon/3.)
    rate=0.
    for y in range(tile_y-1,tile_y+2):
        for x in range(tile_x-1,tile_x+2):
            cell=_synthetic_storm_cell(y,x)
            if cell is None:continue
            base_lat,base_lon,north_kmh,east_kmh,width=cell
            north_km,east_km=_synthetic_storm_displacement(cell,station,time_s)
            center_lat=base_lat+north_km/111.2
            center_lon=base_lon+east_km/max(15.,111.2*math.cos(math.radians(lat)))
            dy=(lat-center_lat)*111.2
            dx=(lon-center_lon)*max(15.,111.2*math.cos(math.radians(lat)))
            intensity=math.exp(-.5*(dx*dx+dy*dy)/(width*width))
            rate=max(rate,_synthetic_storm_peak(rain_grade,cell,station)*climate*intensity)
    return round(rate,3)

def forecast_rain_rate_mmh(station,time_s):
    """Use the observation/forecast interval containing simulation time t.

    A forecast scenario may apply a uniform relative uncertainty factor to
    every predicted interval. Historical IMERG inputs leave the factor at 1.
    """
    series=station.get("forecast_rain") or {}
    ends=series.get("hour_ends_s") or []
    rates=series.get("rates_mmh") or []
    if not ends or len(ends)!=len(rates):
        raise ValueError(f"信关站 {station.get('name','')} 缺少覆盖当前时段的雨强序列")
    if float(time_s)>float(ends[-1]):
        raise ValueError(f"信关站 {station.get('name','')} 的雨强数据没有覆盖 t={time_s:g}s")
    index=min(len(rates)-1,bisect.bisect_right(ends,float(time_s)))
    scale=max(0.,float(station.get("forecast_rain_scale",1.)))
    predicted=max(0.,float(rates[index]))*scale
    if "forecast_rain_scale" not in station:return predicted
    grade="轻雨" if scale<.9 else "重雨" if scale>1.1 else "中雨"
    return max(predicted,_uncertainty_rain_floor(station,grade,time_s))

def forecast_rain_attenuation_db(station,elevation_deg,time_s):
    """Estimate path fade from a station rain rate and link elevation.

    A point or grid-cell rain rate does not resolve rain along the full slant
    path. Apply the P.618 effective-path reduction as an approximation rather
    than treating the whole path as uniformly rainy.
    """
    rate=forecast_rain_rate_mmh(station,time_s)
    if rate<=0:return 0.
    return attenuation_db(station,elevation_deg,time_s,percentage=.01,rain_rate_mmh=rate)

def synthetic_rain_attenuation_db(station,elevation_deg,rain_grade,time_s,satellite=None):
    """Reproducible scenario rain, integrated along the actual slant path.

    This is a scenario rain field, not a P.618 annual exceedance prediction.
    Explicit grades use a local station cell; random grades use moving cells.
    P.838 supplies the specific attenuation in each path segment.
    """
    elevation=max(.1,float(elevation_deg));lat=float(station.get("lat",0));lon=float(station.get("lon",0))
    altitude=float(station.get("alt",0))/1000.
    height,climate=_synthetic_site(lat,lon,float(station.get("alt",0)),
        station.get("rain_rate_001_mmh"),station.get("rain_height_km"))
    if height<=altitude:return 0.
    slant=(height-altitude)/max(math.sin(math.radians(elevation)),1e-6)
    horizontal=slant*math.cos(math.radians(elevation))
    if satellite is None:azimuth=0.
    else:
        sat_lat=math.radians(float(satellite[2] if isinstance(satellite,(list,tuple)) else satellite.get("lat",lat)))
        delta_lon=math.radians(float(satellite[3] if isinstance(satellite,(list,tuple)) else satellite.get("lon",lon))-lon)
        ground_lat=math.radians(lat)
        azimuth=math.atan2(math.sin(delta_lon)*math.cos(sat_lat),math.cos(ground_lat)*math.sin(sat_lat)-math.sin(ground_lat)*math.cos(sat_lat)*math.cos(delta_lon))
    frequency=float(station.get("frequency_ghz",39.));tilt=float(station.get("polarization_tilt_deg",45.))
    k,alpha=_p838(round(frequency*1000),round(tilt*10),round(elevation*10))
    # Storm centres are generated on a global 3-degree grid. Nearby stations
    # therefore see the same moving cell, while dry stations stay dry.
    # Storm cells are at least 70 km wide; two midpoint samples resolve a
    # typical 15–25 km rain path, with four for longer low-elevation paths.
    attenuation=0.;segments=2 if horizontal<=25. else 4
    for index in range(segments):
        distance=horizontal*(index+.5)/segments
        if (station.get("rain_grade") or rain_grade)!="随机":
            rate=_configured_station_rain_rate(station,distance,rain_grade,time_s)
            attenuation+=k*rate**alpha*slant/segments
            continue
        north=distance*math.cos(azimuth);east=distance*math.sin(azimuth)
        local_lat=lat+north/111.2;local_lon=lon+east/max(15.,111.2*math.cos(math.radians(lat)))
        rate=0.;tile_y=math.floor(local_lat/3.);tile_x=math.floor(local_lon/3.)
        for y in range(tile_y-1,tile_y+2):
            for x in range(tile_x-1,tile_x+2):
                cell=_synthetic_storm_cell(y,x)
                if cell is None:continue
                base_lat,base_lon,north_kmh,east_kmh,width=cell
                north_km,east_km=_synthetic_storm_displacement(cell,station,time_s)
                center_lat=base_lat+north_km/111.2
                center_lon=base_lon+east_km/max(15.,111.2*math.cos(math.radians(local_lat)))
                dy=(local_lat-center_lat)*111.2
                dx=(local_lon-center_lon)*max(15.,111.2*math.cos(math.radians(local_lat)))
                intensity=math.exp(-.5*(dx*dx+dy*dy)/(width*width))
                rate=max(rate,_synthetic_storm_peak(rain_grade,cell,station)*climate*intensity)
        attenuation+=k*rate**alpha*slant/segments
    return round(max(0.,attenuation),3)

@lru_cache(maxsize=8192)
def _static_path(lat_mdeg,alt_m,elev_tenths,f_mghz,tau_tenths,rain_height_m):
    lat=lat_mdeg/1000.;alt=max(0.,alt_m/1000.);elevation=max(.1,elev_tenths/10.);theta=math.radians(elevation)
    rain_height=rain_height_m/1000.;dh=max(0.,rain_height-alt)
    if elevation>=5:slant=dh/max(math.sin(theta),1e-9)
    else:
        re=8500.;slant=2*dh/(math.sqrt(math.sin(theta)**2+2*dh/re)+math.sin(theta)) if dh else 0.
    horizontal=max(0.,slant*math.cos(theta));k,alpha=_p838(f_mghz,tau_tenths,elev_tenths)
    return slant,horizontal,k,alpha,rain_height

def attenuation_db(station,elevation_deg,time_s,percentage=.01,rain_rate_mmh=None):
    """P.618 attenuation exceeded for ``percentage`` of an average year."""
    elevation=max(.1,float(elevation_deg));frequency=float(station.get("frequency_ghz",39.0));tau=float(station.get("polarization_tilt_deg",45.0))
    alt_m=int(round(float(station.get("alt",0))));lat=float(station.get("lat",0));rate=_rain_rate(station,time_s) if rain_rate_mmh is None else max(0.,float(rain_rate_mmh))
    slant,horizontal,k,alpha,rain_height=_static_path(round(lat*1000),alt_m,round(elevation*10),round(frequency*1000),round(tau*10),round(_rain_height_km(station)*1000))
    if slant<=0:return 0.
    gamma=k*rate**alpha;f=max(1.,frequency)
    rh=1/(1+.78*math.sqrt(max(horizontal*gamma/f,0.))-.38*(1-math.exp(-2*horizontal)))
    hr=horizontal*rh;zeta=math.degrees(math.atan2(max(0.,rain_height-alt_m/1000),max(hr,1e-9)))
    lr=hr/max(math.cos(math.radians(elevation)),1e-9) if zeta>elevation else max(0.,(rain_height-alt_m/1000)/max(math.sin(math.radians(elevation)),1e-9))
    chi=max(0.,36-abs(lat));vertical_denominator=1+math.sqrt(max(math.sin(math.radians(elevation)),0))*(31*(1-math.exp(-elevation/(1+chi)))*math.sqrt(max(lr*gamma,0))/f**2-.45)
    v=1/max(.05,vertical_denominator)
    a001=max(0.,gamma*lr*v);p=max(.001,min(5.,float(percentage)))
    if abs(p-.01)<1e-12:return round(a001,3)
    beta=0. if p>=1 or abs(lat)>=36 else (-.005*(abs(lat)-36) if elevation>=25 else -.005*(abs(lat)-36)+1.8-4.25*math.sin(math.radians(elevation)))
    exponent=-(.655+.033*math.log(p)-.045*math.log(max(a001,1e-9))-beta*(1-p)*math.sin(math.radians(elevation)))
    return round(max(0.,a001*(p/.01)**exponent),3)

def link_rain_state(station,elevation_deg,time_s,enabled,sense_sigma,forecast_hours,use_p90):
    if not enabled:return {"rain_rate_mmh":0.,"actual_attenuation_db":0.,"sensed_attenuation_db":0.,"planning_attenuation_db":0.,"severity":"clear","model":"disabled"}
    actual=attenuation_db(station,elevation_deg,time_s);phase=math.radians((float(station.get("lat",0))*11+float(station.get("lon",0))*7)%360)
    sensed=max(0.,actual*(1+sense_sigma*math.sin(time_s/37+phase)))
    # Without an external weather time series, advancing the clock by the
    # forecast lead must not create a different synthetic storm.  Keep the
    # point forecast stationary and express lead-time uncertainty separately.
    forecast=actual
    # A dry point forecast must not imply zero planning uncertainty.  Use a
    # small climate-derived floor based on P.837 R0.01 and the P.618 1-percent
    # attenuation, while keeping the real-time sensed attenuation unchanged.
    climate_a1=attenuation_db(station,elevation_deg,time_s,percentage=1.,rain_rate_mmh=_p837_r001(station))
    uncertainty=max(forecast*min(.65,.10+.08*max(0.,forecast_hours)),.10*climate_a1)
    planning=forecast+(1.2816*uncertainty if use_p90 else 0.);severity="heavy" if actual>=8 else ("moderate" if actual>=3 else ("light" if actual>=.5 else "clear"))
    return {"rain_rate_mmh":round(_rain_rate(station,time_s),3),"rain_rate_001_mmh":round(_p837_r001(station),3),"actual_attenuation_db":round(actual,3),
            "sensed_attenuation_db":round(sensed,3),"planning_attenuation_db":round(planning,3),"severity":severity,"model":"ITU-R P.618/P.837/P.838",
            "rain_rate_source":"station_override" if station.get("rain_rate_001_mmh") not in (None,"") else "ITU-R P.837 map"}

