"""ITU-R based rain attenuation for planning and real-time scheduling.

P.618 supplies the slant-path procedure, P.838 the specific attenuation law,
and P.837 the station R0.01 input. Static coefficients and geometry are cached.
"""
from __future__ import annotations
import math
from functools import lru_cache

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

def _p837_r001(station):
    supplied=station.get("rain_rate_001_mmh")
    if supplied not in (None,""):return max(.1,float(supplied))
    lat=float(station.get("lat",0));lon=float(station.get("lon",0))
    tropical=1/(1+math.exp((abs(lat)-27)/4.5));monsoon=1/(1+math.exp((103-lon)/5.5))
    return max(5.,18+62*tropical*(.42+.58*monsoon))

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


# Network planning grades use conditions that are meaningful over a 30-minute
# operating window.  Applying the annual 0.01% extreme simultaneously to every
# gateway represents a nationwide worst case and eliminated every feeder link.
# P.618 still performs the attenuation extrapolation from each site's R0.01.
RAIN_GRADE_PERCENTILES={"轻雨":5.0,"中雨":1.0,"重雨":.1}

# Offline rain grades share one moving, spatially correlated rain field.  The
# percentile controls the peak attenuation, while the profile controls the
# footprint and the non-storm background.  This represents site diversity:
# geographically separated gateways do not all experience the percentile
# extreme at the same instant.  It is a reproducible synthetic field used
# when no gridded weather product has been supplied.
RAIN_GRADE_SPATIAL_PROFILES={
    "轻雨": {"radius_km":360.,"background":.12,"peak":.62},
    "中雨": {"radius_km":480.,"background":.14,"peak":.82},
    "重雨": {"radius_km":620.,"background":.16,"peak":1.00},
}


def _surface_distance_km(lat1,lon1,lat2,lon2):
    p1,p2=math.radians(lat1),math.radians(lat2);dp=p2-p1;dl=math.radians(lon2-lon1)
    a=math.sin(dp/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 6371.0088*2*math.asin(min(1.,math.sqrt(max(0.,a))))


def _scenario_storm_centres(time_s):
    """Three deterministic rain cells moving once per five-minute slot."""
    slot=max(0,int(float(time_s)//300.))
    centres=[]
    for index in range(3):
        phase=.73*slot+2.09439510239*index
        lat=32.+10.5*math.sin(phase+.4*index)
        lon=104.+20.5*math.cos(.81*phase+1.1*index)
        centres.append((lat,lon))
    return centres


def scenario_spatial_factor(station,rain_grade,time_s=0.):
    profile=RAIN_GRADE_SPATIAL_PROFILES.get(str(rain_grade),RAIN_GRADE_SPATIAL_PROFILES["中雨"])
    lat=float(station.get("lat",0));lon=float(station.get("lon",0));radius=max(1.,float(profile["radius_km"]))
    influence=max(math.exp(-.5*(_surface_distance_km(lat,lon,c_lat,c_lon)/radius)**2) for c_lat,c_lon in _scenario_storm_centres(time_s))
    background=max(0.,min(1.,float(profile["background"])));peak=max(background,min(1.,float(profile["peak"])))
    return round(background+(peak-background)*influence,6)


def scenario_attenuation_db(station,elevation_deg,rain_grade,time_s=0.):
    """ITU attenuation sampled through a reproducible correlated rain field."""
    percentage=RAIN_GRADE_PERCENTILES.get(str(rain_grade),1.0)
    peak=attenuation_db(station,elevation_deg,0.,percentage=percentage,rain_rate_mmh=_p837_r001(station))
    return round(peak*scenario_spatial_factor(station,rain_grade,time_s),3)

@lru_cache(maxsize=8192)
def _static_path(lat_mdeg,alt_m,elev_tenths,f_mghz,tau_tenths):
    lat=lat_mdeg/1000.;alt=max(0.,alt_m/1000.);elevation=max(.1,elev_tenths/10.);theta=math.radians(elevation)
    rain_height=max(1.5,5.0-.075*abs(lat))+.36;dh=max(0.,rain_height-alt)
    if elevation>=5:slant=dh/max(math.sin(theta),1e-9)
    else:
        re=8500.;slant=2*dh/(math.sqrt(math.sin(theta)**2+2*dh/re)+math.sin(theta)) if dh else 0.
    horizontal=max(0.,slant*math.cos(theta));k,alpha=_p838(f_mghz,tau_tenths,elev_tenths)
    return slant,horizontal,k,alpha,rain_height

def attenuation_db(station,elevation_deg,time_s,percentage=.01,rain_rate_mmh=None):
    """P.618 attenuation exceeded for ``percentage`` of an average year."""
    elevation=max(.1,float(elevation_deg));frequency=float(station.get("frequency_ghz",39.0));tau=float(station.get("polarization_tilt_deg",45.0))
    alt_m=int(round(float(station.get("alt",0))));lat=float(station.get("lat",0));rate=_rain_rate(station,time_s) if rain_rate_mmh is None else max(0.,float(rain_rate_mmh))
    slant,horizontal,k,alpha,rain_height=_static_path(round(lat*1000),alt_m,round(elevation*10),round(frequency*1000),round(tau*10))
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
            "rain_rate_source":"station_p837" if station.get("rain_rate_001_mmh") not in (None,"") else "geographic_fallback"}

