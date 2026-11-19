"""Import half-hourly IMERG point exports for historical rain replay."""
from __future__ import annotations

import csv
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path


INTERVAL = timedelta(minutes=30)


def _utc(value):
    result = datetime.fromisoformat(str(value).strip().replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('IMERG 时间必须包含 UTC 时区，例如 2024-07-01T00:00:00Z')
    return result.astimezone(timezone.utc)


def load_imerg_csv(paths, stations, start_iso, duration_s):
    """Return a complete station snapshot; reject missing half-hour bins.

    Accept NASA/ERDDAP columns time,latitude,longitude,precipitation (mm/h),
    or station,time_utc,rain_rate_mmh. NASA's second units row is skipped.
    """
    paths = [Path(path) for path in paths]
    if not paths:
        raise ValueError('请先导入覆盖所有信关站和仿真时段的 IMERG CSV 文件。')
    start = _utc(start_iso)
    end = start + timedelta(seconds=float(duration_s))
    if end <= start:
        raise ValueError('仿真时长必须大于 0。')
    first_bin = start.replace(minute=30 if start.minute >= 30 else 0,
                              second=0, microsecond=0)
    required = []
    cursor = first_bin
    while cursor < end:
        required.append(cursor)
        cursor += INTERVAL
    values = [dict() for _ in stations]
    for path in paths:
        with path.open('r', encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            columns = set(reader.fieldnames or ())
            nasa = {'time', 'latitude', 'longitude', 'precipitation'} <= columns
            named = {'station', 'time_utc', 'rain_rate_mmh'} <= columns
            if not nasa and not named:
                raise ValueError(f'{path.name} 缺少 IMERG 列 time,latitude,longitude,precipitation。')
            for row in reader:
                time_text = row['time'] if nasa else row['time_utc']
                if str(time_text).strip().upper() == 'UTC':
                    continue
                stamp = _utc(time_text)
                if stamp not in required:
                    continue
                if nasa:
                    lat = float(row['latitude']); lon = float(row['longitude'])
                    distances = [math.hypot(lat-float(site['lat']), lon-float(site['lon']))
                                 for site in stations]
                    index = min(range(len(stations)), key=distances.__getitem__)
                    if distances[index] > .15:
                        continue
                    raw = row['precipitation']
                else:
                    index = next((i for i, site in enumerate(stations)
                                  if str(site.get('name')) == str(row['station']).strip()), None)
                    if index is None:
                        continue
                    raw = row['rain_rate_mmh']
                rate = float(raw)
                if not math.isfinite(rate) or rate < 0:
                    raise ValueError(f'{path.name} 的 {stations[index]["name"]} {stamp.isoformat()} 雨强缺测或无效。')
                previous = values[index].get(stamp)
                if previous is not None and abs(previous-rate) > 1e-6:
                    raise ValueError(f'{stations[index]["name"]} {stamp.isoformat()} 有冲突的雨强记录。')
                values[index][stamp] = rate
    records = []
    for index, site in enumerate(stations):
        missing = [stamp for stamp in required if stamp not in values[index]]
        if missing:
            raise ValueError(f'{site["name"]} 缺少 {missing[0]:%Y-%m-%d %H:%M} UTC 起的半小时 IMERG 数据。')
        records.append({'name': site.get('name'), 'lat': float(site['lat']),
                        'lon': float(site['lon']),
                        'times': [(stamp+INTERVAL).isoformat() for stamp in required],
                        'rates': [values[index][stamp] for stamp in required]})
    return {'start_iso': start.isoformat(), 'source': 'NASA IMERG V07 half-hour CSV',
            'files': [str(path.resolve()) for path in paths], 'stations': records}
