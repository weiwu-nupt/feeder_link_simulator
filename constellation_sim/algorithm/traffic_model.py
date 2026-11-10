"""Ground traffic generation and access-satellite association.

This module implements section 5.4 demand generation and access association:

    ground demand T(g, t) [Gb/s] -> visible access satellites -> satellite ingress load [Gb/s]

The caller may supply each satellite's shortest current ISL distance to a feeder
satellite. It is used as a path-ranking term without modeling packet transport.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class TrafficCell:
    """One fixed ground service cell."""

    cell_id: str
    latitude_deg: float
    longitude_deg: float
    base_weight: float
    population: float = 0.0
    enabled: bool = True


@dataclass(frozen=True)
class VisibilityCandidate:
    """A satellite which can serve a cell through the user link."""

    satellite_id: int
    elevation_deg: float


@dataclass
class TrafficModelConfig:
    # T(g,t) = total_reference_gbps * normalized_weight(g) * C(local_time).
    normalize_population_weights: bool = True
    population_exponent: float = 1.0
    total_reference_gbps: float = 1000.0

    # Local-time double Gaussian. Values follow section 5.4 baselines.
    base_level: float = 0.15
    office_peak_hour: float = 10.5
    office_peak_sigma_h: float = 2.0
    office_peak_amplitude: float = 0.55
    evening_peak_hour: float = 20.5
    evening_peak_sigma_h: float = 2.5
    evening_peak_amplitude: float = 0.85

    # Optional reproducible random-demand component.
    random_sigma: float = 0.0
    random_seed: int = 20261023

    # Dense-cell and access-association policy.
    dense_percentile: float = 95.0
    dense_absolute_weight: float = 0.0
    normal_max_satellites: int = 2
    dense_max_satellites: int = 4
    dense_cluster_radius_km: float = 180.0
    dense_cluster_min_cells: int = 2
    min_elevation_deg: float = 20.0
    elevation_score_weight: float = 0.55
    load_balance_score_weight: float = 0.45


@dataclass
class CellAllocation:
    cell_id: str
    demand_weight: float
    local_hour: float
    dense: bool
    visible_satellite_count: int
    allocations: dict[int, float] = field(default_factory=dict)
    unassigned_weight: float = 0.0
    dense_cluster_id: int = -1


@dataclass
class SatelliteTrafficLoad:
    satellite_id: int
    assigned_weight: float = 0.0
    served_cell_count: int = 0


@dataclass
class TrafficFrameResult:
    utc_time: datetime
    cells: list[CellAllocation]
    satellites: dict[int, SatelliteTrafficLoad]
    total_demand_weight: float
    assigned_weight: float
    unassigned_weight: float


def _cyclic_hour_distance(hour: float, centre: float) -> float:
    """Shortest signed separation on a 24-hour clock."""
    return (hour - centre + 12.0) % 24.0 - 12.0


def local_hour(utc_time: datetime, longitude_deg: float) -> float:
    """Document-defined solar local time: UTC + longitude / 15."""
    if utc_time.tzinfo is None:
        utc_time = utc_time.replace(tzinfo=timezone.utc)
    utc_time = utc_time.astimezone(timezone.utc)
    utc_hour = (
        utc_time.hour
        + utc_time.minute / 60.0
        + utc_time.second / 3600.0
        + utc_time.microsecond / 3.6e9
    )
    return (utc_hour + longitude_deg / 15.0) % 24.0


def double_peak_coefficient(hour: float, cfg: TrafficModelConfig) -> float:
    """24-hour office/life double-peak coefficient C(T)."""
    d1 = _cyclic_hour_distance(hour, cfg.office_peak_hour)
    d2 = _cyclic_hour_distance(hour, cfg.evening_peak_hour)
    return (
        cfg.base_level
        + cfg.office_peak_amplitude
        * math.exp(-(d1 * d1) / (2.0 * cfg.office_peak_sigma_h**2))
        + cfg.evening_peak_amplitude
        * math.exp(-(d2 * d2) / (2.0 * cfg.evening_peak_sigma_h**2))
    )


class TrafficModel:
    """Generate demand and allocate it to access satellites frame by frame."""

    def __init__(self, cells: Sequence[TrafficCell], config: TrafficModelConfig):
        self.cells = [cell for cell in cells if cell.enabled and cell.base_weight > 0.0]
        self.config = config
        self._weights = self._prepare_weights(self.cells)

    def _prepare_weights(self, cells: Sequence[TrafficCell]) -> dict[str, float]:
        cfg = self.config
        weighted = {
            cell.cell_id: max(0.0, cell.base_weight) ** cfg.population_exponent
            for cell in cells
        }
        if not cfg.normalize_population_weights:
            return weighted
        total = sum(weighted.values())
        if total <= 0.0:
            return {cell.cell_id: 0.0 for cell in cells}
        return {cell_id: value / total for cell_id, value in weighted.items()}

    def _random_multiplier(self, cell_id: str, frame_key: str) -> float:
        """Reproducible lognormal multiplier with mean one."""
        sigma = self.config.random_sigma
        if sigma <= 0.0:
            return 1.0
        digest = hashlib.blake2b(
            f"{self.config.random_seed}:{frame_key}:{cell_id}".encode("utf-8"),
            digest_size=8,
        ).digest()
        rng = random.Random(int.from_bytes(digest, "big"))
        return math.exp(sigma * rng.gauss(0.0, 1.0) - sigma * sigma / 2.0)

    def generate_demand(self, utc_time: datetime) -> tuple[dict[str, float], dict[str, float]]:
        """Return demand in Gb/s and local hour for every active cell."""
        frame_key = utc_time.astimezone(timezone.utc).isoformat()
        demand: dict[str, float] = {}
        hours: dict[str, float] = {}
        for cell in self.cells:
            hour = local_hour(utc_time, cell.longitude_deg)
            hours[cell.cell_id] = hour
            demand[cell.cell_id] = (
                self.config.total_reference_gbps
                *
                self._weights[cell.cell_id]
                * double_peak_coefficient(hour, self.config)
                * self._random_multiplier(cell.cell_id, frame_key)
            )
        return demand, hours

    def _dense_threshold(self, values: Iterable[float]) -> float:
        ordered = sorted(v for v in values if v > 0.0)
        if not ordered:
            return math.inf
        p = min(100.0, max(0.0, self.config.dense_percentile)) / 100.0
        index = min(len(ordered) - 1, max(0, math.ceil(p * len(ordered)) - 1))
        percentile_threshold = ordered[index]
        if self.config.dense_absolute_weight > 0.0:
            return max(percentile_threshold, self.config.dense_absolute_weight)
        return percentile_threshold

    @staticmethod
    def _distance_km(a: TrafficCell, b: TrafficCell) -> float:
        lat1, lat2 = math.radians(a.latitude_deg), math.radians(b.latitude_deg)
        dlat = lat2 - lat1
        dlon = math.radians(b.longitude_deg - a.longitude_deg)
        value = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
        return 12_742.0 * math.asin(min(1.0, math.sqrt(value)))

    def _dense_clusters(self, demand: Mapping[str, float]) -> tuple[dict[str, int], list[list[TrafficCell]]]:
        """Group adjacent high-demand cells; isolated hot cells stay normal."""
        threshold = self._dense_threshold(demand.values())
        hot = [cell for cell in self.cells if demand[cell.cell_id] >= threshold]
        neighbours: list[list[int]] = [[] for _ in hot]
        radius = max(0.0, self.config.dense_cluster_radius_km)
        for i, left in enumerate(hot):
            for j in range(i + 1, len(hot)):
                if self._distance_km(left, hot[j]) <= radius:
                    neighbours[i].append(j)
                    neighbours[j].append(i)
        membership: dict[str, int] = {}
        clusters: list[list[TrafficCell]] = []
        visited: set[int] = set()
        for start in range(len(hot)):
            if start in visited:
                continue
            stack, component = [start], []
            visited.add(start)
            while stack:
                index = stack.pop()
                component.append(hot[index])
                for nxt in neighbours[index]:
                    if nxt not in visited:
                        visited.add(nxt)
                        stack.append(nxt)
            if len(component) < self.config.dense_cluster_min_cells:
                continue
            cluster_id = len(clusters)
            clusters.append(component)
            membership.update({cell.cell_id: cluster_id for cell in component})
        return membership, clusters

    @staticmethod
    def _quality(candidate: VisibilityCandidate, min_elevation_deg: float) -> float:
        # Maps the useful elevation interval to [0, 1].
        denominator = max(1e-9, 90.0 - min_elevation_deg)
        return min(1.0, max(0.0, (candidate.elevation_deg - min_elevation_deg) / denominator))

    def run_frame(
        self,
        utc_time: datetime,
        visibility: Mapping[str, Sequence[VisibilityCandidate]],
    ) -> TrafficFrameResult:
        """Generate one demand frame and perform load-aware access allocation.

        Normal cells use one access satellite. Adjacent high-demand cells form
        clusters whose cells are spread across a shared cooperative satellite pool.
        """
        cfg = self.config
        demand, hours = self.generate_demand(utc_time)
        cluster_of, clusters = self._dense_clusters(demand)
        satellite_ids = {
            item.satellite_id
            for candidates in visibility.values()
            for item in candidates
        }
        sat_loads = {sid: SatelliteTrafficLoad(satellite_id=sid) for sid in satellite_ids}
        results_by_cell: dict[str, CellAllocation] = {}
        def make_allocation(cell: TrafficCell, dense: bool, cluster_id: int) -> CellAllocation:
            return CellAllocation(cell.cell_id, demand[cell.cell_id], hours[cell.cell_id], dense,
                                  len(visibility.get(cell.cell_id, ())), dense_cluster_id=cluster_id)

        # Allocate each dense cluster across a shared satellite pool.  A cell
        # represents many terminals, so its aggregate traffic may be shared by
        # a small number of simultaneously visible access satellites.  This
        # prevents one grid cell from creating an impossible egress bottleneck
        # when every business satellite is limited to two feeder exits.
        for cluster_id, cells in sorted(enumerate(clusters), key=lambda item: -sum(demand[c.cell_id] for c in item[1])):
            candidate_stats: dict[int, list[float]] = {}
            for cell in cells:
                for item in visibility.get(cell.cell_id, ()):
                    if item.elevation_deg >= cfg.min_elevation_deg:
                        candidate_stats.setdefault(item.satellite_id, []).append(
                            self._quality(item, cfg.min_elevation_deg))
            ranked_pool = sorted(candidate_stats, key=lambda sid: (
                sum(candidate_stats[sid]) / len(candidate_stats[sid])
            ), reverse=True)[:max(1, cfg.dense_max_satellites)]
            cluster_counts = {sid: 0 for sid in ranked_pool}
            for cell in sorted(cells, key=lambda item: demand[item.cell_id], reverse=True):
                allocation = make_allocation(cell, True, cluster_id)
                candidates = [item for item in visibility.get(cell.cell_id, ())
                              if item.satellite_id in cluster_counts and item.elevation_deg >= cfg.min_elevation_deg]
                if candidates:
                    max_count = max(cluster_counts.values(), default=0)
                    def cluster_score(item: VisibilityCandidate) -> float:
                        diversity = 1.0 if max_count <= 0 else 1.0 - cluster_counts[item.satellite_id] / max_count
                        return (cfg.elevation_score_weight * self._quality(item, cfg.min_elevation_deg)
                                + cfg.load_balance_score_weight * diversity)
                    selected = sorted(candidates,key=cluster_score,reverse=True)[:min(4,len(candidates))]
                    share=allocation.demand_weight/len(selected)
                    allocation.allocations = {item.satellite_id: share for item in selected}
                    for item in selected:cluster_counts[item.satellite_id] += 1
                else:
                    allocation.unassigned_weight = allocation.demand_weight
                for sid, assigned in allocation.allocations.items():
                    sat_loads[sid].assigned_weight += assigned
                    sat_loads[sid].served_cell_count += 1
                results_by_cell[cell.cell_id] = allocation

        running_max_load=max((load.assigned_weight for load in sat_loads.values()),default=0.0)
        for cell in sorted(self.cells, key=lambda item: demand[item.cell_id], reverse=True):
            if cell.cell_id in results_by_cell:
                continue
            cell_demand = demand[cell.cell_id]
            candidates = [
                item for item in visibility.get(cell.cell_id, ())
                if item.elevation_deg >= cfg.min_elevation_deg
            ]
            # The load maximum is constant while candidates for this cell are
            # scored. Computing it inside score() caused an O(cells × candidates
            # × visible-satellites) scan in large constellations.
            max_load = running_max_load

            def score(item: VisibilityCandidate) -> float:
                sid = item.satellite_id
                load_score = 1.0 if max_load <= 0.0 else 1.0 - sat_loads[sid].assigned_weight / max_load
                return (
                    cfg.elevation_score_weight * self._quality(item, cfg.min_elevation_deg)
                    + cfg.load_balance_score_weight * load_score
                )

            allocation = make_allocation(cell, False, -1)
            if candidates:
                selected=sorted(candidates,key=score,reverse=True)[:min(max(1,cfg.normal_max_satellites),len(candidates))]
                share=cell_demand/len(selected)
                allocation.allocations = {item.satellite_id: share for item in selected}
            else:
                allocation.unassigned_weight = cell_demand
            for sid, assigned in allocation.allocations.items():
                sat_loads[sid].assigned_weight += assigned
                sat_loads[sid].served_cell_count += 1
                running_max_load=max(running_max_load,sat_loads[sid].assigned_weight)
            results_by_cell[cell.cell_id] = allocation

        results = [results_by_cell[cell.cell_id] for cell in self.cells]

        total = sum(item.demand_weight for item in results)
        assigned = sum(sum(item.allocations.values()) for item in results)
        return TrafficFrameResult(
            utc_time=utc_time,
            cells=results,
            satellites=sat_loads,
            total_demand_weight=total,
            assigned_weight=assigned,
            unassigned_weight=max(0.0, total - assigned),
        )


def calculate_visibility(
    cells: Sequence[TrafficCell],
    satellite_positions: Sequence[Sequence[float]],
    min_elevation_deg: float,
) -> dict[str, list[VisibilityCandidate]]:
    """Calculate cell-to-satellite visibility without an all-pairs scan.

    ``satellite_positions`` rows are ``[id, plane, lat_deg, lon_deg, alt_m]``.
    A unit-sphere KD-tree first finds satellites inside the largest possible
    footprint; the exact spherical elevation test then removes false matches.
    """
    try:
        import numpy as np
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError("大规模业务网格可见性计算需要 scipy") from exc
    if not cells:
        return {cell.cell_id: [] for cell in cells}
    earth_radius = 6_378_135.0
    rows = np.asarray(satellite_positions)
    if rows.size == 0:
        return {cell.cell_id: [] for cell in cells}
    sat_lat = np.radians(rows[:, 2].astype(float, copy=False))
    sat_lon = np.radians(rows[:, 3].astype(float, copy=False))
    sat_alt = rows[:, 4].astype(float, copy=False)
    sat_unit = np.column_stack((
        np.cos(sat_lat) * np.cos(sat_lon),
        np.cos(sat_lat) * np.sin(sat_lon),
        np.sin(sat_lat),
    ))
    tree = cKDTree(sat_unit)
    elevation = math.radians(min_elevation_deg)
    max_radius = earth_radius + max(0.0, float(sat_alt.max()))
    max_central = math.acos(earth_radius / max_radius * math.cos(elevation)) - elevation
    chord_radius = 2.0 * math.sin(max(0.0, max_central) / 2.0)
    output: dict[str, list[VisibilityCandidate]] = {}
    cell_lat=np.radians(np.asarray([cell.latitude_deg for cell in cells],dtype=float));cell_lon=np.radians(np.asarray([cell.longitude_deg for cell in cells],dtype=float))
    ground_units=np.column_stack((np.cos(cell_lat)*np.cos(cell_lon),np.cos(cell_lat)*np.sin(cell_lon),np.sin(cell_lat)))
    try:index_groups=tree.query_ball_point(ground_units,chord_radius,workers=1)
    except TypeError:index_groups=tree.query_ball_point(ground_units,chord_radius)
    for cell,ground_unit,index_group in zip(cells,ground_units,index_groups):
        indices=np.asarray(index_group,dtype=np.int64)
        if not indices.size:
            output[cell.cell_id]=[];continue
        cosine=np.clip(sat_unit[indices]@ground_unit,-1.0,1.0);central=np.arccos(cosine);radius=earth_radius+sat_alt[indices]
        slant=np.sqrt(np.maximum(1.0,radius*radius+earth_radius*earth_radius-2.0*radius*earth_radius*cosine))
        elevation_deg=np.degrees(np.arcsin(np.clip((radius*cosine-earth_radius)/slant,-1.0,1.0)));valid=elevation_deg>=min_elevation_deg
        output[cell.cell_id]=[VisibilityCandidate(int(rows[int(index),0]),float(value)) for index,value in zip(indices[valid],elevation_deg[valid])]
    return output

