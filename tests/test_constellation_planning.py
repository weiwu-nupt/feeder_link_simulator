import sys
import ast
import marshal
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from constellation_sim.algorithm.planning import (
    PlannerConfig, SlowFastLoopPlanner, _component_delivery_bound, _repair_routes_local,
)


def link(gateway, satellite, antenna, rate=1.0, switch=0.02):
    return {'gateway': gateway, 'satellite': satellite, 'antenna_id': antenna,
            'rate_gbps': rate, 'estimated_switch_s': switch,
            'gsl_distance_km': 600.0}


class HandoverTests(unittest.TestCase):
    def test_idle_east_feeder_can_be_replaced_by_visible_west_successor(self):
        cfg=PlannerConfig(selection_strategy='slot_tracking',rain_enabled=False,
            geo_interference_enabled=False,antenna_switch_window_s=10.)
        planner=SlowFastLoopPlanner([{'lat':0.,'lon':0.,'antenna_count':1,'channel_count':1}],cfg)
        old={'gateway':0,'satellite':0,'rate_gbps':1.,'azimuth_deg':90.,
             'pointing_elevation_deg':50.,'gsl_distance_km':600.}
        new={'gateway':0,'satellite':1,'rate_gbps':1.,'azimuth_deg':270.,
             'pointing_elevation_deg':50.,'gsl_distance_km':600.,'next_slot_visible':True}
        previous={'valid_from_s':0.,'valid_until_s':60.,'links':[old],'routes':[]}
        chosen=planner._retain_slot_links([new],[old,new],previous,
            [[(1,1000.)],[(0,1000.)]],{0:.5},.5,60.,{(0,0),(0,1)})
        self.assertEqual([row['satellite'] for row in chosen],[1])

    def test_exhausted_feeder_stops_futile_route_searches(self):
        import heapq
        graph=[[] for _ in range(100)]
        for node in range(99):
            graph[node].append((node+1,1.))
            graph[node+1].append((node,1.))
        pops=[]
        original_pop=heapq.heappop
        def counted_pop(queue):
            pops.append(1)
            return original_pop(queue)
        with patch('constellation_sim.algorithm.planning.heapq.heappop',side_effect=counted_pop):
            routes=_repair_routes_local(graph,{node:1. for node in range(100)},
                {0:1.},max_visited_per_source=500,max_exits_per_source=1)
        self.assertAlmostEqual(sum(row['demand_gbps'] for row in routes),1.)
        self.assertLess(len(pops),150)

    def test_result_history_keeps_metrics_without_paths(self):
        source=(ROOT/'constellation_sim'/'ui'/'main_window.py').read_text(encoding='utf-8')
        parsed=ast.parse(source)
        names={'_result_history_snapshot','_station_metrics'}
        functions=[node for node in parsed.body if isinstance(node,ast.FunctionDef) and node.name in names]
        scope={}
        exec(compile(ast.Module(body=functions,type_ignores=[]),'<history>', 'exec'),scope)
        routes=[{'source':i,'feeder':1,'path':[i]+list(range(20)),'demand_gbps':0.001,
                 'hops':1,'distance_km':100.,'propagation_delay_ms':2.0,
                 'unused_payload':str(i)*100} for i in range(1000)]
        data={'index':5,'time_s':5.,'planning':{'active_plan':{
            'links':[link(0,1,0,3.)],'routes':routes},
            'active_evaluation':{'active_links':[dict(link(0,1,0,3.),scheduled_gbps=1.)],
                                 'routes':routes,'route_available_ratio':1.},
            'decision_context':{'traffic_demand_gbps':1.,'constraint_source':'test'},
            'delivered_gbps':1.}}
        snapshot=scope['_result_history_snapshot'](data)
        before=scope['_station_metrics'](data,None)
        after=scope['_station_metrics'](snapshot,None)
        self.assertAlmostEqual(before['demand'],after['demand'])
        self.assertAlmostEqual(before['scheduled'],after['scheduled'])
        self.assertAlmostEqual(before['average_delay_ms'],after['average_delay_ms'])
        self.assertNotIn('path',snapshot['planning']['active_plan']['routes'][0])
        self.assertLess(len(marshal.dumps(snapshot)),len(marshal.dumps(data))/3)

    def test_component_shortfall_replaces_surplus_feeder(self):
        cfg=PlannerConfig(selection_strategy='slot_tracking',antenna_steering_type='electronic',
            rain_enabled=False,geo_interference_enabled=False,traffic_landing_ratio=1.,
            capacity_reserve_ratio=0.,electronic_scan_from_zenith_deg=90.)
        planner=SlowFastLoopPlanner([{'lat':0.,'lon':0.,'alt':0.,'antenna_count':2,'channel_count':2}],cfg)
        positions=[[0,0,0.,0.,508000.],[1,0,0.,1.,508000.],
                   [2,1,1.,0.,508000.],[3,1,1.,1.,508000.]]
        previous={'links':[
            {'gateway':0,'satellite':0,'antenna_id':0,'rate_gbps':1.5,
             'azimuth_deg':0.,'pointing_elevation_deg':70.,'los_unit':[0.,0.,1.]},
            {'gateway':0,'satellite':1,'antenna_id':1,'rate_gbps':1.5,
             'azimuth_deg':0.,'pointing_elevation_deg':70.,'los_unit':[0.,0.,1.]}],
            'routes':[{'source':0,'feeder':0,'path':[0],'demand_gbps':0.2}]}
        plan=planner.generate(60.,positions,[(0,1,0),(2,3,0)],
            [(0,0,70.),(0,1,70.),(0,2,45.),(0,3,45.)],
            [(0,.2),(2,1.)],commit=False,previous_plan=previous)
        self.assertGreaterEqual(plan['capacity_component_rescue_count'],1)
        self.assertTrue(any(row['satellite'] in (2,3) for row in plan['links']))
        self.assertAlmostEqual(plan['reachable_demand_gbps'],1.2)

    def test_disconnected_capacity_is_not_counted_as_reachable(self):
        bound = _component_delivery_bound(4, {(0, 1), (2, 3)},
            {0: 2.0, 2: 2.0}, [link(0, 1, 0, 3.0), link(0, 3, 1, 1.0)])
        self.assertAlmostEqual(bound, 3.0)

    def test_electronic_switch_starts_together_and_routes_new_target_in_first_phase(self):
        cfg = PlannerConfig(antenna_steering_type='electronic',
            electronic_switch_ms=20.0, selection_strategy='slot_tracking',
            realtime_exit_limit_per_source=2)
        planner = SlowFastLoopPlanner([{'lat': 0, 'lon': 0}], cfg)
        previous = {'links': [link(0, 1, 0), link(0, 3, 1)],
                    'routes': [{'source': 0, 'feeder': 1, 'path': [0, 1], 'demand_gbps': 0.5},
                               {'source': 2, 'feeder': 3, 'path': [2, 3], 'demand_gbps': 0.5}]}
        plan = {'valid_from_s': 60.0, 'valid_until_s': 120.0,
                'links': [link(0, 2, 0), link(0, 0, 1)],
                'demand_by_source_gbps': {'0': 0.5, '2': 0.5},
                'priority_demands_by_satellite_gbps': {}}
        starts = planner._slot_switch_starts(plan, previous)
        self.assertEqual(starts, {(0, 0): 60.0, (0, 1): 60.0})
        positions = [[i, 0, 0., float(i), 508000.] for i in range(4)]
        edges = [(0, 1, 0), (1, 2, 0), (2, 3, 0)]
        planner.prepare_transition_routes(plan, previous, positions, edges)
        self.assertIn(2, {row['feeder'] for row in plan['transition_routes'][0]['routes']})
        self.assertIn(0, {row['feeder'] for row in plan['transition_routes'][0]['routes']})
        plan.update(id='electronic', name='electronic', routes=plan['transition_routes'][0]['routes'])
        planner.state.plans=[previous]
        planner.install_offline_plan(plan)
        actual_edges={(0,1),(1,2),(2,3)}
        pairs={(0,satellite):60.0 for satellite in range(4)}
        before=planner._evaluate(60.0,planner.state.execution_plan,pairs,1.0,actual_edges,
            positions=positions,demand_by_source={0:0.5,2:0.5})
        after=planner._evaluate(60.02,planner.state.execution_plan,pairs,1.0,actual_edges,
            positions=positions,demand_by_source={0:0.5,2:0.5})
        self.assertEqual(before['active_link_count'],0)
        self.assertEqual(after['active_link_count'],2)
        self.assertAlmostEqual(after['scheduled_gbps'],1.0)

    def test_mechanical_waiting_phase_keeps_old_feeder(self):
        cfg = PlannerConfig(antenna_steering_type='mechanical',
            selection_strategy='slot_tracking', realtime_exit_limit_per_source=2)
        planner = SlowFastLoopPlanner([{'lat': 0, 'lon': 0}], cfg)
        previous = {'links': [link(0, 1, 0, switch=8), link(0, 3, 1, switch=8)],
                    'routes': [{'source': 0, 'feeder': 1, 'path': [0, 1], 'demand_gbps': 0.5},
                               {'source': 2, 'feeder': 3, 'path': [2, 3], 'demand_gbps': 0.5}]}
        plan = {'valid_from_s': 60.0, 'valid_until_s': 120.0,
                'links': [link(0, 2, 0, switch=8), link(0, 0, 1, switch=8)],
                'demand_by_source_gbps': {'0': 0.5, '2': 0.5},
                'priority_demands_by_satellite_gbps': {}}
        positions = [[i, 0, 0., float(i), 508000.] for i in range(4)]
        edges = [(0, 1, 0), (1, 2, 0), (2, 3, 0)]
        planner.prepare_transition_routes(plan, previous, positions, edges)
        waiting = next(x for x in plan['transition_routes'] if x['start_s'] == 65.0)
        self.assertIn(3, {row['feeder'] for row in waiting['routes']})


if __name__ == '__main__':
    unittest.main()
