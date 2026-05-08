import math
import random

from citymind.ambulance_sa import AmbulancePlacementSA
from citymind.city_graph import CityGraph, LocationType, RiskLevel
from citymind.layout_csp import LayoutPlannerCSP
from citymind.road_ga import RoadNetworkGA


def _build_seeded_graph(rows: int, cols: int) -> CityGraph:
    graph = CityGraph(rows, cols)
    layout = LayoutPlannerCSP(graph, seed=7).solve()
    assert layout.success is True
    ga = RoadNetworkGA(graph, population_size=30, generations=50, seed=11)
    roads = ga.optimize()
    ga.apply_selected_roads(roads.selected_edges)
    return graph


def test_state_shape_and_bounds() -> None:
    graph = _build_seeded_graph(6, 6)
    sa = AmbulancePlacementSA(graph, iterations=200, seed=23)
    result = sa.optimize()
    assert len(result.positions) == 3
    assert len(set(result.positions)) == 3
    for r, c in result.positions:
        assert 0 <= r < graph.rows
        assert 0 <= c < graph.cols


def test_objective_better_than_random_baseline() -> None:
    graph = _build_seeded_graph(6, 6)
    sa = AmbulancePlacementSA(graph, iterations=300, seed=23)
    result = sa.optimize()

    rng = random.Random(99)
    baseline_positions = tuple(rng.sample(sorted(graph.nodes.keys()), 3))
    baseline = sa._objective(baseline_positions)
    assert result.worst_case_distance <= baseline


def test_fixed_seed_is_deterministic() -> None:
    graph1 = _build_seeded_graph(6, 6)
    graph2 = _build_seeded_graph(6, 6)
    sa1 = AmbulancePlacementSA(graph1, iterations=280, seed=31)
    sa2 = AmbulancePlacementSA(graph2, iterations=280, seed=31)
    r1 = sa1.optimize()
    r2 = sa2.optimize()
    assert r1.positions == r2.positions
    assert round(r1.worst_case_distance, 6) == round(r2.worst_case_distance, 6)


def test_dynamic_cost_awareness() -> None:
    graph = _build_seeded_graph(5, 5)
    sa = AmbulancePlacementSA(graph, iterations=100, seed=42)
    fixed_state = tuple(sorted(graph.nodes.keys())[:3])
    before = sa._objective(fixed_state)

    center = (2, 2)
    graph.set_risk(center, RiskLevel.HIGH)
    for n in graph.neighbors(center, include_blocked=True):
        graph.set_edge_blocked(center, n, True)

    after = sa._objective(fixed_state)
    assert after >= before


def test_integration_smoke_has_finite_objective() -> None:
    graph = _build_seeded_graph(10, 10)
    sa = AmbulancePlacementSA(graph, iterations=400, seed=23)
    result = sa.optimize()
    assert math.isfinite(result.worst_case_distance)
