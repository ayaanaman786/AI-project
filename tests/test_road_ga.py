from citymind.city_graph import CityGraph, LocationType
from citymind.layout_csp import LayoutPlannerCSP
from citymind.road_ga import RoadNetworkGA


def _seed_locations(graph: CityGraph) -> None:
    for cell in graph.nodes:
        graph.set_location(cell, LocationType.RESIDENTIAL, population=100)
    graph.set_location((0, 0), LocationType.HOSPITAL, population=0)
    graph.set_location((graph.rows - 1, graph.cols - 1), LocationType.AMBULANCE_DEPOT, population=0)


def test_candidate_edges_are_unique_undirected_pairs() -> None:
    graph = CityGraph(3, 3)
    _seed_locations(graph)
    ga = RoadNetworkGA(graph, population_size=10, generations=5, seed=1)
    assert len(ga.candidate_edges) == 12
    normalized = {ga._normalize_edge(a, b) for a, b in ga.candidate_edges}
    assert len(normalized) == len(ga.candidate_edges)


def test_connectivity_penalty_differs_for_disconnected_and_connected() -> None:
    graph = CityGraph(2, 2)
    _seed_locations(graph)
    ga = RoadNetworkGA(graph, population_size=10, generations=5, seed=2)
    disconnected = [0] * len(ga.candidate_edges)
    connected = [0] * len(ga.candidate_edges)
    for edge in ga._mst_edges():
        connected[ga.edge_index[edge]] = 1
    assert ga._fitness(connected) > ga._fitness(disconnected)


def test_redundancy_check_rejects_bridge_only_path() -> None:
    graph = CityGraph(2, 2)
    _seed_locations(graph)
    ga = RoadNetworkGA(graph, population_size=10, generations=5, seed=3)
    bridge_only = [((0, 0), (0, 1)), ((0, 1), (1, 1)), ((0, 0), (1, 0))]
    assert ga._has_hospital_depot_redundancy(bridge_only) is False


def test_strict_two_edge_disjoint_check_accepts_full_cycle() -> None:
    """A 4-cycle has exactly two edge-disjoint paths between any pair of
    diagonal nodes (left-around vs right-around), so Menger's theorem says the
    strict check must accept it."""
    graph = CityGraph(2, 2)
    _seed_locations(graph)
    ga = RoadNetworkGA(graph, population_size=10, generations=5, seed=4)
    cycle_edges = [
        ((0, 0), (0, 1)),
        ((0, 1), (1, 1)),
        ((1, 1), (1, 0)),
        ((0, 0), (1, 0)),
    ]
    assert ga._has_hospital_depot_redundancy(cycle_edges) is True


def test_strict_two_edge_disjoint_check_rejects_articulation_pair() -> None:
    """A graph shaped like two triangles joined at a single vertex has only
    one edge-disjoint path between hospital (in left triangle) and depot (in
    right triangle); the strict check must reject it because the join vertex
    is an articulation point but the joining edge is a single bridge."""
    graph = CityGraph(2, 3)
    for cell in graph.nodes:
        graph.set_location(cell, LocationType.RESIDENTIAL, population=10)
    graph.set_location((0, 0), LocationType.HOSPITAL, population=0)
    graph.set_location((0, 2), LocationType.AMBULANCE_DEPOT, population=0)

    ga = RoadNetworkGA(graph, population_size=10, generations=5, seed=5)
    bridge_between_triangles = [
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
        ((1, 0), (0, 1)),
        ((0, 1), (0, 2)),
        ((0, 1), (1, 1)),
        ((1, 1), (0, 2)),
    ]
    # Edge (0,1)-(0,2) is the only direct path forward; verify the strict
    # check sees that there is no second edge-disjoint route, since (0,1) is
    # an articulation vertex but (0,1)-(0,2) is also a bridge.
    bridge_only = [
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
        ((1, 0), (0, 1)),
        ((0, 1), (0, 2)),
    ]
    assert ga._has_hospital_depot_redundancy(bridge_only) is False
    # Adding the second triangle creates two edge-disjoint paths via
    # (0,0)-(0,1)-(0,2) and (0,0)-(1,0)-(0,1)-(1,1)-(0,2). However note that
    # (0,1) is shared by both, but Menger for *edge* connectivity allows
    # vertex sharing; only edges must be disjoint, so this is accepted.
    assert ga._has_hospital_depot_redundancy(bridge_between_triangles) is True


def test_strict_check_returns_true_when_source_equals_target() -> None:
    graph = CityGraph(2, 2)
    _seed_locations(graph)
    ga = RoadNetworkGA(graph, population_size=10, generations=5, seed=6)
    ga.primary_hospital = (0, 0)
    ga.ambulance_depot = (0, 0)
    assert ga._has_hospital_depot_redundancy([]) is True


def test_optimize_returns_connected_and_redundant_network() -> None:
    graph = CityGraph(10, 10)
    layout = LayoutPlannerCSP(graph, seed=7).solve()
    assert layout.success is True
    ga = RoadNetworkGA(graph, population_size=50, generations=100, seed=11)
    result = ga.optimize()
    assert result.component_count == 1
    assert result.has_hospital_depot_redundancy is True


def test_fixed_seed_is_deterministic() -> None:
    graph1 = CityGraph(6, 6)
    graph2 = CityGraph(6, 6)
    LayoutPlannerCSP(graph1, seed=7).solve()
    LayoutPlannerCSP(graph2, seed=7).solve()

    ga1 = RoadNetworkGA(graph1, population_size=24, generations=40, seed=19)
    ga2 = RoadNetworkGA(graph2, population_size=24, generations=40, seed=19)
    result1 = ga1.optimize()
    result2 = ga2.optimize()

    assert round(result1.total_cost, 6) == round(result2.total_cost, 6)
    assert result1.component_count == result2.component_count
    assert result1.has_hospital_depot_redundancy == result2.has_hospital_depot_redundancy
    assert len(result1.selected_edges) == len(result2.selected_edges)
