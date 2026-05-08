from citymind.city_graph import CityGraph, LocationType, RiskLevel


def test_graph_initializes_grid_and_edges() -> None:
    graph = CityGraph(3, 3)
    assert len(graph.nodes) == 9
    assert graph.hop_distance((0, 0), (2, 2)) == 4


def test_effective_cost_uses_risk_and_blocked_state() -> None:
    graph = CityGraph(2, 2)
    graph.set_risk((0, 0), RiskLevel.HIGH)
    graph.set_risk((0, 1), RiskLevel.LOW)
    assert graph.effective_cost((0, 0), (0, 1)) == 1.5
    graph.set_edge_blocked((0, 0), (0, 1), True)
    assert graph.effective_cost((0, 0), (0, 1)) == float("inf")


def test_residential_location_applies_base_cost_discount() -> None:
    graph = CityGraph(2, 2)
    assert graph.adjacency[(0, 0)][(0, 1)].base_cost == 1.0
    graph.set_location((0, 0), LocationType.RESIDENTIAL, population=70)
    assert graph.adjacency[(0, 0)][(0, 1)].base_cost == 0.8


def test_inaccessible_node_is_skipped_by_neighbors_and_effective_cost() -> None:
    graph = CityGraph(2, 2)
    assert (0, 1) in graph.neighbors((0, 0))
    graph.set_accessible((0, 1), False)
    assert (0, 1) not in graph.neighbors((0, 0))
    assert graph.effective_cost((0, 0), (0, 1)) == float("inf")
    graph.set_accessible((0, 1), True)
    assert (0, 1) in graph.neighbors((0, 0))
