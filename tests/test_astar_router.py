import pytest

from citymind.astar_router import AStarRouter
from citymind.city_graph import CityGraph, LocationType


def _seed_graph() -> CityGraph:
    graph = CityGraph(3, 3)
    for c in graph.nodes:
        graph.set_location(c, LocationType.RESIDENTIAL, population=100)
    return graph


def test_heuristic_sanity() -> None:
    router = AStarRouter()
    assert router._heuristic((0, 0), (0, 0)) == 0.0
    assert router._heuristic((0, 0), (1, 1)) >= 0.0


def test_astar_matches_dijkstra_cost_on_static_graph() -> None:
    graph = _seed_graph()
    router = AStarRouter()
    result = router.find_path(graph, (0, 0), (2, 2))
    _, dijkstra_cost = graph.shortest_path((0, 0), (2, 2))
    assert result.found is True
    assert result.cost == dijkstra_cost


def test_astar_finds_alternative_when_edge_blocked() -> None:
    graph = _seed_graph()
    graph.set_edge_blocked((0, 0), (0, 1), True)
    router = AStarRouter()
    result = router.find_path(graph, (0, 0), (0, 2))
    assert result.found is True
    assert result.path[0] == (0, 0)
    assert result.path[-1] == (0, 2)
    assert (0, 1) not in result.path[:2]


def test_astar_returns_not_found_when_unreachable() -> None:
    graph = _seed_graph()
    center = (1, 1)
    for n in graph.neighbors(center, include_blocked=True):
        graph.set_edge_blocked(center, n, True)
    router = AStarRouter()
    result = router.find_path(graph, (0, 0), center)
    assert result.found is False
    assert result.cost == float("inf")


def test_heuristic_admissibility_guard_rejects_too_high_floor() -> None:
    """Ensure the router refuses to run with an inadmissible heuristic floor.

    With residential edges at base 0.8, a heuristic floor of 1.5 would
    overestimate true cost in many configurations and break A*'s shortest-path
    guarantee. The guard must raise ValueError on `find_path` rather than
    silently produce wrong answers."""
    graph = _seed_graph()
    router = AStarRouter(min_edge_cost=1.5)
    with pytest.raises(ValueError):
        router.find_path(graph, (0, 0), (2, 2))


def test_min_base_cost_property_reflects_residential_discount() -> None:
    graph = _seed_graph()
    assert graph.min_base_cost == pytest.approx(0.8)
    plain = CityGraph(2, 2)
    assert plain.min_base_cost == pytest.approx(1.0)
