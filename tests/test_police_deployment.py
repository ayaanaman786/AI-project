from citymind.city_graph import CityGraph, LocationType, RiskLevel
from citymind.police_deployment import PoliceDeploymentPlanner


def _grid_with_risk(rows: int = 5, cols: int = 5) -> CityGraph:
    graph = CityGraph(rows, cols)
    for r in range(rows):
        for c in range(cols):
            graph.set_location((r, c), LocationType.RESIDENTIAL, population=10 + r * cols + c)
            level = RiskLevel.LOW
            if (r + c) % 5 == 0:
                level = RiskLevel.HIGH
            elif (r + c) % 3 == 0:
                level = RiskLevel.MEDIUM
            graph.set_risk((r, c), level)
    return graph


def test_plan_returns_at_most_requested_positions_and_marks_graph() -> None:
    graph = _grid_with_risk()
    planner = PoliceDeploymentPlanner(graph=graph, num_officers=5, coverage_radius=1)
    result = planner.plan()
    assert len(result.positions) <= 5
    assert len(set(result.positions)) == len(result.positions)
    for cell in result.positions:
        assert graph.nodes[cell].police_post is True


def test_total_covered_risk_beats_random_baseline() -> None:
    graph = _grid_with_risk(6, 6)
    planner = PoliceDeploymentPlanner(graph=graph, num_officers=4, coverage_radius=1)
    result = planner.plan()
    arbitrary = sorted(graph.nodes.keys())[:4]
    arbitrary_total = sum(
        graph.nodes[c].risk_mult
        for c in {n for cell in arbitrary for n in planner._cells_within_hops(cell, 1)}
    )
    assert result.total_covered_risk >= arbitrary_total


def test_min_spacing_prevents_adjacent_posts() -> None:
    graph = _grid_with_risk(4, 4)
    planner = PoliceDeploymentPlanner(graph=graph, num_officers=3, coverage_radius=1, min_spacing=1)
    result = planner.plan()
    for i, a in enumerate(result.positions):
        for b in result.positions[i + 1 :]:
            assert abs(a[0] - b[0]) + abs(a[1] - b[1]) > 1
