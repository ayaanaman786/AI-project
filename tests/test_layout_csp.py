from citymind.city_graph import CityGraph, LocationType
from citymind.layout_csp import (
    RULE_INDUSTRIAL_ADJACENT_TO_PROTECTED,
    RULE_QUOTA_MISMATCH,
    LayoutPlannerCSP,
)


def test_min_conflicts_uses_hop_distance_when_isolated_hospital() -> None:
    graph = CityGraph(4, 4)
    planner = LayoutPlannerCSP(graph=graph, seed=7)

    assignment = {cell: LocationType.RESIDENTIAL for cell in graph.nodes}
    assignment[(0, 0)] = LocationType.HOSPITAL
    assignment[(3, 3)] = LocationType.HOSPITAL
    assignment[(0, 3)] = LocationType.AMBULANCE_DEPOT
    assignment[(3, 0)] = LocationType.SCHOOL
    assignment[(2, 1)] = LocationType.INDUSTRIAL
    assignment[(1, 2)] = LocationType.POWER_PLANT

    far_cell = (3, 2)
    assert assignment[far_cell] == LocationType.RESIDENTIAL

    for neighbor in graph.neighbors(far_cell, include_blocked=True):
        graph.set_edge_blocked(far_cell, neighbor, True)

    planner.hospital_reach = {
        cell: planner._cells_within_hops(cell, planner.hospital_max_hops)
        for cell in graph.nodes
    }

    violations, rule, breakdown = planner._count_violations(assignment)
    assert violations >= 1
    assert rule is not None
    assert breakdown
    assert sum(breakdown.values()) == violations


def test_infeasible_layout_reports_per_rule_breakdown() -> None:
    """Force an unsatisfiable instance (industrials sharing a 3x3 grid with
    hospitals) and assert min-conflicts surfaces a per-rule breakdown."""
    graph = CityGraph(3, 3)
    quotas = {
        LocationType.HOSPITAL: 4,
        LocationType.INDUSTRIAL: 4,
        LocationType.AMBULANCE_DEPOT: 1,
        LocationType.RESIDENTIAL: 0,
        LocationType.SCHOOL: 0,
        LocationType.POWER_PLANT: 0,
    }
    planner = LayoutPlannerCSP(graph=graph, quotas=quotas, seed=7)
    result = planner.solve()

    assert result.success is False
    assert result.violations > 0
    assert isinstance(result.violation_breakdown, dict)
    assert sum(result.violation_breakdown.values()) == result.violations
    assert RULE_INDUSTRIAL_ADJACENT_TO_PROTECTED in result.violation_breakdown
    assert result.violated_rule is not None


def test_oversized_quotas_register_quota_mismatch_rule() -> None:
    """Quotas exceeding the grid budget can never be satisfied; the breakdown
    should include the quota_mismatch rule."""
    graph = CityGraph(3, 3)
    quotas = {
        LocationType.HOSPITAL: 6,  # exceeds the 9-cell budget on its own
        LocationType.INDUSTRIAL: 6,
        LocationType.AMBULANCE_DEPOT: 1,
        LocationType.RESIDENTIAL: 0,
        LocationType.SCHOOL: 0,
        LocationType.POWER_PLANT: 0,
    }
    planner = LayoutPlannerCSP(graph=graph, quotas=quotas, seed=7)
    result = planner.solve()

    assert result.success is False
    assert RULE_QUOTA_MISMATCH in result.violation_breakdown
