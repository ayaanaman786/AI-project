"""End-to-end determinism check for SetupPipeline.

Running the same SeedBundle twice must produce byte-identical artifacts:
layout, road network, ambulance positions, KMeans clusters, predicted risk
labels, police posts, civilians, and team start. This guards against
inadvertent reliance on global RNG state, dict ordering, or wall-clock
behavior anywhere in the pipeline.
"""

from __future__ import annotations

from citymind.city_graph import LocationType
from citymind.ui import SeedBundle, SetupPipeline


def _node_signature(graph) -> dict:
    """Stable, JSON-comparable summary of all per-cell state we care about."""
    sig = {}
    for cell, node in sorted(graph.nodes.items()):
        sig[cell] = (
            node.location_type.value if node.location_type is not None else None,
            node.population,
            node.risk_level.value,
            round(node.risk_mult, 6),
            node.cluster_id,
            node.police_post,
            node.accessible,
        )
    return sig


def _edge_signature(graph) -> dict:
    sig = {}
    for a, neighbors in sorted(graph.adjacency.items()):
        for b, edge in sorted(neighbors.items()):
            if a < b:
                sig[(a, b)] = (round(edge.base_cost, 6), edge.blocked)
    return sig


def test_setup_pipeline_is_byte_for_byte_deterministic() -> None:
    seeds = SeedBundle()
    a = SetupPipeline(rows=8, cols=8, civilian_count=2, seeds=seeds).run()
    b = SetupPipeline(rows=8, cols=8, civilian_count=2, seeds=seeds).run()

    # Layout
    assert _node_signature(a.graph) == _node_signature(b.graph)
    assert _edge_signature(a.graph) == _edge_signature(b.graph)
    assert a.layout_result.success == b.layout_result.success
    assert a.layout_result.violations == b.layout_result.violations
    assert a.layout_result.violation_breakdown == b.layout_result.violation_breakdown

    # Roads
    assert sorted(a.road_result.selected_edges) == sorted(b.road_result.selected_edges)
    assert a.road_result.has_hospital_depot_redundancy == b.road_result.has_hospital_depot_redundancy
    assert round(a.road_result.total_cost, 6) == round(b.road_result.total_cost, 6)

    # Ambulances
    assert a.ambulance_result.positions == b.ambulance_result.positions
    assert round(a.ambulance_result.worst_case_distance, 6) == round(
        b.ambulance_result.worst_case_distance, 6
    )

    # Clustering and predicted risk
    assert a.clustering_result.cluster_ids == b.clustering_result.cluster_ids
    assert a.risk_result.labels_predicted == b.risk_result.labels_predicted
    assert a.risk_result.class_distribution == b.risk_result.class_distribution

    # Police posts
    assert a.police_result.positions == b.police_result.positions

    # Mission
    assert a.team_start == b.team_start
    assert a.civilians == b.civilians


def test_changing_mission_seed_changes_civilians() -> None:
    """Determinism is driven by seeds, not hidden state. Mission civilian
    selection uses an RNG, so two distinct mission seeds must yield different
    civilian samples even when the layout is identical."""
    a = SetupPipeline(rows=8, cols=8, civilian_count=3, seeds=SeedBundle(mission=17)).run()
    b = SetupPipeline(rows=8, cols=8, civilian_count=3, seeds=SeedBundle(mission=42)).run()
    # Layout is fully determined by backtracking, so it must agree even though
    # mission seeds differ.
    assert _node_signature(a.graph) == _node_signature(b.graph)
    # But the chosen civilians should differ for two distinct seeds.
    assert a.civilians != b.civilians


def test_layout_seed_affects_min_conflicts_fallback_path() -> None:
    """When CSP has to fall back to min-conflicts (infeasible quotas), the
    layout seed drives the random initial assignment and swap order, so two
    layout seeds should produce different best-effort assignments."""
    quotas = {
        LocationType.HOSPITAL: 4,
        LocationType.INDUSTRIAL: 4,
        LocationType.AMBULANCE_DEPOT: 1,
        LocationType.RESIDENTIAL: 0,
        LocationType.SCHOOL: 0,
        LocationType.POWER_PLANT: 0,
    }
    from citymind.city_graph import CityGraph
    from citymind.layout_csp import LayoutPlannerCSP

    g1 = CityGraph(3, 3)
    g2 = CityGraph(3, 3)
    r1 = LayoutPlannerCSP(graph=g1, quotas=quotas, seed=7).solve()
    r2 = LayoutPlannerCSP(graph=g2, quotas=quotas, seed=42).solve()
    assert r1.success is False
    assert r2.success is False
    # Different seeds should produce different fallback layouts.
    assert r1.assignment != r2.assignment
