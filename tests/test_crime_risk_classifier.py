import pytest

from citymind.city_graph import CityGraph, LocationType, RiskLevel
from citymind.astar_router import AStarRouter
from citymind.ambulance_sa import AmbulancePlacementSA
from citymind.crime_kmeans import CrimeKMeansClusterer
from citymind.crime_risk_classifier import CrimeRiskClassifier
from citymind.layout_csp import LayoutPlannerCSP
from citymind.road_ga import RoadNetworkGA


def _layout_graph(rows: int = 10, cols: int = 10) -> CityGraph:
    graph = CityGraph(rows, cols)
    result = LayoutPlannerCSP(graph, seed=7).solve()
    assert result.success is True
    return graph


def test_synthetic_data_is_reproducible_under_seed() -> None:
    graph = _layout_graph(8, 8)
    clustering = CrimeKMeansClusterer(k=3, seed=29).fit_assign(graph)
    pipeline_a = CrimeRiskClassifier(seed=31)
    pipeline_b = CrimeRiskClassifier(seed=31)

    data_a = pipeline_a.generate_synthetic_data(clustering)
    data_b = pipeline_b.generate_synthetic_data(clustering)
    assert data_a.labels == data_b.labels
    assert data_a.class_distribution == data_b.class_distribution
    assert data_a.incident_rates.tolist() == data_b.incident_rates.tolist()


def test_labeling_covers_all_rows_and_balances_terciles() -> None:
    graph = _layout_graph(8, 8)
    clustering = CrimeKMeansClusterer(k=3, seed=29).fit_assign(graph)
    data = CrimeRiskClassifier(seed=31).generate_synthetic_data(clustering)
    assert len(data.labels) == len(clustering.cells)
    assert set(data.labels) == {RiskLevel.LOW.value, RiskLevel.MEDIUM.value, RiskLevel.HIGH.value}
    counts = data.class_distribution
    assert max(counts.values()) - min(counts.values()) <= 1


def test_decision_tree_metrics_and_graph_writeback() -> None:
    graph = _layout_graph(8, 8)
    clustering = CrimeKMeansClusterer(k=3, seed=29).fit_assign(graph)
    pipeline = CrimeRiskClassifier(seed=31)
    synthetic_data, model_result = pipeline.run(graph, clustering)

    assert 0.0 <= model_result.accuracy <= 1.0
    assert model_result.confusion_labels == [RiskLevel.LOW.value, RiskLevel.MEDIUM.value, RiskLevel.HIGH.value]
    assert len(model_result.confusion_matrix_rows) == 3
    assert all(len(row) == 3 for row in model_result.confusion_matrix_rows)
    assert sum(sum(row) for row in model_result.confusion_matrix_rows) > 0
    assert sum(synthetic_data.class_distribution.values()) == len(clustering.cells)

    for cell in clustering.cells:
        assert graph.nodes[cell].risk_level in {RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH}
        assert graph.nodes[cell].risk_mult in {1.0, 1.5, 2.0}


def test_effective_edge_cost_responds_to_risk_multiplier() -> None:
    graph = CityGraph(2, 2)
    graph.set_location((0, 0), LocationType.RESIDENTIAL, population=50)
    graph.set_location((0, 1), LocationType.RESIDENTIAL, population=45)
    base = graph.effective_cost((0, 0), (0, 1))
    graph.set_risk((0, 0), RiskLevel.HIGH)
    graph.set_risk((0, 1), RiskLevel.MEDIUM)
    bumped = graph.effective_cost((0, 0), (0, 1))
    assert base == 0.8
    assert bumped == pytest.approx(1.4)


def test_challenges_2_3_4_consume_risk_adjusted_edge_costs() -> None:
    graph = _layout_graph(8, 8)
    a, b = (0, 0), (0, 1)
    graph.set_location(a, LocationType.HOSPITAL, population=0)
    graph.set_location(b, LocationType.AMBULANCE_DEPOT, population=0)

    ga = RoadNetworkGA(graph=graph, use_effective_cost=True, seed=11)
    ga_before = ga._edge_cost((a, b))

    router = AStarRouter(min_edge_cost=0.8)
    route_before = router.find_path(graph, a, b).cost

    sa = AmbulancePlacementSA(graph=graph, seed=23)
    sa_before = sa._dijkstra_distances(a).get(b, float("inf"))

    graph.set_risk(a, RiskLevel.HIGH)
    graph.set_risk(b, RiskLevel.MEDIUM)

    ga_after = ga._edge_cost((a, b))
    route_after = router.find_path(graph, a, b).cost
    sa_after = sa._dijkstra_distances(a).get(b, float("inf"))

    assert ga_after > ga_before
    assert route_after > route_before
    assert sa_after > sa_before
