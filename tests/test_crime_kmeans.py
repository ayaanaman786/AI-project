from collections import Counter

from citymind.city_graph import CityGraph, LocationType
from citymind.crime_kmeans import CrimeKMeansClusterer
from citymind.layout_csp import LayoutPlannerCSP


def _layout_graph(rows: int = 10, cols: int = 10) -> CityGraph:
    graph = CityGraph(rows, cols)
    result = LayoutPlannerCSP(graph, seed=7).solve()
    assert result.success is True
    return graph


def test_kmeans_assigns_cluster_ids_and_writes_to_graph() -> None:
    graph = _layout_graph(8, 8)
    clusterer = CrimeKMeansClusterer(k=3, seed=29)
    result = clusterer.fit_assign(graph)
    assert len(result.cells) > 0
    assert len(result.cluster_ids) == len(result.cells)
    for cell in result.cells:
        assert graph.nodes[cell].cluster_id is not None


def test_feature_schema_has_expected_shape() -> None:
    graph = _layout_graph(8, 8)
    clusterer = CrimeKMeansClusterer(k=3, seed=29)
    result = clusterer.fit_assign(graph)
    assert result.feature_matrix.shape[0] == len(result.cells)
    assert result.feature_matrix.shape[1] == len(result.feature_names)
    assert "population_density" in result.feature_names
    assert "dist_to_nearest_industrial" in result.feature_names
    assert "industrial_within_3_hops" in result.feature_names


def test_kmeans_deterministic_under_fixed_seed() -> None:
    graph1 = _layout_graph(8, 8)
    graph2 = _layout_graph(8, 8)
    c1 = CrimeKMeansClusterer(k=3, seed=29).fit_assign(graph1)
    c2 = CrimeKMeansClusterer(k=3, seed=29).fit_assign(graph2)
    assert c1.cells == c2.cells
    assert Counter(c1.cluster_ids) == Counter(c2.cluster_ids)


def test_no_industrial_cells_is_handled() -> None:
    graph = CityGraph(4, 4)
    for r in range(graph.rows):
        for c in range(graph.cols):
            graph.set_location((r, c), LocationType.RESIDENTIAL, population=40 + (r * 10) + c)
    clusterer = CrimeKMeansClusterer(k=3, seed=29)
    result = clusterer.fit_assign(graph, cells=sorted(graph.nodes.keys()))
    assert len(result.cells) == 16
    assert all(cid in {0, 1, 2} for cid in result.cluster_ids)
