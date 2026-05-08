"""Challenge 5 Step 1: unsupervised neighborhood clustering with K-Means."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .city_graph import Cell, CityGraph, LocationType


@dataclass
class KMeansClusteringResult:
    cells: List[Cell]
    feature_matrix: np.ndarray
    feature_names: List[str]
    cluster_ids: List[int]
    centroids: np.ndarray


class CrimeKMeansClusterer:
    def __init__(
        self,
        k: int = 3,
        seed: int = 29,
        max_iter: int = 300,
        normalize_features: bool = True,
    ) -> None:
        self.k = k
        self.seed = seed
        self.max_iter = max_iter
        self.normalize_features = normalize_features
        self._type_order: List[LocationType] = [
            LocationType.RESIDENTIAL,
            LocationType.HOSPITAL,
            LocationType.SCHOOL,
            LocationType.INDUSTRIAL,
            LocationType.POWER_PLANT,
            LocationType.AMBULANCE_DEPOT,
        ]
        self._feature_names: List[str] = [
            "population_density",
            "dist_to_nearest_industrial",
            "industrial_within_3_hops",
        ] + [f"type_{t.value.replace(' ', '_').lower()}" for t in self._type_order]

    def fit_assign(self, graph: CityGraph, cells: Optional[Sequence[Cell]] = None) -> KMeansClusteringResult:
        target_cells = list(cells) if cells is not None else self._default_target_cells(graph)
        if not target_cells:
            raise ValueError("No target cells available for K-Means clustering")

        feature_matrix = self._build_feature_matrix(graph, target_cells)
        transformed = self._prepare_features(feature_matrix)

        model = KMeans(
            n_clusters=self.k,
            random_state=self.seed,
            n_init=10,
            max_iter=self.max_iter,
        )
        labels = model.fit_predict(transformed).tolist()

        for cell, cluster_id in zip(target_cells, labels):
            graph.set_cluster(cell, int(cluster_id))

        return KMeansClusteringResult(
            cells=target_cells,
            feature_matrix=feature_matrix,
            feature_names=self._feature_names[:],
            cluster_ids=[int(x) for x in labels],
            centroids=model.cluster_centers_,
        )

    def _default_target_cells(self, graph: CityGraph) -> List[Cell]:
        # Neighborhood cells for Step 1: residential + mixed city-function cells.
        return sorted(
            [
                cell
                for cell, node in graph.nodes.items()
                if node.location_type in {LocationType.RESIDENTIAL, LocationType.INDUSTRIAL}
            ]
        )

    def _build_feature_matrix(self, graph: CityGraph, cells: Sequence[Cell]) -> np.ndarray:
        industrial_cells = graph.find_by_type(LocationType.INDUSTRIAL)
        rows: List[List[float]] = []
        for cell in cells:
            node = graph.nodes[cell]
            population_density = float(node.population)
            nearest_industrial = self._nearest_industrial_manhattan(cell, industrial_cells, graph)
            industrial_3_hops = float(self._industrial_within_3_hops(cell, industrial_cells, graph))
            one_hot = self._one_hot_type(node.location_type)
            rows.append([population_density, nearest_industrial, industrial_3_hops, *one_hot])
        return np.array(rows, dtype=float)

    def _prepare_features(self, feature_matrix: np.ndarray) -> np.ndarray:
        if not self.normalize_features:
            return feature_matrix
        transformed = feature_matrix.copy()
        scaler = StandardScaler()
        transformed[:, :3] = scaler.fit_transform(transformed[:, :3])
        return transformed

    def _nearest_industrial_manhattan(
        self, cell: Cell, industrial_cells: Sequence[Cell], graph: CityGraph
    ) -> float:
        if not industrial_cells:
            return float(graph.rows + graph.cols)
        return float(min(abs(cell[0] - i[0]) + abs(cell[1] - i[1]) for i in industrial_cells))

    def _industrial_within_3_hops(
        self, cell: Cell, industrial_cells: Sequence[Cell], graph: CityGraph
    ) -> int:
        count = 0
        for industrial in industrial_cells:
            hops = graph.hop_distance(cell, industrial, max_hops=3)
            if hops is not None and hops <= 3:
                count += 1
        return count

    def _one_hot_type(self, location_type: Optional[LocationType]) -> List[float]:
        return [1.0 if location_type == t else 0.0 for t in self._type_order]
