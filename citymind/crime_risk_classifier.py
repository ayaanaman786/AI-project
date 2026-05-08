"""Challenge 5 Step 2+3: synthetic labels + Decision Tree risk classification."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from .city_graph import Cell, CityGraph, RiskLevel
from .crime_kmeans import KMeansClusteringResult


RISK_LABELS: List[str] = [RiskLevel.LOW.value, RiskLevel.MEDIUM.value, RiskLevel.HIGH.value]


@dataclass
class SyntheticCrimeData:
    incident_rates: np.ndarray
    labels: List[str]
    industrial_proximity_scores: np.ndarray
    cluster_risk_biases: np.ndarray
    feature_contributions: Dict[str, np.ndarray]
    class_distribution: Dict[str, int]
    cluster_bias_by_id: Dict[int, float]


@dataclass
class CrimeRiskModelResult:
    labels_predicted: List[str]
    accuracy: float
    confusion_matrix_rows: List[List[int]]
    confusion_labels: List[str]
    class_distribution: Dict[str, int]
    model: DecisionTreeClassifier


class CrimeRiskClassifier:
    def __init__(
        self,
        w1: float = 0.55,
        w2: float = 0.30,
        w3: float = 0.15,
        noise_sigma: float = 0.08,
        seed: int = 31,
    ) -> None:
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.noise_sigma = noise_sigma
        self.seed = seed

    def generate_synthetic_data(self, clustering: KMeansClusteringResult) -> SyntheticCrimeData:
        feature_idx = {name: i for i, name in enumerate(clustering.feature_names)}
        pop = self._minmax(clustering.feature_matrix[:, feature_idx["population_density"]])
        dist = self._minmax(clustering.feature_matrix[:, feature_idx["dist_to_nearest_industrial"]])
        industrial_proximity = 1.0 - dist
        cluster_bias_by_id = self._derive_cluster_bias(clustering)
        cluster_bias = np.array([cluster_bias_by_id[cid] for cid in clustering.cluster_ids], dtype=float)

        rng = np.random.default_rng(self.seed)
        noise = rng.normal(0.0, self.noise_sigma, size=len(clustering.cells))
        rate = (self.w1 * pop) + (self.w2 * industrial_proximity) + (self.w3 * cluster_bias) + noise

        labels = self._labels_from_terciles(rate)
        class_distribution = dict(Counter(labels))
        return SyntheticCrimeData(
            incident_rates=rate,
            labels=labels,
            industrial_proximity_scores=industrial_proximity,
            cluster_risk_biases=cluster_bias,
            feature_contributions={
                "pop_density_term": self.w1 * pop,
                "industrial_proximity_term": self.w2 * industrial_proximity,
                "cluster_risk_bias_term": self.w3 * cluster_bias,
                "noise_term": noise,
            },
            class_distribution=class_distribution,
            cluster_bias_by_id=cluster_bias_by_id,
        )

    def fit_predict_apply(
        self, graph: CityGraph, clustering: KMeansClusteringResult, synthetic_data: SyntheticCrimeData
    ) -> CrimeRiskModelResult:
        x = np.column_stack(
            [
                clustering.feature_matrix,
                np.asarray(clustering.cluster_ids, dtype=float).reshape(-1, 1),
            ]
        )
        y = np.asarray(synthetic_data.labels)

        x_train, x_test, y_train, y_test = self._split_dataset(x, y)
        model = DecisionTreeClassifier(random_state=self.seed, max_depth=5, min_samples_leaf=2)
        model.fit(x_train, y_train)

        y_test_pred = model.predict(x_test)
        accuracy = float(accuracy_score(y_test, y_test_pred))

        y_full_pred = model.predict(x).tolist()
        for cell, label in zip(clustering.cells, y_full_pred):
            graph.set_risk(cell, RiskLevel(label))

        cm = confusion_matrix(y_test, y_test_pred, labels=RISK_LABELS)
        return CrimeRiskModelResult(
            labels_predicted=y_full_pred,
            accuracy=accuracy,
            confusion_matrix_rows=cm.tolist(),
            confusion_labels=RISK_LABELS[:],
            class_distribution=dict(Counter(y_full_pred)),
            model=model,
        )

    def run(self, graph: CityGraph, clustering: KMeansClusteringResult) -> tuple[SyntheticCrimeData, CrimeRiskModelResult]:
        synthetic = self.generate_synthetic_data(clustering)
        result = self.fit_predict_apply(graph, clustering, synthetic)
        return synthetic, result

    def _split_dataset(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        class_counts = Counter(y.tolist())
        use_stratify = min(class_counts.values()) >= 2 and len(class_counts) >= 2
        try:
            return train_test_split(
                x,
                y,
                test_size=0.2,
                random_state=self.seed,
                stratify=y if use_stratify else None,
            )
        except ValueError:
            return train_test_split(x, y, test_size=0.2, random_state=self.seed, stratify=None)

    def _derive_cluster_bias(self, clustering: KMeansClusteringResult) -> Dict[int, float]:
        pop_idx = clustering.feature_names.index("population_density")
        dist_idx = clustering.feature_names.index("dist_to_nearest_industrial")
        ids = sorted(set(clustering.cluster_ids))
        risk_score_by_cluster: Dict[int, float] = {}
        for cid in ids:
            row_idx = [i for i, cluster_id in enumerate(clustering.cluster_ids) if cluster_id == cid]
            pop_mean = float(np.mean(clustering.feature_matrix[row_idx, pop_idx]))
            dist_mean = float(np.mean(clustering.feature_matrix[row_idx, dist_idx]))
            risk_score_by_cluster[cid] = pop_mean - dist_mean

        ordered = sorted(risk_score_by_cluster.items(), key=lambda kv: kv[1])
        if len(ordered) == 1:
            return {ordered[0][0]: 0.5}
        scaled = np.linspace(0.2, 1.0, num=len(ordered))
        return {cluster_id: float(scaled[i]) for i, (cluster_id, _) in enumerate(ordered)}

    def _labels_from_terciles(self, rates: np.ndarray) -> List[str]:
        n = len(rates)
        order = np.argsort(rates, kind="mergesort")
        labels = np.empty(n, dtype=object)
        low_cut = n // 3
        med_cut = (2 * n) // 3
        labels[order[:low_cut]] = RiskLevel.LOW.value
        labels[order[low_cut:med_cut]] = RiskLevel.MEDIUM.value
        labels[order[med_cut:]] = RiskLevel.HIGH.value
        return labels.tolist()

    def _minmax(self, values: np.ndarray) -> np.ndarray:
        min_v = float(np.min(values))
        max_v = float(np.max(values))
        if np.isclose(max_v, min_v):
            return np.zeros_like(values, dtype=float)
        return (values - min_v) / (max_v - min_v)
