"""Setup pipeline: runs all CityMind challenges against a fresh CityGraph.

This module is intentionally free of any UI dependencies so the simulation
can be exercised headlessly (e.g. in tests).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ..ambulance_sa import AmbulancePlacementResult, AmbulancePlacementSA
from ..city_graph import Cell, CityGraph, LocationType
from ..crime_kmeans import CrimeKMeansClusterer, KMeansClusteringResult
from ..crime_risk_classifier import (
    CrimeRiskClassifier,
    CrimeRiskModelResult,
    SyntheticCrimeData,
)
from ..layout_csp import LayoutPlannerCSP, LayoutResult
from ..police_deployment import PoliceDeploymentPlanner, PoliceDeploymentResult
from ..road_ga import RoadNetworkGA, RoadNetworkResult
from ..simulation import EmergencyMissionRunner, default_mission


@dataclass(frozen=True)
class SeedBundle:
    """Seed values that fully determine a setup run."""

    layout: int = 7
    kmeans: int = 29
    risk_classifier: int = 31
    road_ga: int = 11
    sa: int = 23
    mission: int = 17
    flood: int = 101


@dataclass
class SetupArtifacts:
    """Concrete output of a SetupPipeline run."""

    graph: CityGraph
    layout_result: LayoutResult
    clustering_result: KMeansClusteringResult
    synthetic_data: SyntheticCrimeData
    risk_result: CrimeRiskModelResult
    police_result: PoliceDeploymentResult
    road_result: RoadNetworkResult
    ambulance_result: AmbulancePlacementResult
    sa_optimizer: AmbulancePlacementSA
    runner: EmergencyMissionRunner
    team_start: Cell
    civilians: List[Cell]
    seeds: SeedBundle
    notes: List[str] = field(default_factory=list)


class SetupPipeline:
    """Wires Challenges 1-5 + police deployment + mission generation together."""

    def __init__(
        self,
        rows: int = 10,
        cols: int = 10,
        seeds: Optional[SeedBundle] = None,
        civilian_count: int = 3,
        flood_probability: float = 0.015,
        num_officers: int = 10,
        coverage_radius: int = 2,
        risk_shift_probability: float = 0.35,
        sa_reopt_threshold: int = 2,
        max_blocks_per_tick: int = 1,
        repair_after_failures: int = 2,
        hospital_max_hops: int = 3,
        industrial_max_hops_for_power: int = 2,
        industrial_forbidden_neighbors: Optional[set] = None,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.seeds = seeds or SeedBundle()
        self.civilian_count = civilian_count
        self.flood_probability = flood_probability
        self.num_officers = num_officers
        self.coverage_radius = coverage_radius
        self.risk_shift_probability = risk_shift_probability
        self.sa_reopt_threshold = sa_reopt_threshold
        self.max_blocks_per_tick = max_blocks_per_tick
        self.repair_after_failures = repair_after_failures
        self.hospital_max_hops = hospital_max_hops
        self.industrial_max_hops_for_power = industrial_max_hops_for_power
        self.industrial_forbidden_neighbors = industrial_forbidden_neighbors

    def run(self) -> SetupArtifacts:
        notes: List[str] = []
        graph = CityGraph(rows=self.rows, cols=self.cols)

        layout_result = LayoutPlannerCSP(
            graph=graph,
            seed=self.seeds.layout,
            hospital_max_hops=self.hospital_max_hops,
            industrial_max_hops_for_power=self.industrial_max_hops_for_power,
            industrial_forbidden_neighbors=self.industrial_forbidden_neighbors,
        ).solve()
        if layout_result.violation_breakdown:
            notes.append(
                f"Layout success={layout_result.success} "
                f"violations={layout_result.violations} "
                f"breakdown={dict(layout_result.violation_breakdown)}"
            )
        else:
            notes.append(
                f"Layout success={layout_result.success} violations={layout_result.violations}"
            )

        clustering_result = CrimeKMeansClusterer(k=3, seed=self.seeds.kmeans).fit_assign(graph)
        notes.append(f"K-Means clustered cells={len(clustering_result.cells)}")

        risk_pipeline = CrimeRiskClassifier(seed=self.seeds.risk_classifier)
        synthetic_data, risk_result = risk_pipeline.run(graph, clustering_result)
        notes.append(
            f"DecisionTree accuracy={round(risk_result.accuracy, 4)} "
            f"distribution={risk_result.class_distribution}"
        )

        police_planner = PoliceDeploymentPlanner(
            graph=graph,
            num_officers=self.num_officers,
            coverage_radius=self.coverage_radius,
        )
        police_result = police_planner.plan()
        notes.append(
            f"Police posts={len(police_result.positions)} "
            f"covered_risk={round(police_result.total_covered_risk, 3)}"
        )

        road_ga = RoadNetworkGA(
            graph=graph,
            use_effective_cost=True,
            seed=self.seeds.road_ga,
        )
        road_result = road_ga.optimize()
        road_ga.apply_selected_roads(road_result.selected_edges)
        notes.append(
            f"Roads built={len(road_result.selected_edges)} "
            f"cost={round(road_result.total_cost, 2)} "
            f"redundant={road_result.has_hospital_depot_redundancy}"
        )

        sa_optimizer = AmbulancePlacementSA(graph=graph, seed=self.seeds.sa)
        ambulance_result = sa_optimizer.optimize()
        sa_optimizer.apply_positions(ambulance_result.positions)
        notes.append(
            f"Ambulances={ambulance_result.positions} "
            f"worst_case={round(ambulance_result.worst_case_distance, 3)}"
        )

        team_start, civilians = default_mission(
            graph,
            civilian_count=self.civilian_count,
            seed=self.seeds.mission,
        )
        runner = EmergencyMissionRunner(
            graph=graph,
            team_start=team_start,
            civilians_ordered=civilians,
            flood_probability=self.flood_probability,
            seed=self.seeds.flood,
            risk_shift_probability=self.risk_shift_probability,
            sa_reopt_threshold=self.sa_reopt_threshold,
            sa_optimizer=sa_optimizer,
            max_blocks_per_tick=self.max_blocks_per_tick,
            repair_after_failures=self.repair_after_failures,
        )
        notes.append(f"Mission start={team_start} civilians={civilians}")

        return SetupArtifacts(
            graph=graph,
            layout_result=layout_result,
            clustering_result=clustering_result,
            synthetic_data=synthetic_data,
            risk_result=risk_result,
            police_result=police_result,
            road_result=road_result,
            ambulance_result=ambulance_result,
            sa_optimizer=sa_optimizer,
            runner=runner,
            team_start=team_start,
            civilians=civilians,
            seeds=self.seeds,
            notes=notes,
        )
