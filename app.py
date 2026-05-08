"""CityMind bootstrap runner for Challenge 1 + Challenge 2."""

from collections import Counter

from citymind.ambulance_sa import AmbulancePlacementSA
from citymind.city_graph import CityGraph
from citymind.crime_kmeans import CrimeKMeansClusterer
from citymind.crime_risk_classifier import CrimeRiskClassifier
from citymind.layout_csp import LayoutPlannerCSP
from citymind.police_deployment import PoliceDeploymentPlanner
from citymind.road_ga import RoadNetworkGA
from citymind.simulation import EmergencyMissionRunner, default_mission


def main() -> None:
    graph = CityGraph(rows=10, cols=10)
    planner = LayoutPlannerCSP(graph)
    layout_result = planner.solve()
    counts = Counter(node.location_type.value for node in graph.nodes.values() if node.location_type)
    print("Layout success:", layout_result.success)
    print("Violations:", layout_result.violations, "| Rule:", layout_result.violated_rule)
    if layout_result.violation_breakdown:
        print("Violation breakdown:", dict(layout_result.violation_breakdown))
    print("Type counts:", dict(counts))

    clusterer = CrimeKMeansClusterer(k=3, seed=29)
    cluster_result = clusterer.fit_assign(graph)
    cluster_counts = Counter(cluster_result.cluster_ids)
    print("KMeans clustered cells:", len(cluster_result.cells))
    print("KMeans cluster distribution:", dict(cluster_counts))

    risk_pipeline = CrimeRiskClassifier(seed=31)
    synthetic_data, risk_result = risk_pipeline.run(graph, cluster_result)
    print(
        "Synthetic rates config:",
        {
            "w1": risk_pipeline.w1,
            "w2": risk_pipeline.w2,
            "w3": risk_pipeline.w3,
            "sigma": risk_pipeline.noise_sigma,
            "seed": risk_pipeline.seed,
        },
    )
    print("Synthetic label distribution:", synthetic_data.class_distribution)
    print("Risk classifier accuracy:", round(risk_result.accuracy, 4))
    print("Risk confusion labels:", risk_result.confusion_labels)
    print("Risk confusion matrix:", risk_result.confusion_matrix_rows)
    print("Predicted risk distribution:", risk_result.class_distribution)

    police_planner = PoliceDeploymentPlanner(graph=graph, num_officers=10, coverage_radius=2)
    police_result = police_planner.plan()
    print("Police deployment positions:", police_result.positions)
    print("Police deployment covered risk total:", round(police_result.total_covered_risk, 3))

    ga = RoadNetworkGA(graph=graph, use_effective_cost=True, seed=11)
    road_result = ga.optimize()
    ga.apply_selected_roads(road_result.selected_edges)
    print("Road optimization fitness:", round(road_result.best_fitness, 2))
    print("Road total cost:", round(road_result.total_cost, 2))
    print("Road components:", road_result.component_count)
    print("Hospital-Depot redundancy:", road_result.has_hospital_depot_redundancy)
    print("Selected roads:", len(road_result.selected_edges))

    sa = AmbulancePlacementSA(graph=graph, seed=23)
    ambulance_result = sa.optimize()
    sa.apply_positions(ambulance_result.positions)
    print("Ambulance positions:", ambulance_result.positions)
    print("Worst-case response distance:", round(ambulance_result.worst_case_distance, 3))
    print("SA iterations:", ambulance_result.iterations, "| Best iteration:", ambulance_result.best_iteration)
    print("SA final temperature:", round(ambulance_result.temperature_final, 4))

    team_start, civilians = default_mission(graph, civilian_count=3, seed=17)
    runner = EmergencyMissionRunner(
        graph=graph,
        team_start=team_start,
        civilians_ordered=civilians,
        flood_probability=0.015,
        seed=101,
        sa_optimizer=sa,
        max_blocks_per_tick=1,
        repair_after_failures=2,
    )
    mission = runner.run(max_steps=20)
    print("Mission start:", team_start)
    print("Mission civilians:", civilians)
    print("Mission completed:", mission.completed, "| Reached:", mission.target_index, "/", len(civilians))
    print("Mission event log (last 10):")
    for line in mission.event_log[-10:]:
        print(" ", line)


if __name__ == "__main__":
    main()
