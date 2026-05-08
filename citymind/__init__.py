"""CityMind core package."""

from .ambulance_sa import AmbulancePlacementResult, AmbulancePlacementSA
from .astar_router import AStarResult, AStarRouter
from .city_graph import CityGraph, LocationType, RiskLevel
from .crime_risk_classifier import CrimeRiskClassifier, CrimeRiskModelResult, SyntheticCrimeData
from .crime_kmeans import CrimeKMeansClusterer, KMeansClusteringResult
from .police_deployment import PoliceDeploymentPlanner, PoliceDeploymentResult
from .road_ga import RoadNetworkGA, RoadNetworkResult
from .simulation import EmergencyMissionRunner, FloodEventManager, MissionState, default_mission

__all__ = [
    "CityGraph",
    "LocationType",
    "RiskLevel",
    "CrimeRiskClassifier",
    "CrimeRiskModelResult",
    "SyntheticCrimeData",
    "PoliceDeploymentPlanner",
    "PoliceDeploymentResult",
    "RoadNetworkGA",
    "RoadNetworkResult",
    "CrimeKMeansClusterer",
    "KMeansClusteringResult",
    "AmbulancePlacementSA",
    "AmbulancePlacementResult",
    "AStarRouter",
    "AStarResult",
    "MissionState",
    "FloodEventManager",
    "EmergencyMissionRunner",
    "default_mission",
    "UIApp",
]


def __getattr__(name):
    if name == "UIApp":
        from .ui.app import UIApp

        return UIApp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
