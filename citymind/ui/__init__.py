"""CityMind 3D UI package (Ursina-based)."""

from .setup_pipeline import SetupArtifacts, SetupPipeline, SeedBundle

__all__ = ["SetupArtifacts", "SetupPipeline", "SeedBundle"]


def _import_app():
    from .app import UIApp

    return UIApp


def __getattr__(name):
    if name == "UIApp":
        return _import_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
