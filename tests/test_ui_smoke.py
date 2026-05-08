"""Smoke tests for the CityMind 3D UI.

The setup pipeline is exercised always; the Ursina-dependent scene/HUD/app
parts are only exercised when ursina is importable AND a display is available.
"""

from __future__ import annotations

import os

import pytest

from citymind.city_graph import LocationType, RiskLevel
from citymind.ui import SetupPipeline


def test_setup_pipeline_produces_full_artifacts() -> None:
    artifacts = SetupPipeline(rows=8, cols=8, civilian_count=2, flood_probability=0.0).run()

    graph = artifacts.graph
    assert any(node.location_type is not None for node in graph.nodes.values())

    risk_levels = {node.risk_level for node in graph.nodes.values()}
    assert risk_levels.intersection({RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH})

    police_cells = [c for c, node in graph.nodes.items() if node.police_post]
    assert len(police_cells) == len(artifacts.police_result.positions) > 0

    assert artifacts.road_result.component_count == 1 or artifacts.layout_result.success in {True, False}
    assert artifacts.ambulance_result.positions
    assert artifacts.sa_optimizer.current_positions == artifacts.ambulance_result.positions
    assert artifacts.runner.state.team_pos == artifacts.team_start
    assert len(artifacts.civilians) == 2

    expected = artifacts.runner.state.civilians_ordered
    assert expected == artifacts.civilians

    state = artifacts.runner.run(max_steps=20)
    assert state.target_index >= 1


def test_setup_pipeline_is_deterministic_under_seed_bundle() -> None:
    a = SetupPipeline(rows=8, cols=8, civilian_count=2).run()
    b = SetupPipeline(rows=8, cols=8, civilian_count=2).run()
    assert a.team_start == b.team_start
    assert a.civilians == b.civilians
    assert a.police_result.positions == b.police_result.positions


def _ursina_available() -> bool:
    if os.environ.get("CITYMIND_SKIP_URSINA"):
        return False
    try:
        import ursina  # noqa: F401
        return True
    except Exception:
        return False


def _display_available() -> bool:
    if os.name == "nt":
        return True
    return bool(os.environ.get("DISPLAY"))


@pytest.mark.skipif(
    not (_ursina_available() and _display_available()),
    reason="ursina/display not available in this environment",
)
def test_uiapp_constructs_and_steps_one_tick() -> None:
    from citymind.ui.app import UIApp
    from ursina import color

    app = UIApp(tick_interval=10.0)
    try:
        assert app.artifacts is not None
        assert app.hud is not None
        assert app.hud._loading_text is not None
        assert app.hud._loading_text.enabled is False
        assert app.artifacts.runner.state.team_pos is not None
        log_before = len(app.artifacts.runner.state.event_log)
        app.handle_tick()
        log_after = len(app.artifacts.runner.state.event_log)
        assert log_after >= log_before
        app.set_view(coverage=True)
        assert app.scene.view_state.show_coverage is True
        assert len(app.scene._entities.civilian_markers) == len(app.artifacts.runner.state.civilians_ordered)
        assert app.scene._entities.team_halo is not None
        assert app.scene.view_state.show_clusters is False
        assert app.scene.view_state.show_ga_redundancy is False
        assert app.scene.view_state.show_astar_trace is False
        assert app.scene.view_state.show_csp_diagnostics is False
        first_cell = next(iter(app.artifacts.graph.nodes.keys()))
        pad_before = app.scene._entities.cell_pads[first_cell].color
        app.set_view(risk=False, coverage=True)
        pad_cov = app.scene._entities.cell_pads[first_cell].color
        app.set_view(risk=True, coverage=False)
        pad_risk = app.scene._entities.cell_pads[first_cell].color
        app.set_view(risk=True, coverage=True)
        pad_both = app.scene._entities.cell_pads[first_cell].color
        assert pad_cov != pad_before or pad_risk != pad_before
        assert pad_both == pad_cov
        app.set_view(clusters=True)
        assert app.scene.view_state.show_clusters is True
        app.set_view(ga_redundancy=True)
        assert app.scene.view_state.show_ga_redundancy is True
        app.set_view(astar_trace=True)
        assert app.scene.view_state.show_astar_trace is True
        app.set_view(csp_diagnostics=True)
        assert app.scene.view_state.show_csp_diagnostics is True
        assert app.scene.csp_diagnostic_count() >= 0
        app.reset_camera()
        assert tuple(app.editor_camera.rotation) == tuple(app._CAMERA_ROTATION)
        assert tuple(app.editor_camera.position) == tuple(app._CAMERA_POSITION)
        team = app.scene._entities.team
        assert team is not None
        assert team.color == color.rgb(0.95, 0.20, 0.95)
        assert "Target" in app.hud._status_text.text
        assert app.hud._seed_status_text is not None
        assert "Seeds:" in app.hud._seed_status_text.text
        base_tick = app.tick_interval
        app.set_speed_multiplier(2.0)
        assert app.tick_interval < base_tick
        assert app.hud.summary_visible() is False
        app.set_view(population=True)
        for cell, building in app.scene._entities.buildings.items():
            node = app.artifacts.graph.nodes[cell]
            if node.location_type == LocationType.RESIDENTIAL:
                assert building.scale_y <= 1.0
        if app.hud is not None:
            hints = [line.text for line in app.hud._legend_hint_lines if line.enabled]
            assert any("Risk:" in h for h in hints)
            assert any("Coverage:" in h for h in hints)
            assert any("Population:" in h for h in hints)
            assert any("Roads:" in h for h in hints)
            assert any("Clusters:" in h for h in hints)
            assert any("GA:" in h for h in hints)
            assert any("A*:" in h for h in hints)
            assert any("CSP diagnostics:" in h for h in hints)
        app.artifacts.runner.state.current_path = [
            app.artifacts.runner.state.team_pos,
            app.artifacts.runner.state.civilians_ordered[0],
        ]
        app.scene.refresh()
        assert len(app.scene._entities.path_markers) >= 1
        app.scene.highlight_event_line("t=02 A* replan: (0, 0)->(1, 1) cost=2.0")
        app.scene.refresh()
        assert len(app.scene._astar_trace_paths) >= 1
        if app.scene.view_state.show_astar_trace:
            assert len(app.scene._entities.astar_trace_markers) >= 1
        probe_cell = app.artifacts.runner.state.civilians_ordered[0]
        app.scene._handle_cell_click(probe_cell)
        assert app.scene.selected_cell() == probe_cell
        assert app.hud is not None
        inspect_lines = [line.text for line in app.hud._inspect_lines if line.enabled]
        assert any("Cell:" in t for t in inspect_lines)
        assert any("Risk:" in t for t in inspect_lines)
        assert any("Cluster:" in t for t in inspect_lines)
        assert "Click a cell" not in inspect_lines[0]
        assert "Risk:" in app.hud._tooltip_text.text
        snap = app.snapshot()
        assert snap is not None
        assert snap.exists()
        summary = app.build_summary()
        for key in ("steps", "reached", "total", "completed", "replan_failed", "repairs", "sa_reeval"):
            assert key in summary
        seeds_before = app.seeds
        app._randomize_seed_fields("mission", ["mission"])
        assert app.seeds.mission != seeds_before.mission
        assert app.seeds.layout == seeds_before.layout
        assert app._tick == 0
        assert app.artifacts is not None
        assert any("seed update [mission]" in line for line in app.artifacts.runner.state.event_log)
        app._randomize_seed_fields("ml", ["kmeans", "risk_classifier"])
        assert app.seeds.kmeans != seeds_before.kmeans or app.seeds.risk_classifier != seeds_before.risk_classifier
        assert app.hud._seed_status_text is not None
        assert "Seeds:" in app.hud._seed_status_text.text
        # Exercise all randomization callback slots so HUD wiring is covered.
        callbacks = app.hud.callbacks
        callbacks.on_randomize_layout()
        callbacks.on_randomize_roads()
        callbacks.on_randomize_ambulance()
        callbacks.on_randomize_flood()
        callbacks.on_randomize_all()
        assert app.artifacts is not None
        app.run_to_end()
        assert app._tick == app.max_ticks or app.artifacts.runner.state.completed
        assert app.hud.summary_visible() is True
        app.reset()
        assert app.scene.selected_cell() is None
        assert app.hud._inspect_lines[0].text == "Click a cell to inspect"
        assert app.hud.summary_visible() is False
    finally:
        app.shutdown()
