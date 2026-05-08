"""CityMind 3D UIApp: Ursina app + setup pipeline + scene + HUD wiring."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from math import isclose
from pathlib import Path
import random
from typing import Optional

from ursina import (
    EditorCamera,
    Entity,
    Ursina,
    application,
    camera,
    color,
    held_keys,
    time as ursina_time,
    window,
)

from .hud import Hud, HudCallbacks
from .scene import CityScene
from .setup_pipeline import SeedBundle, SetupArtifacts, SetupPipeline


class UIApp:
    """Top-level controller for the 3D UI."""

    MAX_TICKS = 20
    # Tilted isometric: pitch down, yaw, no roll. The +x offset compensates for
    # the right HUD panel being wider than the left sidebar so the city sits in
    # the visible center region instead of the geometric window center.
    _CAMERA_ROTATION = (50, 32, 0)
    _CAMERA_POSITION = (1.5, 13.5, -10.5)
    _CAMERA_FOV = 38
    _MAX_STEPS_PRESETS = (20, 30, 40, 60)

    def __init__(
        self,
        seeds: Optional[SeedBundle] = None,
        tick_interval: float = 0.9,
        headless: bool = False,
        title: str = "CityMind - Urban Intelligence System",
    ) -> None:
        self.seeds = seeds or SeedBundle()
        self.tick_interval = tick_interval
        self._base_tick_interval = tick_interval
        self._speed_multiplier = 1.0
        self.headless = headless
        self._tick = 0
        self._running = False
        self._accumulator = 0.0
        self._last_event_count = 0
        self._last_ambulances: Optional[tuple] = None
        self._action_hint = "ready"
        self._interp_active = False
        self._interp_t = 0.0
        self._interp_duration = 0.45
        self._interp_from = (0.0, 0.55, 0.0)
        self._interp_to = (0.0, 0.55, 0.0)
        self._selected_cell = None
        self._last_snapshot_path: Optional[Path] = None
        self.max_ticks = self.MAX_TICKS
        self._seed_rng = random.Random()
        self._pending_seed_log_lines: list[str] = []

        self.app: Optional[Ursina] = None
        self.scene = CityScene()
        self.hud: Optional[Hud] = None
        self.editor_camera: Optional[EditorCamera] = None
        self.artifacts: Optional[SetupArtifacts] = None
        self._update_entity: Optional[Entity] = None
        self._input_entity: Optional[Entity] = None

        if not self.headless:
            self.app = Ursina(title=title, borderless=False, vsync=True)
            try:
                window.color = color.rgb(0.05, 0.06, 0.08)
            except Exception:
                pass
            self._disable_debug_overlays()
            self.editor_camera = EditorCamera(
                rotation=self._CAMERA_ROTATION,
                position=self._CAMERA_POSITION,
            )
            camera.fov = self._CAMERA_FOV
            self.hud = Hud(self._build_callbacks())
            self.hud.build()
            self._update_entity = Entity()
            self._update_entity.update = self._on_frame
            self._input_entity = Entity()
            self._input_entity.input = self._on_input

        self._run_setup()

    def run(self) -> None:
        if self.app is None:
            raise RuntimeError("Cannot run() in headless mode")
        self.app.run()

    def step(self) -> None:
        if self.artifacts is None or self._tick >= self.max_ticks:
            return
        runner = self.artifacts.runner
        if runner.state.completed:
            return
        prev_pos = runner.state.team_pos
        prev_amb = self.artifacts.sa_optimizer.current_positions
        self._tick += 1
        runner.step(tick=self._tick)
        self._set_action_hint_from_latest_event()
        self._begin_team_interpolation(prev_pos, runner.state.team_pos)
        self._highlight_new_events()
        new_amb = self.artifacts.sa_optimizer.current_positions
        if new_amb != prev_amb:
            self.scene.invalidate_coverage_cache()
        self.scene.refresh()
        if self._interp_active:
            self.scene.set_team_visual_position(self._interp_from)
        self._refresh_hud()

    def reset(self) -> None:
        self._tick = 0
        self._running = False
        self._accumulator = 0.0
        self._last_event_count = 0
        self.scene.destroy()
        self._run_setup()

    def toggle_run(self) -> None:
        self._running = not self._running
        self._accumulator = 0.0
        self._refresh_hud()

    def handle_tick(self) -> None:
        """Public alias for the smoke test: run exactly one tick."""
        self.step()

    def set_speed_multiplier(self, speed: float) -> None:
        self._speed_multiplier = max(0.1, float(speed))
        self.tick_interval = self._base_tick_interval / self._speed_multiplier
        if self.hud is not None:
            self.hud.set_speed(self._speed_multiplier)
        self._action_hint = f"speed {self._speed_multiplier:g}x"
        self._refresh_hud()

    def run_to_end(self) -> None:
        if self.artifacts is None:
            return
        prev_interp = self._interp_active
        self._interp_active = False
        while self._tick < self.max_ticks and not self.artifacts.runner.state.completed:
            self.step()
        self._interp_active = prev_interp
        self._running = False
        self._action_hint = "run to end"
        self._refresh_hud()

    def build_summary(self) -> dict:
        if self.artifacts is None:
            return {
                "steps": self._tick,
                "reached": 0,
                "total": 0,
                "completed": False,
                "replan_failed": 0,
                "repairs": 0,
                "sa_reeval": 0,
            }
        state = self.artifacts.runner.state
        logs = state.event_log
        return {
            "steps": self._tick,
            "reached": state.target_index,
            "total": len(state.civilians_ordered),
            "completed": state.completed,
            "replan_failed": sum("replan failed" in l for l in logs),
            "repairs": sum("repair:" in l for l in logs),
            "sa_reeval": sum("SA re-eval" in l for l in logs),
        }

    def snapshot(self) -> Optional[Path]:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path.cwd() / "snapshots"
        out_dir.mkdir(exist_ok=True)
        path = out_dir / f"citymind_step{self._tick:02d}_{stamp}.png"
        saved = False
        if not self.headless:
            try:
                shot = getattr(window, "screenshot", None)
                if callable(shot):
                    shot(name=str(path))
                    saved = True
            except Exception:
                saved = False
        if not saved:
            path.write_text("snapshot placeholder (headless/fallback)", encoding="utf-8")
        self._last_snapshot_path = path
        self._action_hint = f"snapshot {path.name}"
        self._refresh_hud()
        return path

    def reset_camera(self) -> None:
        if self.editor_camera is None:
            return
        self.editor_camera.rotation = self._CAMERA_ROTATION
        self.editor_camera.position = self._CAMERA_POSITION
        camera.fov = self._CAMERA_FOV

    def _build_callbacks(self) -> HudCallbacks:
        return HudCallbacks(
            on_step=self.step,
            on_run_pause=self.toggle_run,
            on_reset=self.reset,
            on_toggle_roads=lambda v: self.set_view(roads=v),
            on_toggle_risk=lambda v: self.set_view(risk=v),
            on_toggle_coverage=lambda v: self.set_view(coverage=v),
            on_toggle_population=lambda v: self.set_view(population=v),
            on_toggle_clusters=lambda v: self.set_view(clusters=v),
            on_toggle_ga_redundancy=lambda v: self.set_view(ga_redundancy=v),
            on_toggle_astar_trace=lambda v: self.set_view(astar_trace=v),
            on_toggle_csp_diagnostics=lambda v: self.set_view(csp_diagnostics=v),
            on_camera_reset=self.reset_camera,
            on_speed_changed=self.set_speed_multiplier,
            on_run_to_end=self.run_to_end,
            on_snapshot=self.snapshot,
            on_cycle_max_steps=self.cycle_max_steps,
            on_randomize_layout=lambda: self._randomize_seed_fields("layout", ["layout"]),
            on_randomize_roads=lambda: self._randomize_seed_fields("roads", ["road_ga"]),
            on_randomize_ambulance=lambda: self._randomize_seed_fields("ambulance", ["sa"]),
            on_randomize_mission=lambda: self._randomize_seed_fields("mission", ["mission"]),
            on_randomize_flood=lambda: self._randomize_seed_fields("flood", ["flood"]),
            on_randomize_ml=lambda: self._randomize_seed_fields(
                "ml", ["kmeans", "risk_classifier"]
            ),
            on_randomize_all=lambda: self._randomize_seed_fields(
                "all",
                ["layout", "kmeans", "risk_classifier", "road_ga", "sa", "mission", "flood"],
            ),
        )

    def cycle_max_steps(self) -> None:
        presets = self._MAX_STEPS_PRESETS
        try:
            idx = presets.index(self.max_ticks)
            self.max_ticks = presets[(idx + 1) % len(presets)]
        except ValueError:
            self.max_ticks = presets[0]
        if self.hud is not None:
            self.hud.set_max_steps(self.max_ticks)
        self._action_hint = f"max steps {self.max_ticks}"
        self._refresh_hud()

    def _next_seed_value(self) -> int:
        # Keep values in a compact, human-readable range for HUD/event logs.
        return self._seed_rng.randint(1, 999_999)

    def _seed_summary_text(self) -> str:
        s = self.seeds
        return (
            f"L={s.layout} R={s.road_ga} A={s.sa} "
            f"M={s.mission} F={s.flood} ML={s.kmeans}/{s.risk_classifier}"
        )

    def _randomize_seed_fields(self, group: str, fields: list[str]) -> None:
        updates = {name: self._next_seed_value() for name in fields}
        self.seeds = replace(self.seeds, **updates)
        changed = ", ".join(f"{k}={v}" for k, v in sorted(updates.items()))
        self._pending_seed_log_lines.append(f"seed update [{group}]: {changed}")
        self._action_hint = f"randomized {group}"
        self.reset()

    def set_view(self, **toggles: bool) -> None:
        scene_kwargs = {}
        if "roads" in toggles:
            scene_kwargs["show_roads"] = bool(toggles["roads"])
        if "risk" in toggles:
            scene_kwargs["show_risk"] = bool(toggles["risk"])
        if "coverage" in toggles:
            scene_kwargs["show_coverage"] = bool(toggles["coverage"])
        if "population" in toggles:
            scene_kwargs["show_population"] = bool(toggles["population"])
        if "clusters" in toggles:
            scene_kwargs["show_clusters"] = bool(toggles["clusters"])
        if "ga_redundancy" in toggles:
            scene_kwargs["show_ga_redundancy"] = bool(toggles["ga_redundancy"])
        if "astar_trace" in toggles:
            scene_kwargs["show_astar_trace"] = bool(toggles["astar_trace"])
        if "csp_diagnostics" in toggles:
            scene_kwargs["show_csp_diagnostics"] = bool(toggles["csp_diagnostics"])
        self.scene.set_view(**scene_kwargs)
        if self.hud is not None:
            for key, value in toggles.items():
                self.hud.set_toggle_state(key, bool(value))
            self.hud.update_legend_hints(
                show_risk=self.scene.view_state.show_risk,
                show_coverage=self.scene.view_state.show_coverage,
                show_population=self.scene.view_state.show_population,
                show_roads=self.scene.view_state.show_roads,
                show_clusters=self.scene.view_state.show_clusters,
                show_ga_redundancy=self.scene.view_state.show_ga_redundancy,
                show_astar_trace=self.scene.view_state.show_astar_trace,
                show_csp_diagnostics=self.scene.view_state.show_csp_diagnostics,
                csp_diag_count=self.scene.csp_diagnostic_count(),
            )

    def _run_setup(self) -> None:
        if self.hud is not None:
            self.hud.set_loading(True, "Building city and initializing modules...")
        pipeline = SetupPipeline(seeds=self.seeds)
        self.artifacts = pipeline.run()
        for line in self._pending_seed_log_lines:
            self.artifacts.runner.state.event_log.append(line)
        self._pending_seed_log_lines.clear()
        self.scene.build(self.artifacts)
        self.scene.set_on_cell_selected(self._on_cell_selected)
        self._last_event_count = 0
        self._action_hint = "ready"
        self._selected_cell = None
        start_world = self.scene.team_world_position(self.artifacts.runner.state.team_pos)
        self.scene.set_team_visual_position(start_world)
        self._last_ambulances = self.artifacts.sa_optimizer.current_positions
        self.scene.set_view(
            show_roads=True,
            show_risk=True,
            show_coverage=False,
            show_population=False,
            show_clusters=False,
            show_ga_redundancy=False,
            show_astar_trace=False,
            show_csp_diagnostics=False,
        )
        if self.hud is not None:
            self.hud.set_toggle_state("roads", True)
            self.hud.set_toggle_state("risk", True)
            self.hud.set_toggle_state("coverage", False)
            self.hud.set_toggle_state("population", False)
            self.hud.set_toggle_state("clusters", False)
            self.hud.set_toggle_state("ga_redundancy", False)
            self.hud.set_toggle_state("astar_trace", False)
            self.hud.set_toggle_state("csp_diagnostics", False)
            self.hud.update_legend_hints(
                show_risk=True,
                show_coverage=False,
                show_population=False,
                show_roads=True,
                show_clusters=False,
                show_ga_redundancy=False,
                show_astar_trace=False,
                show_csp_diagnostics=False,
                csp_diag_count=self.scene.csp_diagnostic_count(),
            )
            self.hud.clear_inspector()
            self.hud.hide_summary()
            self.hud.set_speed(self._speed_multiplier)
            self.hud.set_max_steps(self.max_ticks)
            self.hud.update_tooltip(self.scene.hover_tooltip_text())
            self.hud.update_seed_status(self._seed_summary_text())
            self.hud.update_layout_status(self.artifacts.layout_result)
            self.hud.set_loading(False)
            self._refresh_hud()

    def _refresh_hud(self) -> None:
        if self.hud is None or self.artifacts is None:
            return
        runner = self.artifacts.runner
        total = len(runner.state.civilians_ordered)
        reached = runner.state.target_index
        current_target = runner.state.current_target
        target_idx = None if current_target is None else (runner.state.target_index + 1)
        self.hud.update_status(
            self._tick,
            self.max_ticks,
            total,
            reached,
            self._running,
            current_target=current_target,
            current_target_index=target_idx,
            action_hint=self._action_hint,
        )
        mode = "live" if self._running else "idle"
        if runner.state.completed or self._tick >= self.max_ticks:
            mode = "summary"
        elif self.scene.view_state.show_risk and not self._running:
            mode = "heatmap"
        self.hud.set_mode(mode)
        self.hud.refresh_log(runner.state.event_log)
        if self._selected_cell is not None:
            self._push_cell_snapshot(self._selected_cell)
        self.hud.update_tooltip(self.scene.hover_tooltip_text())
        if self._tick >= self.max_ticks or runner.state.completed:
            self.hud.show_summary(self.build_summary())

    def _on_frame(self) -> None:
        dt = max(0.0, getattr(ursina_time, "dt", 0.0))
        self._advance_team_interpolation()
        self.scene.animate(dt)
        if self.hud is not None:
            self.hud.animate(dt)
        if not self._running or self.artifacts is None:
            return
        if self._tick >= self.max_ticks or self.artifacts.runner.state.completed:
            self._running = False
            self._refresh_hud()
            return
        self._accumulator += getattr(ursina_time, "dt", 0.0)
        if self._accumulator >= self.tick_interval:
            self._accumulator = 0.0
            self.step()

    def _on_input(self, key: str) -> None:
        if key == "s":
            self.step()
        elif key == "space":
            self.toggle_run()
        elif key == "r":
            self.reset()
        elif key == "1":
            self.set_view(roads=not self.scene.view_state.show_roads)
        elif key == "2":
            self.set_view(risk=not self.scene.view_state.show_risk)
        elif key == "3":
            self.set_view(coverage=not self.scene.view_state.show_coverage)
        elif key == "4":
            self.set_view(population=not self.scene.view_state.show_population)
        elif key == "t":
            self.reset_camera()
        elif key == "5":
            self.set_view(clusters=not self.scene.view_state.show_clusters)
        elif key == "6":
            self.set_view(ga_redundancy=not self.scene.view_state.show_ga_redundancy)
        elif key == "7":
            self.set_view(astar_trace=not self.scene.view_state.show_astar_trace)
        elif key == "8":
            self.set_view(csp_diagnostics=not self.scene.view_state.show_csp_diagnostics)
        elif key == "9":
            self.set_speed_multiplier(0.5)
        elif key == "0":
            self.set_speed_multiplier(2.0)
        elif key == "e":
            self.run_to_end()
        elif key == "p":
            self.snapshot()
        elif key == "escape":
            try:
                application.quit()
            except Exception:
                pass

    def _disable_debug_overlays(self) -> None:
        for attr in ("fps_counter", "entity_counter", "collider_counter"):
            counter = getattr(window, attr, None)
            if counter is None:
                continue
            try:
                counter.enabled = False
            except Exception:
                pass

    def _highlight_new_events(self) -> None:
        if self.artifacts is None:
            return
        events = self.artifacts.runner.state.event_log
        if self._last_event_count < 0:
            self._last_event_count = 0
        for line in events[self._last_event_count :]:
            self.scene.highlight_event_line(line)
        self._last_event_count = len(events)

    def _set_action_hint_from_latest_event(self) -> None:
        if self.artifacts is None:
            self._action_hint = "ready"
            return
        events = self.artifacts.runner.state.event_log
        if not events:
            self._action_hint = "ready"
            return
        line = events[-1]
        if "replan failed" in line:
            self._action_hint = "waiting"
        elif "A* replan" in line:
            self._action_hint = "replanning"
        elif "move:" in line:
            self._action_hint = "moving"
        elif "reached civilian" in line:
            self._action_hint = "rescued"
        else:
            self._action_hint = "active"

    def _begin_team_interpolation(self, from_cell, to_cell) -> None:
        if self.artifacts is None:
            return
        from_world = self.scene.team_world_position(from_cell)
        to_world = self.scene.team_world_position(to_cell)
        self._interp_from = from_world
        self._interp_to = to_world
        self._interp_t = 0.0
        moving = not (
            isclose(from_world[0], to_world[0])
            and isclose(from_world[2], to_world[2])
        )
        self._interp_active = bool(moving and not self.headless)
        if not self._interp_active:
            self.scene.set_team_visual_position(to_world)

    def _advance_team_interpolation(self) -> None:
        if not self._interp_active:
            return
        dt = max(0.0, getattr(ursina_time, "dt", 0.0))
        duration = max(0.05, min(self.tick_interval, self._interp_duration))
        self._interp_t = min(1.0, self._interp_t + dt / duration)
        x = self._interp_from[0] + (self._interp_to[0] - self._interp_from[0]) * self._interp_t
        y = self._interp_from[1] + (self._interp_to[1] - self._interp_from[1]) * self._interp_t
        z = self._interp_from[2] + (self._interp_to[2] - self._interp_from[2]) * self._interp_t
        self.scene.set_team_visual_position((x, y, z))
        if self._interp_t >= 1.0:
            self._interp_active = False

    def _on_cell_selected(self, cell) -> None:
        self._selected_cell = cell
        self._push_cell_snapshot(cell)
        if self.hud is not None:
            self.hud.update_tooltip(self.scene.hover_tooltip_text())

    def _push_cell_snapshot(self, cell) -> None:
        if self.hud is None:
            return
        snapshot = self.scene.cell_snapshot(cell)
        if snapshot is None:
            self.hud.clear_inspector()
            return
        self.hud.update_inspector(snapshot)

    def shutdown(self) -> None:
        try:
            self.scene.destroy()
        except Exception:
            pass
        if self.hud is not None:
            try:
                self.hud.destroy()
            except Exception:
                pass
        self.app = None
