"""HUD layer for CityMind.

Layout uses normalized fractions of the viewport (0..1, top-left origin) so
panels and controls always fit the actual window size. The 3D scene is left
fully visible in the center region between the sidebar, right panel, top bar,
and bottom event log.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from ursina import Button, Entity, Text, camera, color, destroy, window


TOP_H = 0.10
SIDE_W = 0.20
RIGHT_W = 0.24
LOG_H = 0.20


def _rgb01(r: float, g: float, b: float):
    """Build a Color from normalized 0..1 channels (Ursina's native range)."""
    return color.rgb(float(r), float(g), float(b))


def _rgba01(r: float, g: float, b: float, a: float):
    """Build a Color from normalized 0..1 channels and alpha."""
    return color.rgba(float(r), float(g), float(b), float(a))


@dataclass
class HudCallbacks:
    on_step: Callable[[], None]
    on_run_pause: Callable[[], None]
    on_reset: Callable[[], None]
    on_toggle_roads: Callable[[bool], None]
    on_toggle_risk: Callable[[bool], None]
    on_toggle_coverage: Callable[[bool], None]
    on_toggle_population: Callable[[bool], None]
    on_toggle_clusters: Callable[[bool], None]
    on_toggle_ga_redundancy: Callable[[bool], None]
    on_toggle_astar_trace: Callable[[bool], None]
    on_toggle_csp_diagnostics: Callable[[bool], None]
    on_camera_reset: Callable[[], None]
    on_speed_changed: Callable[[float], None]
    on_run_to_end: Callable[[], None]
    on_snapshot: Callable[[], None]
    on_cycle_max_steps: Callable[[], None]
    on_randomize_layout: Callable[[], None]
    on_randomize_roads: Callable[[], None]
    on_randomize_ambulance: Callable[[], None]
    on_randomize_mission: Callable[[], None]
    on_randomize_flood: Callable[[], None]
    on_randomize_ml: Callable[[], None]
    on_randomize_all: Callable[[], None]


_TOKENS = {
    "bg": _rgb01(0.05, 0.07, 0.11),
    "surface": _rgba01(0.10, 0.13, 0.19, 0.97),
    "surface_alt": _rgba01(0.07, 0.10, 0.15, 0.97),
    "border": _rgb01(0.22, 0.27, 0.36),
    "accent": _rgb01(0.23, 0.51, 0.96),
    "running": _rgb01(0.06, 0.72, 0.50),
    "critical": _rgb01(0.93, 0.27, 0.27),
    "warn": _rgb01(0.96, 0.62, 0.04),
    "text": _rgb01(0.97, 0.98, 0.99),
    "muted": _rgb01(0.62, 0.69, 0.78),
    "btn_idle": _rgb01(0.18, 0.22, 0.30),
    "btn_active": _rgb01(0.27, 0.35, 0.50),
}

_LOG_COLORS: List[tuple] = [
    ("warning", _TOKENS["warn"]),
    ("flood", _TOKENS["critical"]),
    ("A* replan", _TOKENS["warn"]),
    ("replan failed", _rgb01(0.95, 0.45, 0.25)),
    ("SA re-eval", _rgb01(0.45, 0.85, 0.95)),
    ("reached civilian", _rgb01(0.45, 0.95, 0.45)),
    ("mission complete", _rgb01(0.45, 0.95, 0.45)),
]

_LAYOUT_RULE_SHORT_LABELS: Dict[str, str] = {
    "industrial_adjacent_to_protected": "IndAdj",
    "residential_hospital_reach": "Res>3h",
    "power_industrial_reach": "Pwr>2h",
    "quota_mismatch": "Quota",
}


_LEGEND_ENTRIES = [
    ("Hospital", _rgb01(0.95, 0.95, 0.95)),
    ("School", _rgb01(0.95, 0.85, 0.25)),
    ("Industrial", _rgb01(0.55, 0.55, 0.62)),
    ("Power Plant", _rgb01(0.95, 0.55, 0.15)),
    ("Ambulance Depot", _rgb01(0.85, 0.20, 0.25)),
    ("Residential", _rgb01(0.30, 0.55, 0.70)),
    ("Police Post", _rgb01(0.20, 0.40, 1.00)),
    ("Team", _rgb01(0.95, 0.20, 0.95)),
]

_TOGGLES: List[tuple] = [
    ("roads", "Roads"),
    ("risk", "Risk Heatmap"),
    ("coverage", "Coverage"),
    ("population", "Population"),
    ("clusters", "Clusters"),
    ("ga_redundancy", "GA Paths"),
    ("astar_trace", "A* Trace"),
    ("csp_diagnostics", "CSP Diag"),
]


class Hud:
    """All on-screen 2D HUD elements for CityMind."""

    def __init__(self, callbacks: HudCallbacks, log_capacity: int = 8) -> None:
        self.callbacks = callbacks
        self.log_capacity = log_capacity
        self._entities: List[Entity] = []
        self._toggle_buttons: Dict[str, Button] = {}
        self._toggle_state: Dict[str, bool] = {}
        self._run_pause_button: Optional[Button] = None
        self._max_steps_button: Optional[Button] = None
        self._status_text: Optional[Text] = None
        self._layout_status_text: Optional[Text] = None
        self._seed_status_text: Optional[Text] = None
        self._seed_summary: str = ""
        self._inspect_title: Optional[Text] = None
        self._inspect_lines: List[Text] = []
        self._tooltip_text: Optional[Text] = None
        self._legend_hint_lines: List[Text] = []
        self._log_lines: List[Text] = []
        self._loading_text: Optional[Text] = None
        self._summary_title: Optional[Text] = None
        self._summary_lines: List[Text] = []
        self._summary_buttons: List[Button] = []
        self._summary_panel: Optional[Entity] = None
        self._summary_visible: bool = False
        self._speed_buttons: Dict[float, Button] = {}
        self._status_chip: Optional[Text] = None
        self._mode: str = "idle"
        self._pulse_t: float = 0.0
        self._last_log_tail: List[str] = []
        self._log_fade: Dict[int, float] = {}

    def build(self) -> None:
        self.destroy()
        self._build_shell()
        self._build_topbar_region()
        self._build_sidebar_region()
        self._build_rightpanel_region()
        self._build_event_log_region()
        self._build_summary_panel()
        self.clear_inspector()

    def destroy(self) -> None:
        for e in self._entities:
            destroy(e)
        self._entities = []
        self._status_text = None
        self._status_chip = None
        self._layout_status_text = None
        self._seed_status_text = None
        self._seed_summary = ""
        self._inspect_title = None
        self._inspect_lines = []
        self._tooltip_text = None
        self._legend_hint_lines = []
        self._log_lines = []
        self._loading_text = None
        self._summary_title = None
        self._summary_lines = []
        self._summary_buttons = []
        self._summary_panel = None
        self._summary_visible = False
        self._speed_buttons = {}
        self._toggle_buttons = {}
        self._toggle_state = {}
        self._run_pause_button = None
        self._max_steps_button = None
        self._last_log_tail = []
        self._log_fade = {}

    def update_status(
        self,
        tick: int,
        max_steps: int,
        total_civilians: int,
        reached: int,
        running: bool,
        current_target: Optional[tuple] = None,
        current_target_index: Optional[int] = None,
        action_hint: str = "",
    ) -> None:
        if self._status_text is None:
            return
        if current_target is None:
            target_text = "Target -"
        else:
            prefix = f"#{current_target_index} " if current_target_index is not None else ""
            target_text = f"Target {prefix}{current_target}"
        self._status_text.text = (
            f"Step {tick:02d}/{max_steps}    Civilians {reached}/{total_civilians}    "
            f"{target_text}    {action_hint or 'ready'}"
        )
        if self._status_chip is not None:
            if running:
                self._status_chip.text = "RUNNING"
                self._status_chip.color = _TOKENS["running"]
                self._mode = "live"
            elif self._summary_visible:
                self._status_chip.text = "PAUSED"
                self._status_chip.color = _TOKENS["warn"]
                self._mode = "summary"
            else:
                self._status_chip.text = "IDLE"
                self._status_chip.color = _TOKENS["accent"]
                self._mode = "idle"
        if self._run_pause_button is not None:
            self._run_pause_button.text = "Pause" if running else "Run"

    def set_mode(self, mode: str) -> None:
        self._mode = mode

    def animate(self, dt: float) -> None:
        if self._status_chip is not None and self._mode == "live":
            self._pulse_t += max(0.0, dt)
            alpha = 0.65 + 0.35 * (0.5 + 0.5 * math.sin(self._pulse_t * 4.19))
            base = _TOKENS["running"]
            self._status_chip.color = color.rgba(base.r, base.g, base.b, alpha)
        for idx, fade in list(self._log_fade.items()):
            line = self._log_lines[idx]
            fade = min(1.0, fade + max(0.0, dt) / 0.2)
            self._log_fade[idx] = fade
            c = line.color
            line.color = color.rgba(c.r, c.g, c.b, fade)
            if fade >= 1.0:
                self._log_fade.pop(idx, None)

    def set_toggle_state(self, name: str, value: bool) -> None:
        self._toggle_state[name] = bool(value)
        button = self._toggle_buttons.get(name)
        if button is None:
            return
        button.color = _TOKENS["accent"] if value else _TOKENS["btn_idle"]
        button.text = self._toggle_label(name, value)
        if button.text_entity is not None:
            button.text_entity.color = _TOKENS["text"]
        if name == "risk" and value:
            self._mode = "heatmap"

    def refresh_log(self, events: Sequence[str]) -> None:
        tail = list(events[-self.log_capacity:])
        for idx, line in enumerate(self._log_lines):
            if idx < len(tail):
                content = tail[len(tail) - 1 - idx]
                line.text = self._truncate(content, max_chars=72)
                line.color = self._color_for_log(content)
                line.enabled = True
                if idx < len(self._last_log_tail) and self._last_log_tail[idx] != line.text:
                    line.color = color.rgba(line.color.r, line.color.g, line.color.b, 0.2)
                    self._log_fade[idx] = 0.2
            else:
                line.text = ""
                line.enabled = False
        self._last_log_tail = [self._truncate(x, max_chars=72) for x in reversed(tail)]

    def update_inspector(self, snapshot: dict) -> None:
        if self._inspect_title is not None:
            self._inspect_title.text = "Selected Cell"
        rows = [
            f"Cell: {snapshot.get('cell')}",
            f"Type: {snapshot.get('location_type')}",
            f"Risk: {snapshot.get('risk_level')} ({snapshot.get('risk_mult')})",
            f"Population: {snapshot.get('population')}",
            f"Cluster: {snapshot.get('cluster_id')}",
            f"Accessible: {snapshot.get('accessible')}",
            f"Police Post: {snapshot.get('police_post')}",
            f"Team Here: {snapshot.get('is_team_here')}",
            f"Current Target: {snapshot.get('is_current_target')}",
            f"On Path: {snapshot.get('is_on_current_path')}",
            f"Open/Blocked: {snapshot.get('open_neighbors')}/{snapshot.get('blocked_neighbors')}",
            self._truncate(snapshot.get("explainability", ""), max_chars=42),
        ]
        for idx, line in enumerate(self._inspect_lines):
            if idx < len(rows):
                line.text = self._truncate(rows[idx], max_chars=44)
                line.enabled = True
            else:
                line.text = ""
                line.enabled = False

    def clear_inspector(self) -> None:
        if self._inspect_title is not None:
            self._inspect_title.text = "No Cell Selected"
        for idx, line in enumerate(self._inspect_lines):
            if idx == 0:
                line.text = "Click a cell to inspect"
                line.enabled = True
            else:
                line.text = ""
                line.enabled = False

    def update_tooltip(self, text: str) -> None:
        if self._tooltip_text is None:
            return
        self._tooltip_text.text = self._truncate(text, max_chars=72)

    def set_loading(self, active: bool, stage: str = "") -> None:
        if self._loading_text is None:
            return
        self._loading_text.enabled = bool(active)
        if active:
            self._loading_text.text = f"CityMind Loading...\n{stage or 'Preparing simulation'}"

    def set_speed(self, speed: float) -> None:
        for value, button in self._speed_buttons.items():
            active = abs(value - speed) < 1e-9
            button.color = _TOKENS["accent"] if active else _TOKENS["btn_idle"]
            if button.text_entity is not None:
                button.text_entity.color = _TOKENS["text"]

    def set_max_steps(self, max_steps: int) -> None:
        if self._max_steps_button is None:
            return
        self._max_steps_button.text = f"Max {max_steps}"
        self._style_button_text(self._max_steps_button, scale=1.0)

    def show_summary(self, summary: dict) -> None:
        self._summary_visible = True
        self._mode = "summary"
        if self._summary_panel is not None:
            self._summary_panel.enabled = True
        if self._summary_title is not None:
            self._summary_title.enabled = True
        rows = [
            f"Steps: {summary.get('steps', 0)}",
            f"Civilians Rescued: {summary.get('reached', 0)}/{summary.get('total', 0)}",
            f"Replans Failed: {summary.get('replan_failed', 0)}",
            f"Repairs: {summary.get('repairs', 0)}",
            f"SA Re-eval: {summary.get('sa_reeval', 0)}",
        ]
        for idx, line in enumerate(self._summary_lines):
            if idx < len(rows):
                line.text = rows[idx]
                line.enabled = True
            else:
                line.text = ""
                line.enabled = False
        for b in self._summary_buttons:
            b.enabled = True

    def hide_summary(self) -> None:
        self._summary_visible = False
        if self._summary_panel is not None:
            self._summary_panel.enabled = False
        if self._summary_title is not None:
            self._summary_title.enabled = False
        for line in self._summary_lines:
            line.text = ""
            line.enabled = False
        for b in self._summary_buttons:
            b.enabled = False

    def summary_visible(self) -> bool:
        return self._summary_visible

    def update_layout_status(self, layout_result) -> None:
        """Surface the CSP layout outcome in the top bar.

        Shows `Layout: OK` when CSP succeeded; otherwise `Layout: N viol [...]`
        with a compact summary of the per-rule breakdown. We avoid Unicode glyphs
        because Ursina's default font does not render them.
        """
        if self._layout_status_text is None:
            return
        success = bool(getattr(layout_result, "success", False))
        violations = int(getattr(layout_result, "violations", 0))
        breakdown = getattr(layout_result, "violation_breakdown", {}) or {}
        if success and not breakdown:
            self._layout_status_text.text = "Layout: OK  |  10x10 grid  |  Mission control"
            self._layout_status_text.color = _TOKENS["muted"]
            if self._seed_status_text is not None:
                self._seed_status_text.text = self._truncate(
                    f"Seeds: {self._seed_summary}" if self._seed_summary else "Seeds: (not set)",
                    max_chars=72,
                )
            return
        # Compact, deterministic ordering: largest violations first.
        ordered = sorted(breakdown.items(), key=lambda kv: (-kv[1], kv[0]))
        parts = []
        for key, count in ordered[:3]:
            short = _LAYOUT_RULE_SHORT_LABELS.get(key, key)
            parts.append(f"{short}:{count}")
        suffix = ", ".join(parts) if parts else ""
        text = f"Layout: {violations} viol [{suffix}]" if suffix else f"Layout: {violations} viol"
        self._layout_status_text.text = self._truncate(text, max_chars=72)
        self._layout_status_text.color = _TOKENS["warn"] if violations > 0 else _TOKENS["muted"]
        if self._seed_status_text is not None:
            self._seed_status_text.text = self._truncate(
                f"Seeds: {self._seed_summary}" if self._seed_summary else "Seeds: (not set)",
                max_chars=72,
            )

    def update_seed_status(self, seed_summary: str) -> None:
        self._seed_summary = seed_summary.strip()
        if self._seed_status_text is not None:
            self._seed_status_text.text = self._truncate(
                f"Seeds: {self._seed_summary}" if self._seed_summary else "Seeds: (not set)",
                max_chars=72,
            )

    def update_legend_hints(
        self,
        *,
        show_risk: bool,
        show_coverage: bool,
        show_population: bool,
        show_roads: bool,
        show_clusters: bool = False,
        show_ga_redundancy: bool = False,
        show_astar_trace: bool = False,
        show_csp_diagnostics: bool = False,
        csp_diag_count: int = 0,
    ) -> None:
        hints: List[str] = []
        if show_risk:
            hints.append("Risk: Low/Medium/High heatmap")
        if show_coverage:
            hints.append("Coverage: nearest ambulance regions")
        if show_population:
            hints.append("Population: taller + deeper blue")
        if show_roads:
            hints.append("Roads: red blocked, green discount")
        if show_clusters:
            hints.append("Clusters: deterministic K-Means colors")
        if show_ga_redundancy:
            hints.append("GA: A/B/shared redundancy edges")
        if show_astar_trace:
            hints.append("A*: last replans trace")
        if show_csp_diagnostics:
            hints.append(f"CSP diagnostics: {csp_diag_count} cells")
        if not hints:
            hints.append("No overlays active")
        for idx, line in enumerate(self._legend_hint_lines):
            if idx < len(hints):
                line.text = self._truncate(hints[idx], max_chars=38)
                line.enabled = True
            else:
                line.text = ""
                line.enabled = False

    def _build_shell(self) -> None:
        # Top bar
        self._add_panel(0.0, 0.0, 1.0, TOP_H, col=_TOKENS["surface"], z=0.10)
        # Left sidebar (below top bar)
        self._add_panel(0.0, TOP_H, SIDE_W, 1.0 - TOP_H, col=_TOKENS["surface"], z=0.10)
        # Right panel (below top bar)
        self._add_panel(1.0 - RIGHT_W, TOP_H, RIGHT_W, 1.0 - TOP_H, col=_TOKENS["surface"], z=0.10)
        # Bottom event log strip (between sidebar and right panel)
        self._add_panel(
            SIDE_W,
            1.0 - LOG_H,
            1.0 - SIDE_W - RIGHT_W,
            LOG_H,
            col=_TOKENS["surface_alt"],
            z=0.10,
        )
        # Subtle borders
        self._add_panel(0.0, TOP_H, 1.0, 0.0025, col=_TOKENS["border"], z=0.05)
        self._add_panel(SIDE_W, TOP_H, 0.0025, 1.0 - TOP_H, col=_TOKENS["border"], z=0.05)
        self._add_panel(1.0 - RIGHT_W, TOP_H, 0.0025, 1.0 - TOP_H, col=_TOKENS["border"], z=0.05)
        self._add_panel(
            SIDE_W,
            1.0 - LOG_H,
            1.0 - SIDE_W - RIGHT_W,
            0.0025,
            col=_TOKENS["border"],
            z=0.05,
        )

    def _build_topbar_region(self) -> None:
        self._entities.append(Text(
            "CityMind",
            parent=camera.ui,
            position=self._fpos(0.012, TOP_H * 0.22),
            scale=1.45,
            color=_TOKENS["text"],
        ))
        self._layout_status_text = Text(
            "Layout: pending",
            parent=camera.ui,
            position=self._fpos(0.012, TOP_H * 0.56),
            scale=0.78,
            color=_TOKENS["muted"],
        )
        self._entities.append(self._layout_status_text)
        self._seed_status_text = Text(
            "Seeds: (not set)",
            parent=camera.ui,
            position=self._fpos(0.012, TOP_H * 0.82),
            scale=0.66,
            color=_TOKENS["muted"],
        )
        self._entities.append(self._seed_status_text)
        self._status_text = Text(
            "",
            parent=camera.ui,
            position=self._fpos(SIDE_W + 0.03, TOP_H * 0.70),
            scale=0.82,
            color=_TOKENS["muted"],
        )
        self._entities.append(self._status_text)
        self._status_chip = Text(
            "IDLE",
            parent=camera.ui,
            position=self._fpos(1.0 - RIGHT_W - 0.06, TOP_H * 0.50),
            scale=1.15,
            color=_TOKENS["accent"],
        )
        self._entities.append(self._status_chip)
        self._loading_text = Text(
            "",
            parent=camera.ui,
            position=self._fpos(0.5, 0.5),
            origin=(0, 0),
            scale=1.6,
            color=_TOKENS["warn"],
            background=True,
        )
        self._loading_text.enabled = False
        self._entities.append(self._loading_text)

    def _build_sidebar_region(self) -> None:
        x0 = 0.010
        width = SIDE_W - 0.020
        cur_y = TOP_H + 0.014

        # LAYERS section
        self._entities.append(Text(
            "LAYERS",
            parent=camera.ui,
            position=self._fpos(x0, cur_y),
            scale=0.7,
            color=_TOKENS["muted"],
        ))
        cur_y += 0.020

        toggle_h = 0.034
        toggle_gap = 0.003
        for key, _label in _TOGGLES:
            b = Button(
                text=self._toggle_label(key, False),
                parent=camera.ui,
                position=self._fpos(x0, cur_y),
                scale=self._fscale(width, toggle_h),
                origin=(-0.5, 0.5),
                color=_TOKENS["btn_idle"],
                on_click=self._make_toggle_handler(key),
            )
            self._style_button_text(b, scale=0.9)
            self._toggle_buttons[key] = b
            self._toggle_state[key] = False
            self._entities.append(b)
            cur_y += toggle_h + toggle_gap

        cur_y += 0.012

        # CONTROLS section
        self._entities.append(Text(
            "CONTROLS",
            parent=camera.ui,
            position=self._fpos(x0, cur_y),
            scale=0.7,
            color=_TOKENS["muted"],
        ))
        cur_y += 0.020

        action_h = 0.040
        action_gap = 0.003
        for label, cb, col in (
            ("Step", self.callbacks.on_step, _TOKENS["accent"]),
            ("Run", self.callbacks.on_run_pause, _TOKENS["running"]),
            ("Reset", self.callbacks.on_reset, _TOKENS["critical"]),
            ("Max 20", self.callbacks.on_cycle_max_steps, _rgb01(0.36, 0.42, 0.58)),
            ("Top View", self.callbacks.on_camera_reset, _rgb01(0.40, 0.46, 0.74)),
            ("Run End", self.callbacks.on_run_to_end, _rgb01(0.32, 0.42, 0.78)),
            ("Snapshot", self.callbacks.on_snapshot, _rgb01(0.30, 0.36, 0.46)),
        ):
            b = Button(
                text=label,
                parent=camera.ui,
                position=self._fpos(x0, cur_y),
                scale=self._fscale(width, action_h),
                origin=(-0.5, 0.5),
                color=col,
                on_click=cb,
            )
            self._style_button_text(b, scale=1.0)
            self._entities.append(b)
            if label == "Run":
                self._run_pause_button = b
            if label.startswith("Max "):
                self._max_steps_button = b
            cur_y += action_h + action_gap

        cur_y += 0.012

        # SPEED section
        self._entities.append(Text(
            "SPEED",
            parent=camera.ui,
            position=self._fpos(x0, cur_y),
            scale=0.7,
            color=_TOKENS["muted"],
        ))
        cur_y += 0.020

        speed_w = (width - 0.006 * 3) / 4
        speed_h = 0.030
        for idx, s in enumerate((0.5, 1.0, 2.0, 4.0)):
            b = Button(
                text=f"{s:g}x",
                parent=camera.ui,
                position=self._fpos(x0 + idx * (speed_w + 0.006), cur_y),
                scale=self._fscale(speed_w, speed_h),
                origin=(-0.5, 0.5),
                color=_TOKENS["btn_idle"],
                on_click=lambda s=s: self.callbacks.on_speed_changed(s),
            )
            self._style_button_text(b, scale=0.85)
            self._speed_buttons[s] = b
            self._entities.append(b)

        cur_y += speed_h + 0.010

        # Keys hint at bottom of sidebar
        keys_y = 1.0 - 0.024
        self._entities.append(Text(
            "S step  Space run  R reset",
            parent=camera.ui,
            position=self._fpos(x0, keys_y),
            scale=0.55,
            color=_TOKENS["muted"],
        ))
        self._entities.append(Text(
            "1-8 toggle layers  T top  E end",
            parent=camera.ui,
            position=self._fpos(x0, keys_y + 0.014),
            scale=0.55,
            color=_TOKENS["muted"],
        ))

    def _build_rightpanel_region(self) -> None:
        x0 = 1.0 - RIGHT_W + 0.014
        width = RIGHT_W - 0.028
        cur_y = TOP_H + 0.014

        # INSPECTOR
        self._entities.append(Text(
            "INSPECTOR",
            parent=camera.ui,
            position=self._fpos(x0, cur_y),
            scale=0.78,
            color=_TOKENS["muted"],
        ))
        cur_y += 0.024

        self._inspect_title = Text(
            "No Cell Selected",
            parent=camera.ui,
            position=self._fpos(x0, cur_y),
            scale=1.05,
            color=_TOKENS["text"],
        )
        self._entities.append(self._inspect_title)
        cur_y += 0.026

        # Slightly fewer visible rows to make room for randomize controls.
        line_h = 0.020
        for i in range(9):
            line = Text(
                "",
                parent=camera.ui,
                position=self._fpos(x0, cur_y + i * line_h),
                scale=0.70,
                color=_TOKENS["muted"],
            )
            self._inspect_lines.append(line)
            self._entities.append(line)
        cur_y += 9 * line_h + 0.014

        # LEGEND
        self._entities.append(Text(
            "LEGEND",
            parent=camera.ui,
            position=self._fpos(x0, cur_y),
            scale=0.78,
            color=_TOKENS["muted"],
        ))
        cur_y += 0.022
        for idx, (label, sw) in enumerate(_LEGEND_ENTRIES):
            row_y = cur_y + idx * 0.020
            swatch = Entity(
                parent=camera.ui,
                model="quad",
                color=sw,
                position=self._fpos(x0, row_y + 0.004),
                scale=self._fscale(0.018, 0.018),
                origin=(-0.5, 0.5),
            )
            text = Text(
                label,
                parent=camera.ui,
                position=self._fpos(x0 + 0.026, row_y + 0.002),
                scale=0.68,
                color=_TOKENS["text"],
            )
            self._entities.append(swatch)
            self._entities.append(text)
        cur_y += len(_LEGEND_ENTRIES) * 0.020 + 0.012

        # ACTIVE OVERLAYS
        self._entities.append(Text(
            "ACTIVE OVERLAYS",
            parent=camera.ui,
            position=self._fpos(x0, cur_y),
            scale=0.78,
            color=_TOKENS["muted"],
        ))
        cur_y += 0.020
        for i in range(8):
            t = Text(
                "",
                parent=camera.ui,
                position=self._fpos(x0, cur_y + i * 0.016),
                scale=0.54,
                color=_rgb01(0.65, 0.85, 0.95),
            )
            self._legend_hint_lines.append(t)
            self._entities.append(t)
        self.update_legend_hints(
            show_risk=True,
            show_coverage=False,
            show_population=False,
            show_roads=True,
        )
        cur_y += 8 * 0.016 + 0.012

        # RANDOMIZE (moved from left sidebar to reduce crowding there)
        self._entities.append(Text(
            "RANDOMIZE",
            parent=camera.ui,
            position=self._fpos(x0, cur_y),
            scale=0.72,
            color=_TOKENS["muted"],
        ))
        cur_y += 0.018

        rnd_h = 0.022
        rnd_gap = 0.002
        for label, cb in (
            ("Rnd Layout", self.callbacks.on_randomize_layout),
            ("Rnd Roads", self.callbacks.on_randomize_roads),
            ("Rnd Amb", self.callbacks.on_randomize_ambulance),
            ("Rnd Mission", self.callbacks.on_randomize_mission),
            ("Rnd Flood", self.callbacks.on_randomize_flood),
            ("Rnd ML", self.callbacks.on_randomize_ml),
            ("Rnd All", self.callbacks.on_randomize_all),
        ):
            b = Button(
                text=label,
                parent=camera.ui,
                position=self._fpos(x0, cur_y),
                scale=self._fscale(width, rnd_h),
                origin=(-0.5, 0.5),
                color=_TOKENS["btn_idle"],
                on_click=cb,
            )
            self._style_button_text(b, scale=0.70)
            self._entities.append(b)
            cur_y += rnd_h + rnd_gap

    def _build_event_log_region(self) -> None:
        x0 = SIDE_W + 0.014
        y0 = 1.0 - LOG_H + 0.012
        log_w = 1.0 - SIDE_W - RIGHT_W - 0.028

        self._entities.append(Text(
            "EVENT LOG",
            parent=camera.ui,
            position=self._fpos(x0, y0),
            scale=0.85,
            color=_TOKENS["warn"],
        ))
        self._tooltip_text = Text(
            "Click a cell to inspect",
            parent=camera.ui,
            position=self._fpos(x0 + log_w * 0.32, y0),
            scale=0.65,
            color=_rgb01(0.75, 0.85, 0.95),
        )
        self._entities.append(self._tooltip_text)

        line_h = (LOG_H - 0.040) / max(1, self.log_capacity)
        for i in range(self.log_capacity):
            line = Text(
                "",
                parent=camera.ui,
                position=self._fpos(x0, y0 + 0.024 + i * line_h),
                scale=0.62,
                color=_TOKENS["text"],
            )
            self._log_lines.append(line)
            self._entities.append(line)

    def _build_summary_panel(self) -> None:
        modal_w = 0.50
        modal_h = 0.42
        x0 = 0.5 - modal_w / 2
        y0 = 0.5 - modal_h / 2
        self._summary_panel = self._add_panel(
            x0,
            y0,
            modal_w,
            modal_h,
            col=_rgba01(0.10, 0.13, 0.18, 0.96),
            z=-0.20,
        )
        self._summary_panel.enabled = False
        self._summary_title = Text(
            "Run Summary",
            parent=camera.ui,
            position=self._fpos(0.5, y0 + 0.05),
            origin=(0, 0),
            scale=1.7,
            color=_TOKENS["text"],
            z=-0.21,
        )
        self._summary_title.enabled = False
        self._entities.append(self._summary_title)
        for i in range(5):
            row = i % 3
            col_idx = i // 3
            t = Text(
                "",
                parent=camera.ui,
                position=self._fpos(x0 + 0.04 + col_idx * 0.22, y0 + 0.12 + row * 0.06),
                scale=0.95,
                color=_TOKENS["text"],
                z=-0.21,
            )
            t.enabled = False
            self._summary_lines.append(t)
            self._entities.append(t)
        export_btn = Button(
            text="Export Report",
            parent=camera.ui,
            position=self._fpos(x0 + 0.04, y0 + modal_h - 0.07),
            scale=self._fscale(0.20, 0.05),
            origin=(-0.5, 0.5),
            color=_TOKENS["accent"],
            on_click=self.callbacks.on_snapshot,
            z=-0.21,
        )
        self._style_button_text(export_btn, scale=0.85)
        restart_btn = Button(
            text="Restart Simulation",
            parent=camera.ui,
            position=self._fpos(x0 + modal_w - 0.24, y0 + modal_h - 0.07),
            scale=self._fscale(0.20, 0.05),
            origin=(-0.5, 0.5),
            color=_TOKENS["running"],
            on_click=self.callbacks.on_reset,
            z=-0.21,
        )
        self._style_button_text(restart_btn, scale=0.85)
        export_btn.enabled = False
        restart_btn.enabled = False
        self._entities.append(export_btn)
        self._entities.append(restart_btn)
        self._summary_buttons.extend([export_btn, restart_btn])

    def _toggle_label(self, key: str, active: bool, label: Optional[str] = None) -> str:
        names = {k: v for k, v in _TOGGLES}
        name = label or names.get(key, key.title())
        prefix = "[x]" if active else "[ ]"
        return f"{prefix} {name}"

    def _color_for_log(self, line: str):
        for keyword, c in _LOG_COLORS:
            if keyword in line:
                return c
        return _TOKENS["text"]

    @staticmethod
    def _truncate(content: str, max_chars: int = 64) -> str:
        if len(content) <= max_chars:
            return content
        return content[: max_chars - 1] + "..."

    def _make_toggle_handler(self, key: str) -> Callable[[], None]:
        def handler() -> None:
            new_val = not self._toggle_state.get(key, False)
            cb_attr = f"on_toggle_{key}"
            cb = getattr(self.callbacks, cb_attr, None)
            if callable(cb):
                cb(new_val)
        return handler

    def _add_panel(
        self,
        fx: float,
        fy: float,
        fw: float,
        fh: float,
        col=None,
        z: float = 0.05,
    ) -> Entity:
        panel = Entity(
            parent=camera.ui,
            model="quad",
            position=self._fpos(fx, fy),
            scale=self._fscale(fw, fh),
            origin=(-0.5, 0.5),
            color=col if col is not None else _TOKENS["surface"],
            z=z,
        )
        self._entities.append(panel)
        return panel

    @staticmethod
    def _fpos(fx: float, fy: float) -> tuple:
        # camera.ui visible range: x in [-aspect/2, +aspect/2], y in [-0.5, +0.5]
        # Map fx, fy in [0, 1] to that visible viewport.
        aspect = float(getattr(window, "aspect_ratio", 1.7777777778))
        return (-aspect * 0.5 + aspect * fx, 0.5 - 1.0 * fy)

    @staticmethod
    def _fscale(fw: float, fh: float) -> tuple:
        aspect = float(getattr(window, "aspect_ratio", 1.7777777778))
        return (aspect * fw, 1.0 * fh)

    def _style_button_text(self, btn: Button, scale: float = 0.9) -> None:
        if btn.text_entity is None:
            return
        try:
            btn.text_entity.scale = scale
        except Exception:
            pass
        try:
            btn.text_entity.color = _TOKENS["text"]
        except Exception:
            pass
        try:
            btn.text_entity.z = -0.01
        except Exception:
            pass
