"""3D scene rendering for CityMind: builds Ursina entities from the shared graph."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

from ursina import Entity, Text, color, destroy

from ..city_graph import Cell, CityGraph, LocationType, RiskLevel
from .setup_pipeline import SetupArtifacts


Edge = Tuple[Cell, Cell]


def _rgb01(r: float, g: float, b: float):
    """Build a Color from normalized 0..1 channels (Ursina's native range)."""
    return color.rgb(float(r), float(g), float(b))


def _rgba01(r: float, g: float, b: float, a: float):
    """Build a Color from normalized 0..1 channels and alpha."""
    return color.rgba(float(r), float(g), float(b), float(a))


@dataclass
class ViewState:
    """Boolean toggles that control which overlays are active."""

    show_roads: bool = True
    show_risk: bool = True
    show_coverage: bool = False
    show_population: bool = False
    show_clusters: bool = False
    show_ga_redundancy: bool = False
    show_astar_trace: bool = False
    show_csp_diagnostics: bool = False
    heatmap_opacity: float = 0.68


_LOCATION_COLORS = {
    LocationType.HOSPITAL: _rgb01(0.95, 0.95, 0.95),
    LocationType.SCHOOL: _rgb01(0.95, 0.85, 0.25),
    LocationType.INDUSTRIAL: _rgb01(0.45, 0.45, 0.5),
    LocationType.POWER_PLANT: _rgb01(0.95, 0.55, 0.15),
    LocationType.AMBULANCE_DEPOT: _rgb01(0.85, 0.2, 0.25),
    LocationType.RESIDENTIAL: _rgb01(0.3, 0.55, 0.7),
}

_LOCATION_HEIGHTS = {
    LocationType.HOSPITAL: 1.4,
    LocationType.SCHOOL: 0.9,
    LocationType.INDUSTRIAL: 0.6,
    LocationType.POWER_PLANT: 1.6,
    LocationType.AMBULANCE_DEPOT: 0.7,
    LocationType.RESIDENTIAL: 0.55,
}

_LOCATION_ROOF_COLORS = {
    LocationType.HOSPITAL: _rgb01(0.92, 0.30, 0.30),
    LocationType.SCHOOL: _rgb01(0.55, 0.32, 0.18),
    LocationType.INDUSTRIAL: _rgb01(0.30, 0.30, 0.34),
    LocationType.POWER_PLANT: _rgb01(0.30, 0.30, 0.34),
    LocationType.AMBULANCE_DEPOT: _rgb01(0.95, 0.95, 0.95),
    LocationType.RESIDENTIAL: _rgb01(0.55, 0.72, 0.85),
}

_RISK_TINT = {
    RiskLevel.LOW: _rgb01(0.25, 0.7, 0.3),
    RiskLevel.MEDIUM: _rgb01(0.95, 0.85, 0.2),
    RiskLevel.HIGH: _rgb01(0.85, 0.25, 0.25),
}

_TEAM_COLOR = _rgb01(0.95, 0.2, 0.95)
_RESIDENTIAL_POP_BASE_HEIGHT = 0.45
_RESIDENTIAL_POP_HEIGHT_SCALE = 0.55
_PATH_COLOR = _rgb01(0.95, 0.95, 0.2)
_CIVILIAN_PENDING = _rgb01(0.95, 0.65, 0.25)
_CIVILIAN_CURRENT = _rgb01(0.25, 0.95, 0.95)
_CIVILIAN_REACHED = _rgb01(0.45, 0.75, 0.45)
_ROAD_OPEN_STANDARD = _rgb01(0.72, 0.76, 0.82)
_ROAD_OPEN_DISCOUNT = _rgb01(0.45, 0.92, 0.65)
_ROAD_BLOCKED = _rgb01(0.95, 0.25, 0.25)
_ROAD_FLASH = _rgb01(1.0, 0.9, 0.2)
_GA_PATH_A = _rgb01(0.3, 0.8, 1.0)
_GA_PATH_B = _rgb01(0.9, 0.5, 1.0)
_GA_PATH_SHARED = _rgb01(1.0, 0.95, 0.35)
_CSP_DIAG = _rgb01(1.0, 0.35, 0.35)


@dataclass
class _Entities:
    cell_pads: Dict[Cell, Entity] = field(default_factory=dict)
    buildings: Dict[Cell, Entity] = field(default_factory=dict)
    building_accents: Dict[Cell, List[Entity]] = field(default_factory=dict)
    roads: Dict[Edge, Entity] = field(default_factory=dict)
    intersections: List[Entity] = field(default_factory=list)
    grid_lines: List[Entity] = field(default_factory=list)
    police: Dict[Cell, Entity] = field(default_factory=dict)
    ambulances: List[Entity] = field(default_factory=list)
    ambulance_accents: List[Entity] = field(default_factory=list)
    path_markers: List[Entity] = field(default_factory=list)
    astar_trace_markers: List[Entity] = field(default_factory=list)
    csp_diagnostic_markers: List[Entity] = field(default_factory=list)
    civilian_markers: Dict[int, Tuple[Entity, Text]] = field(default_factory=dict)
    civilian_rings: Dict[int, Entity] = field(default_factory=dict)
    completion_rings: List[Tuple[Entity, float]] = field(default_factory=list)
    team: Optional[Entity] = None
    team_halo: Optional[Entity] = None
    team_glow: Optional[Entity] = None
    ground: Optional[Entity] = None


class CityScene:
    """Builds and maintains 3D entities for a CityMind setup."""

    def __init__(self) -> None:
        self.view_state = ViewState()
        self._entities = _Entities()
        self._coverage_cache: Dict[Cell, int] = {}
        self._artifacts: Optional[SetupArtifacts] = None
        self._origin_x: float = 0.0
        self._origin_z: float = 0.0
        self._flash_road: Optional[Edge] = None
        self._flash_road_ticks: int = 0
        self._pulse_civilian_idx: Optional[int] = None
        self._pulse_civilian_ticks: int = 0
        self._entity_to_cell: Dict[int, Cell] = {}
        self._selected_cell: Optional[Cell] = None
        self._on_cell_selected: Optional[Callable[[Cell], None]] = None
        self._ga_path_a_edges: Set[Edge] = set()
        self._ga_path_b_edges: Set[Edge] = set()
        self._astar_trace_paths: List[List[Cell]] = []
        self._csp_diag_cells: Set[Cell] = set()
        self._path_flow_phase: float = 0.0

    def build(self, artifacts: SetupArtifacts) -> None:
        self.destroy()
        self._artifacts = artifacts
        graph = artifacts.graph
        self._origin_x = -graph.cols / 2.0 + 0.5
        self._origin_z = -graph.rows / 2.0 + 0.5

        self._entities.ground = Entity(
            model="plane",
            color=_rgb01(0.10, 0.12, 0.16),
            scale=(graph.cols + 8, 1, graph.rows + 8),
            position=(0, -0.05, 0),
            collider=None,
        )
        self._build_ground_grid(graph)

        for cell, node in graph.nodes.items():
            world = self._cell_world(cell)
            pad_color = _LOCATION_COLORS.get(node.location_type, _rgb01(0.18, 0.20, 0.24))
            pad = Entity(
                model="cube",
                color=pad_color,
                scale=(1.0, 0.04, 1.0),
                position=(world[0], 0.02, world[2]),
            )
            pad.on_click = lambda cell=cell: self._handle_cell_click(cell)
            self._entities.cell_pads[cell] = pad
            self._entity_to_cell[id(pad)] = cell

            if node.location_type is not None:
                building_height = _LOCATION_HEIGHTS.get(node.location_type, 0.5)
                building = Entity(
                    model="cube",
                    color=pad_color,
                    scale=(0.62, building_height, 0.62),
                    position=(world[0], building_height / 2.0 + 0.04, world[2]),
                )
                building.on_click = lambda cell=cell: self._handle_cell_click(cell)
                self._entities.buildings[cell] = building
                self._entity_to_cell[id(building)] = cell
                self._build_building_accents(cell, node.location_type, world, building_height)

        self._build_roads(graph)
        self._build_intersections(graph)
        self._build_police(graph)
        self._build_ambulances(artifacts.sa_optimizer.current_positions or artifacts.ambulance_result.positions)
        self._build_civilian_markers(artifacts.runner.state.civilians_ordered)
        self._build_team(artifacts.runner.state.team_pos)
        self._refresh_path()
        self._refresh_civilian_marker_states()
        self._recompute_ga_redundancy()
        self._recompute_csp_diagnostics()
        self._recompute_coverage_cache()
        self._apply_view_state()

    def destroy(self) -> None:
        for e in list(self._entities.cell_pads.values()):
            destroy(e)
        for e in list(self._entities.buildings.values()):
            destroy(e)
        for accents in list(self._entities.building_accents.values()):
            for e in accents:
                destroy(e)
        for e in list(self._entities.roads.values()):
            destroy(e)
        for e in self._entities.intersections:
            destroy(e)
        for e in self._entities.grid_lines:
            destroy(e)
        for e in list(self._entities.police.values()):
            destroy(e)
        for e in self._entities.ambulances:
            destroy(e)
        for e in self._entities.ambulance_accents:
            destroy(e)
        for e in self._entities.path_markers:
            destroy(e)
        for e in self._entities.astar_trace_markers:
            destroy(e)
        for e in self._entities.csp_diagnostic_markers:
            destroy(e)
        for marker, label in self._entities.civilian_markers.values():
            destroy(marker)
            destroy(label)
        for e in self._entities.civilian_rings.values():
            destroy(e)
        for e, _ in self._entities.completion_rings:
            destroy(e)
        if self._entities.team is not None:
            destroy(self._entities.team)
        if self._entities.team_halo is not None:
            destroy(self._entities.team_halo)
        if self._entities.team_glow is not None:
            destroy(self._entities.team_glow)
        if self._entities.ground is not None:
            destroy(self._entities.ground)
        self._entities = _Entities()
        self._coverage_cache.clear()
        self._artifacts = None
        self._flash_road = None
        self._flash_road_ticks = 0
        self._pulse_civilian_idx = None
        self._pulse_civilian_ticks = 0
        self._entity_to_cell.clear()
        self._selected_cell = None
        self._ga_path_a_edges.clear()
        self._ga_path_b_edges.clear()
        self._astar_trace_paths = []
        self._csp_diag_cells.clear()
        self._path_flow_phase = 0.0

    def refresh(self) -> None:
        if self._artifacts is None:
            return
        artifacts = self._artifacts
        self._refresh_roads(artifacts.graph)
        ambulance_positions = (
            artifacts.sa_optimizer.current_positions
            or artifacts.ambulance_result.positions
        )
        self._refresh_ambulances(ambulance_positions)
        self._refresh_team(artifacts.runner.state.team_pos)
        self._refresh_path()
        self._refresh_civilian_marker_states()
        if self.view_state.show_ga_redundancy:
            self._recompute_ga_redundancy()
        self._advance_effects()
        self._apply_view_state()

    def set_view(self, **toggles: bool) -> None:
        for key, value in toggles.items():
            if hasattr(self.view_state, key):
                setattr(self.view_state, key, bool(value))
        self._apply_view_state()

    def set_heatmap_opacity(self, value: float) -> None:
        self.view_state.heatmap_opacity = max(0.0, min(1.0, float(value)))
        self._apply_view_state()

    def invalidate_coverage_cache(self) -> None:
        self._recompute_coverage_cache()
        if self.view_state.show_coverage:
            self._apply_view_state()

    def set_on_cell_selected(self, handler: Optional[Callable[[Cell], None]]) -> None:
        self._on_cell_selected = handler

    def selected_cell(self) -> Optional[Cell]:
        return self._selected_cell

    def clear_selected_cell(self) -> None:
        self._selected_cell = None
        self._apply_view_state()

    def hover_tooltip_text(self) -> str:
        if self._selected_cell is None:
            return "Click a cell to inspect"
        snap = self.cell_snapshot(self._selected_cell)
        if snap is None:
            return "Click a cell to inspect"
        return (
            f"{snap['cell']} {snap['location_type']} | "
            f"Risk: {snap['risk_level']} | Cluster: {snap['cluster_id']}"
        )

    def cell_snapshot(self, cell: Cell) -> Optional[dict]:
        if self._artifacts is None:
            return None
        if cell not in self._artifacts.graph.nodes:
            return None
        graph = self._artifacts.graph
        node = graph.nodes[cell]
        state = self._artifacts.runner.state
        neighbors_all = graph.neighbors(cell, include_blocked=True, include_inaccessible=True)
        blocked_edges = 0
        for n in neighbors_all:
            if graph.adjacency[cell][n].blocked:
                blocked_edges += 1
        return {
            "cell": cell,
            "location_type": node.location_type.value if node.location_type is not None else "Unassigned",
            "population": node.population,
            "risk_level": node.risk_level.value,
            "risk_mult": round(node.risk_mult, 2),
            "cluster_id": node.cluster_id if node.cluster_id is not None else "-",
            "accessible": node.accessible,
            "police_post": node.police_post,
            "is_team_here": state.team_pos == cell,
            "is_current_target": state.current_target == cell,
            "is_on_current_path": cell in state.current_path,
            "open_neighbors": len(neighbors_all) - blocked_edges,
            "blocked_neighbors": blocked_edges,
            "explainability": f"RiskLabel: {node.risk_level.value} (mult={node.risk_mult:.1f}) | ClusterId: {node.cluster_id if node.cluster_id is not None else '-'}",
        }

    def _handle_cell_click(self, cell: Cell) -> None:
        self._selected_cell = cell
        self._apply_view_state()
        if self._on_cell_selected is not None:
            self._on_cell_selected(cell)

    def team_world_position(self, cell: Cell) -> Tuple[float, float, float]:
        world = self._cell_world(cell)
        return (world[0], 0.55, world[2])

    def _build_ground_grid(self, graph: CityGraph) -> None:
        line_color = _rgb01(0.16, 0.18, 0.22)
        # Vertical lines (between columns)
        for c in range(graph.cols + 1):
            x = self._origin_x + c - 0.5
            line = Entity(
                model="cube",
                color=line_color,
                position=(x, 0.0, 0.0),
                scale=(0.04, 0.01, graph.rows + 0.4),
                collider=None,
            )
            self._entities.grid_lines.append(line)
        # Horizontal lines (between rows)
        for r in range(graph.rows + 1):
            z = self._origin_z + r - 0.5
            line = Entity(
                model="cube",
                color=line_color,
                position=(0.0, 0.0, z),
                scale=(graph.cols + 0.4, 0.01, 0.04),
                collider=None,
            )
            self._entities.grid_lines.append(line)

    def _build_roads(self, graph: CityGraph) -> None:
        seen: Set[Edge] = set()
        for a in graph.nodes:
            for b, edge in graph.adjacency[a].items():
                key = self._edge_key(a, b)
                if key in seen:
                    continue
                seen.add(key)
                wa = self._cell_world(a)
                wb = self._cell_world(b)
                mid = ((wa[0] + wb[0]) / 2.0, 0.07, (wa[2] + wb[2]) / 2.0)
                horizontal = abs(a[1] - b[1]) > 0
                if horizontal:
                    scale = (1.0, 0.05, 0.30)
                else:
                    scale = (0.30, 0.05, 1.0)
                bar = Entity(
                    model="cube",
                    color=_ROAD_OPEN_STANDARD,
                    position=mid,
                    scale=scale,
                )
                self._entities.roads[key] = bar

    def _build_intersections(self, graph: CityGraph) -> None:
        # Add a small cap at every cell center so the road grid reads as a continuous network.
        cap_color = _rgb01(0.62, 0.66, 0.74)
        for cell in graph.nodes:
            world = self._cell_world(cell)
            cap = Entity(
                model="cube",
                color=cap_color,
                position=(world[0], 0.075, world[2]),
                scale=(0.32, 0.04, 0.32),
                collider=None,
            )
            self._entities.intersections.append(cap)

    def _build_building_accents(
        self,
        cell: Cell,
        location_type: LocationType,
        world: Tuple[float, float, float],
        building_height: float,
    ) -> None:
        accents: List[Entity] = []
        x, _, z = world
        roof_y = building_height + 0.04 + 0.02
        roof_color = _LOCATION_ROOF_COLORS.get(location_type, _rgb01(0.5, 0.5, 0.5))

        if location_type is LocationType.HOSPITAL:
            # Red cross on roof
            accents.append(Entity(
                model="cube",
                color=_rgb01(0.95, 0.25, 0.25),
                position=(x, roof_y + 0.03, z),
                scale=(0.32, 0.06, 0.10),
                collider=None,
            ))
            accents.append(Entity(
                model="cube",
                color=_rgb01(0.95, 0.25, 0.25),
                position=(x, roof_y + 0.03, z),
                scale=(0.10, 0.06, 0.32),
                collider=None,
            ))
        elif location_type is LocationType.SCHOOL:
            # Brown roof slab
            accents.append(Entity(
                model="cube",
                color=roof_color,
                position=(x, roof_y, z),
                scale=(0.66, 0.08, 0.66),
                collider=None,
            ))
        elif location_type is LocationType.INDUSTRIAL:
            # Chimney on the corner
            accents.append(Entity(
                model="cube",
                color=_rgb01(0.30, 0.30, 0.34),
                position=(x + 0.18, roof_y + 0.18, z + 0.18),
                scale=(0.12, 0.45, 0.12),
                collider=None,
            ))
            accents.append(Entity(
                model="cube",
                color=_rgb01(0.18, 0.20, 0.24),
                position=(x + 0.18, roof_y + 0.42, z + 0.18),
                scale=(0.16, 0.04, 0.16),
                collider=None,
            ))
        elif location_type is LocationType.POWER_PLANT:
            # Two cooling towers
            for dx, dz in ((-0.16, -0.16), (0.16, 0.16)):
                accents.append(Entity(
                    model="cube",
                    color=_rgb01(0.30, 0.30, 0.34),
                    position=(x + dx, roof_y + 0.20, z + dz),
                    scale=(0.18, 0.55, 0.18),
                    collider=None,
                ))
                accents.append(Entity(
                    model="cube",
                    color=_rgb01(0.18, 0.20, 0.24),
                    position=(x + dx, roof_y + 0.50, z + dz),
                    scale=(0.22, 0.06, 0.22),
                    collider=None,
                ))
        elif location_type is LocationType.AMBULANCE_DEPOT:
            # White cross on roof
            accents.append(Entity(
                model="cube",
                color=_rgb01(0.97, 0.97, 0.97),
                position=(x, roof_y + 0.03, z),
                scale=(0.30, 0.06, 0.10),
                collider=None,
            ))
            accents.append(Entity(
                model="cube",
                color=_rgb01(0.97, 0.97, 0.97),
                position=(x, roof_y + 0.03, z),
                scale=(0.10, 0.06, 0.30),
                collider=None,
            ))
        elif location_type is LocationType.RESIDENTIAL:
            # Lighter blue roof slab
            accents.append(Entity(
                model="cube",
                color=roof_color,
                position=(x, roof_y, z),
                scale=(0.66, 0.06, 0.66),
                collider=None,
            ))

        if accents:
            self._entities.building_accents[cell] = accents

    def _build_police(self, graph: CityGraph) -> None:
        for cell, node in graph.nodes.items():
            if not node.police_post:
                continue
            world = self._cell_world(cell)
            pillar = Entity(
                model="cube",
                color=_rgb01(0.20, 0.42, 1.00),
                position=(world[0] - 0.30, 0.50, world[2] - 0.30),
                scale=(0.20, 1.00, 0.20),
                collider=None,
            )
            cap = Entity(
                model="cube",
                color=_rgb01(0.10, 0.20, 0.55),
                position=(world[0] - 0.30, 1.04, world[2] - 0.30),
                scale=(0.26, 0.08, 0.26),
                collider=None,
            )
            self._entities.police[cell] = pillar
            self._entities.intersections.append(cap)

    def _build_ambulances(self, positions) -> None:
        if positions is None:
            return
        for pos in positions:
            world = self._cell_world(pos)
            body = Entity(
                model="cube",
                color=_rgb01(0.97, 0.97, 0.97),
                position=(world[0] + 0.28, 0.32, world[2] + 0.28),
                scale=(0.36, 0.32, 0.56),
                collider=None,
            )
            stripe = Entity(
                model="cube",
                color=_rgb01(0.92, 0.20, 0.20),
                position=(world[0] + 0.28, 0.48, world[2] + 0.28),
                scale=(0.38, 0.05, 0.58),
                collider=None,
            )
            self._entities.ambulances.append(body)
            self._entities.ambulance_accents.append(stripe)

    def _build_civilian_markers(self, civilians: List[Cell]) -> None:
        for idx, cell in enumerate(civilians):
            world = self._cell_world(cell)
            marker = Entity(
                model="cube",
                color=_CIVILIAN_PENDING,
                position=(world[0], 0.55, world[2]),
                scale=(0.22, 1.00, 0.22),
                collider=None,
            )
            ring = Entity(
                model="quad",
                color=_rgba01(0.95, 0.65, 0.25, 0.55),
                position=(world[0], 0.06, world[2]),
                rotation_x=90,
                scale=(0.85, 0.85, 0.85),
                collider=None,
            )
            # Small flag on top of the pillar so civilians are easy to spot.
            flag = Entity(
                model="cube",
                color=_rgb01(0.95, 0.95, 0.95),
                position=(world[0], 1.18, world[2]),
                scale=(0.22, 0.06, 0.22),
                collider=None,
            )
            try:
                label = Text(
                    text=str(idx + 1),
                    position=(world[0], 1.30, world[2]),
                    origin=(0, 0),
                    scale=2.0,
                    color=_rgb01(0.97, 0.97, 0.99),
                    billboard=True,
                )
            except Exception:
                label = Text(text=str(idx + 1), enabled=False)
            self._entities.civilian_markers[idx] = (marker, label)
            self._entities.civilian_rings[idx] = ring
            self._entities.intersections.append(flag)

    def _build_team(self, cell: Cell) -> None:
        world = self._cell_world(cell)
        team = Entity(
            model="sphere",
            color=_TEAM_COLOR,
            position=(world[0], 0.55, world[2]),
            scale=(0.45, 0.45, 0.45),
            collider=None,
        )
        self._entities.team = team
        self._entities.team_halo = Entity(
            model="quad",
            color=_rgba01(0.95, 0.20, 0.95, 0.65),
            position=(world[0], 0.05, world[2]),
            rotation_x=90,
            scale=(0.95, 0.95, 0.95),
            collider=None,
        )
        self._entities.team_glow = Entity(
            model="quad",
            color=_rgba01(0.95, 0.20, 0.95, 0.25),
            position=(world[0], 0.05, world[2]),
            rotation_x=90,
            scale=(1.55, 1.55, 1.55),
            collider=None,
        )

    def _refresh_roads(self, graph: CityGraph) -> None:
        for (a, b), entity in self._entities.roads.items():
            edge = graph.adjacency[a][b]
            entity.color = self._resolve_road_color(a, b, edge)
            entity.enabled = self.view_state.show_roads

    def _refresh_ambulances(self, positions) -> None:
        if positions is None:
            return
        for idx, pos in enumerate(positions):
            world = self._cell_world(pos)
            if idx < len(self._entities.ambulances):
                self._entities.ambulances[idx].position = (world[0] + 0.28, 0.32, world[2] + 0.28)
            if idx < len(self._entities.ambulance_accents):
                self._entities.ambulance_accents[idx].position = (world[0] + 0.28, 0.48, world[2] + 0.28)

    def _refresh_team(self, cell: Cell) -> None:
        if self._entities.team is None:
            return
        world = self._cell_world(cell)
        self.set_team_visual_position((world[0], 0.55, world[2]))

    def _refresh_path(self) -> None:
        for marker in self._entities.path_markers:
            destroy(marker)
        self._entities.path_markers = []
        if self._artifacts is None:
            return
        path = self._artifacts.runner.state.current_path
        if not path or len(path) < 2:
            return
        for a, b in zip(path[:-1], path[1:]):
            wa = self._cell_world(a)
            wb = self._cell_world(b)
            mid = ((wa[0] + wb[0]) / 2.0, 0.18, (wa[2] + wb[2]) / 2.0)
            horizontal = abs(a[1] - b[1]) > 0
            marker = Entity(
                model="cube",
                color=_PATH_COLOR,
                position=mid,
                scale=(0.96, 0.08, 0.22) if horizontal else (0.22, 0.08, 0.96),
                collider=None,
            )
            self._entities.path_markers.append(marker)

    def _refresh_civilian_marker_states(self) -> None:
        if self._artifacts is None:
            return
        state = self._artifacts.runner.state
        for idx, (marker, _label) in self._entities.civilian_markers.items():
            if idx < state.target_index:
                marker.color = _CIVILIAN_REACHED
                ring = self._entities.civilian_rings.get(idx)
                if ring is not None:
                    ring.color = _rgba01(0.45, 0.85, 0.45, 0.55)
            elif idx == state.target_index:
                marker.color = _CIVILIAN_CURRENT
                ring = self._entities.civilian_rings.get(idx)
                if ring is not None:
                    ring.color = _rgba01(0.25, 0.95, 0.95, 0.65)
            else:
                marker.color = _CIVILIAN_PENDING
                ring = self._entities.civilian_rings.get(idx)
                if ring is not None:
                    ring.color = _rgba01(0.95, 0.65, 0.25, 0.55)

    def set_team_visual_position(self, position: Tuple[float, float, float]) -> None:
        if self._entities.team is not None:
            self._entities.team.position = position
        if self._entities.team_halo is not None:
            self._entities.team_halo.position = (position[0], 0.05, position[2])
        if self._entities.team_glow is not None:
            self._entities.team_glow.position = (position[0], 0.05, position[2])

    def highlight_event_line(self, event_line: str) -> None:
        if "flood: edge" in event_line:
            edge = self._parse_flood_edge(event_line)
            if edge is not None:
                self._flash_road = self._edge_key(edge[0], edge[1])
                self._flash_road_ticks = 3
        elif "reached civilian:" in event_line and self._artifacts is not None:
            reached = self._parse_cell(event_line.split("reached civilian:", 1)[1].strip())
            if reached is None:
                return
            for idx, cell in enumerate(self._artifacts.runner.state.civilians_ordered):
                if cell == reached:
                    self._pulse_civilian_idx = idx
                    self._pulse_civilian_ticks = 3
                    self._spawn_completion_ring(cell)
                    break
        elif "A* replan:" in event_line and self._artifacts is not None:
            path = list(self._artifacts.runner.state.current_path)
            if path:
                self._astar_trace_paths.append(path)
                self._astar_trace_paths = self._astar_trace_paths[-3:]

    def _spawn_completion_ring(self, cell: Cell) -> None:
        world = self._cell_world(cell)
        ring = Entity(
            model="quad",
            color=_rgba01(0.45, 0.95, 0.45, 0.85),
            position=(world[0], 0.07, world[2]),
            rotation_x=90,
            scale=(0.4, 0.4, 0.4),
            collider=None,
        )
        self._entities.completion_rings.append((ring, 0.0))

    def animate(self, dt: float) -> None:
        if dt <= 0.0:
            return
        # Flowing path color
        self._path_flow_phase += dt * 4.0
        for idx, marker in enumerate(self._entities.path_markers):
            phase = int((self._path_flow_phase + idx) % 4)
            marker.color = _PATH_COLOR if phase in (0, 1) else _rgb01(0.55, 0.85, 1.0)
        # Team halo pulse
        if self._entities.team_glow is not None:
            t = self._path_flow_phase * 0.5
            scale = 1.45 + 0.20 * (0.5 + 0.5 * math.sin(t))
            self._entities.team_glow.scale = (scale, scale, scale)
        # Completion rings: expand and fade
        next_rings: List[Tuple[Entity, float]] = []
        for ring, age in self._entities.completion_rings:
            new_age = age + dt
            if new_age >= 0.7:
                destroy(ring)
                continue
            t = new_age / 0.7
            scale = 0.4 + 1.4 * t
            ring.scale = (scale, scale, scale)
            ring.color = _rgba01(0.45, 0.95, 0.45, max(0.0, 0.85 * (1.0 - t)))
            next_rings.append((ring, new_age))
        self._entities.completion_rings = next_rings

    def _advance_effects(self) -> None:
        # Tick-based effects (flash road, civilian pulse). Continuous flow lives in animate().
        if self._pulse_civilian_idx is not None:
            pair = self._entities.civilian_markers.get(self._pulse_civilian_idx)
            if pair is not None:
                marker, _ = pair
                marker.scale = (
                    0.26,
                    1.10,
                    0.26,
                ) if self._pulse_civilian_ticks % 2 == 1 else (0.22, 1.00, 0.22)
            self._pulse_civilian_ticks -= 1
            if self._pulse_civilian_ticks <= 0:
                self._pulse_civilian_idx = None
        if self._flash_road_ticks > 0:
            self._flash_road_ticks -= 1
            if self._flash_road_ticks == 0:
                self._flash_road = None

    @staticmethod
    def _parse_cell(raw: str) -> Optional[Cell]:
        text = raw.strip()
        if not (text.startswith("(") and text.endswith(")")):
            return None
        parts = text[1:-1].split(",")
        if len(parts) != 2:
            return None
        try:
            return (int(parts[0].strip()), int(parts[1].strip()))
        except ValueError:
            return None

    @classmethod
    def _parse_flood_edge(cls, line: str) -> Optional[Edge]:
        if "edge" not in line:
            return None
        edge_str = line.split("edge", 1)[1].split("blocked", 1)[0].strip()
        if "-" not in edge_str:
            return None
        left, right = edge_str.split("-", 1)
        a = cls._parse_cell(left.strip())
        b = cls._parse_cell(right.strip())
        if a is None or b is None:
            return None
        return (a, b)

    def _apply_view_state(self) -> None:
        if self._artifacts is None:
            return
        graph = self._artifacts.graph
        for cell, pad in self._entities.cell_pads.items():
            node = graph.nodes[cell]
            pad.color = self._resolve_pad_color(cell, node)
            if self._selected_cell == cell:
                pad.scale = (1.06, 0.06, 1.06)
            else:
                pad.scale = (1.0, 0.04, 1.0)

        for cell, entity in self._entities.buildings.items():
            node = graph.nodes[cell]
            target_height, target_color = self._resolve_building_visual(cell, node)
            entity.color = target_color
            entity.scale = (0.62, target_height, 0.62)
            entity.position = (
                entity.position[0],
                target_height / 2.0 + 0.04,
                entity.position[2],
            )
            # Re-anchor accents to the new roof height.
            accents = self._entities.building_accents.get(cell)
            if accents:
                self._reposition_accents(cell, node.location_type, entity.position, target_height, accents)

        for entity in self._entities.roads.values():
            entity.enabled = self.view_state.show_roads
        for entity in self._entities.intersections:
            entity.enabled = self.view_state.show_roads
        self._refresh_astar_trace_markers()
        self._refresh_csp_diagnostic_markers()

    def _reposition_accents(
        self,
        _cell: Cell,
        location_type,
        building_pos: Tuple[float, float, float],
        building_height: float,
        accents: List[Entity],
    ) -> None:
        x = building_pos[0]
        z = building_pos[2]
        roof_y = building_height + 0.04 + 0.02

        if location_type is LocationType.HOSPITAL:
            if len(accents) >= 1:
                accents[0].position = (x, roof_y + 0.03, z)
            if len(accents) >= 2:
                accents[1].position = (x, roof_y + 0.03, z)
        elif location_type is LocationType.SCHOOL:
            if accents:
                accents[0].position = (x, roof_y, z)
        elif location_type is LocationType.INDUSTRIAL:
            if len(accents) >= 1:
                accents[0].position = (x + 0.18, roof_y + 0.18, z + 0.18)
            if len(accents) >= 2:
                accents[1].position = (x + 0.18, roof_y + 0.42, z + 0.18)
        elif location_type is LocationType.POWER_PLANT:
            offsets = ((-0.16, -0.16), (0.16, 0.16))
            for i, (dx, dz) in enumerate(offsets):
                base_idx = i * 2
                if base_idx < len(accents):
                    accents[base_idx].position = (x + dx, roof_y + 0.20, z + dz)
                if base_idx + 1 < len(accents):
                    accents[base_idx + 1].position = (x + dx, roof_y + 0.50, z + dz)
        elif location_type is LocationType.AMBULANCE_DEPOT:
            if len(accents) >= 1:
                accents[0].position = (x, roof_y + 0.03, z)
            if len(accents) >= 2:
                accents[1].position = (x, roof_y + 0.03, z)
        elif location_type is LocationType.RESIDENTIAL:
            if accents:
                accents[0].position = (x, roof_y, z)

    def _resolve_pad_color(self, cell: Cell, node) -> color:
        base = _LOCATION_COLORS.get(node.location_type, _rgb01(0.3, 0.3, 0.3))
        if self.view_state.show_clusters and node.cluster_id is not None:
            return self._cluster_color_for_id(node.cluster_id)
        # Deterministic precedence: coverage > risk > base.
        if self.view_state.show_coverage and cell in self._coverage_cache:
            idx = self._coverage_cache[cell]
            return self._coverage_color_for_index(idx)
        if self.view_state.show_risk and node.location_type is not None:
            risk = _RISK_TINT.get(node.risk_level, base)
            return self._mix(base, risk, self.view_state.heatmap_opacity)
        if self._selected_cell == cell:
            return self._mix(base, _rgb01(0.2, 0.55, 0.95), 0.45)
        return base

    def _resolve_building_visual(self, _cell: Cell, node) -> Tuple[float, color]:
        base_color = _LOCATION_COLORS.get(node.location_type, _rgb01(0.3, 0.3, 0.3))
        target_height = _LOCATION_HEIGHTS.get(node.location_type, 0.4)
        if self.view_state.show_population and node.location_type == LocationType.RESIDENTIAL:
            pop_norm = min(1.0, max(0.0, node.population / 200.0))
            target_height = _RESIDENTIAL_POP_BASE_HEIGHT + _RESIDENTIAL_POP_HEIGHT_SCALE * pop_norm
            pop_tint = _rgb01(0.12, 0.35 + 0.45 * pop_norm, 0.95)
            return target_height, self._mix(base_color, pop_tint, 0.55)
        return target_height, base_color

    def _resolve_road_color(self, a: Cell, b: Cell, edge) -> color:
        if self._flash_road is not None and self._edge_key(a, b) == self._flash_road:
            return _ROAD_FLASH
        if edge.blocked:
            return _ROAD_BLOCKED
        if self.view_state.show_ga_redundancy:
            key = self._edge_key(a, b)
            in_a = key in self._ga_path_a_edges
            in_b = key in self._ga_path_b_edges
            if in_a and in_b:
                return _GA_PATH_SHARED
            if in_a:
                return _GA_PATH_A
            if in_b:
                return _GA_PATH_B
        if edge.base_cost < 1.0:
            return _ROAD_OPEN_DISCOUNT
        return _ROAD_OPEN_STANDARD

    @staticmethod
    def _mix(base, tint, amount: float):
        a = max(0.0, min(1.0, amount))
        return _rgba01(
            base.r * (1.0 - a) + tint.r * a,
            base.g * (1.0 - a) + tint.g * a,
            base.b * (1.0 - a) + tint.b * a,
            base.a,
        )

    def _recompute_coverage_cache(self) -> None:
        self._coverage_cache.clear()
        if self._artifacts is None:
            return
        positions = (
            self._artifacts.sa_optimizer.current_positions
            or self._artifacts.ambulance_result.positions
        )
        if not positions:
            return
        graph = self._artifacts.graph
        sa = self._artifacts.sa_optimizer
        dist_maps = [sa._dijkstra_distances(src) for src in positions]
        for cell in graph.nodes:
            best_idx = 0
            best_dist = float("inf")
            for idx, dm in enumerate(dist_maps):
                d = dm.get(cell, float("inf"))
                if d < best_dist:
                    best_dist = d
                    best_idx = idx
            if best_dist != float("inf"):
                self._coverage_cache[cell] = best_idx

    def _coverage_color_for_index(self, idx: int):
        # Deterministic palette generation by index, avoids modulo collisions.
        hue = (idx * 0.17) % 1.0
        sat = 0.62
        val = 0.9
        return color.hsv(hue * 360.0, sat, val)

    def _recompute_ga_redundancy(self) -> None:
        self._ga_path_a_edges.clear()
        self._ga_path_b_edges.clear()
        if self._artifacts is None:
            return
        graph = self._artifacts.graph
        hospitals = sorted(graph.find_by_type(LocationType.HOSPITAL))
        depots = sorted(graph.find_by_type(LocationType.AMBULANCE_DEPOT))
        if not hospitals or not depots:
            return
        source = hospitals[0]
        target = depots[0]
        path_a = self._bfs_path(source, target, banned_edges=set())
        if not path_a:
            return
        self._ga_path_a_edges = self._path_edges(path_a)
        path_b = self._bfs_path(source, target, banned_edges=self._ga_path_a_edges)
        if path_b:
            self._ga_path_b_edges = self._path_edges(path_b)

    def _recompute_csp_diagnostics(self) -> None:
        self._csp_diag_cells.clear()
        if self._artifacts is None:
            return
        graph = self._artifacts.graph
        hospitals = sorted(graph.find_by_type(LocationType.HOSPITAL))
        industrial = sorted(graph.find_by_type(LocationType.INDUSTRIAL))
        power_plants = sorted(graph.find_by_type(LocationType.POWER_PLANT))
        residential = sorted(graph.find_by_type(LocationType.RESIDENTIAL))
        for cell in residential:
            if not hospitals or min(
                self._hop_distance_over_all_edges(cell, h) for h in hospitals
            ) > 3:
                self._csp_diag_cells.add(cell)
        for plant in power_plants:
            if not industrial or min(
                self._hop_distance_over_all_edges(plant, i) for i in industrial
            ) > 2:
                self._csp_diag_cells.add(plant)

    def csp_diagnostic_count(self) -> int:
        return len(self._csp_diag_cells)

    def _bfs_path(self, source: Cell, target: Cell, banned_edges: Set[Edge]) -> List[Cell]:
        if source == target:
            return [source]
        if self._artifacts is None:
            return []
        graph = self._artifacts.graph
        queue: List[Cell] = [source]
        prev: Dict[Cell, Optional[Cell]] = {source: None}
        head = 0
        while head < len(queue):
            node = queue[head]
            head += 1
            for nxt in graph.neighbors(node):
                if self._edge_key(node, nxt) in banned_edges:
                    continue
                if nxt in prev:
                    continue
                prev[nxt] = node
                if nxt == target:
                    queue = []
                    break
                queue.append(nxt)
        if target not in prev:
            return []
        path: List[Cell] = [target]
        cur = target
        while prev[cur] is not None:
            cur = prev[cur]  # type: ignore[assignment]
            path.append(cur)
        path.reverse()
        return path

    def _path_edges(self, path: List[Cell]) -> Set[Edge]:
        return {self._edge_key(a, b) for a, b in zip(path[:-1], path[1:])}

    def _hop_distance_over_all_edges(self, source: Cell, target: Cell) -> int:
        if source == target:
            return 0
        if self._artifacts is None:
            return 10**9
        graph = self._artifacts.graph
        queue: List[Tuple[Cell, int]] = [(source, 0)]
        seen: Set[Cell] = {source}
        head = 0
        while head < len(queue):
            node, dist = queue[head]
            head += 1
            for nxt in graph.neighbors(node, include_blocked=True, include_inaccessible=True):
                if nxt in seen:
                    continue
                if nxt == target:
                    return dist + 1
                seen.add(nxt)
                queue.append((nxt, dist + 1))
        return 10**9

    def _cluster_color_for_id(self, cluster_id: int):
        hue = (cluster_id * 0.27) % 1.0
        return color.hsv(hue * 360.0, 0.68, 0.88)

    def _refresh_astar_trace_markers(self) -> None:
        for e in self._entities.astar_trace_markers:
            destroy(e)
        self._entities.astar_trace_markers = []
        if not self.view_state.show_astar_trace:
            return
        for trace_idx, path in enumerate(reversed(self._astar_trace_paths)):
            alpha = max(0.25, 0.8 - trace_idx * 0.2)
            for cell in path[:18]:
                world = self._cell_world(cell)
                marker = Entity(
                    model="cube",
                    color=_rgba01(0.95, 0.95, 1.0, alpha),
                    position=(world[0], 0.22 + trace_idx * 0.01, world[2]),
                    scale=(0.12, 0.03, 0.12),
                )
                self._entities.astar_trace_markers.append(marker)

    def _refresh_csp_diagnostic_markers(self) -> None:
        for e in self._entities.csp_diagnostic_markers:
            destroy(e)
        self._entities.csp_diagnostic_markers = []
        if not self.view_state.show_csp_diagnostics:
            return
        for cell in sorted(self._csp_diag_cells):
            world = self._cell_world(cell)
            marker = Entity(
                model="cube",
                color=_CSP_DIAG,
                position=(world[0], 0.35, world[2]),
                scale=(0.1, 0.7, 0.1),
            )
            self._entities.csp_diagnostic_markers.append(marker)

    def _cell_world(self, cell: Cell) -> Tuple[float, float, float]:
        r, c = cell
        return (self._origin_x + c, 0.0, self._origin_z + r)

    @staticmethod
    def _edge_key(a: Cell, b: Cell) -> Edge:
        return (a, b) if a <= b else (b, a)
