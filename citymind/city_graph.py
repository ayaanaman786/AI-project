"""Shared city graph model for all CityMind modules."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import heapq
from typing import Dict, Iterable, List, Optional, Set, Tuple

Cell = Tuple[int, int]


class LocationType(str, Enum):
    RESIDENTIAL = "Residential"
    HOSPITAL = "Hospital"
    SCHOOL = "School"
    INDUSTRIAL = "Industrial"
    POWER_PLANT = "Power Plant"
    AMBULANCE_DEPOT = "Ambulance Depot"


class RiskLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"

    @property
    def multiplier(self) -> float:
        return {
            RiskLevel.LOW: 1.0,
            RiskLevel.MEDIUM: 1.5,
            RiskLevel.HIGH: 2.0,
        }[self]


@dataclass
class NodeData:
    location_type: Optional[LocationType] = None
    population: int = 0
    risk_level: RiskLevel = RiskLevel.LOW
    risk_mult: float = 1.0
    accessible: bool = True
    cluster_id: Optional[int] = None
    police_post: bool = False


@dataclass
class EdgeData:
    base_cost: float
    blocked: bool = False


class CityGraph:
    """Grid-backed graph with dynamic edge costs and block states."""

    # Base-cost constants exposed publicly so heuristics elsewhere (notably
    # the A* admissibility guard) can reason about the minimum possible edge
    # cost without depending on hard-coded magic numbers.
    STANDARD_BASE_COST: float = 1.0
    RESIDENTIAL_BASE_COST: float = 0.8

    def __init__(self, rows: int = 10, cols: int = 10) -> None:
        if rows <= 0 or cols <= 0:
            raise ValueError("rows and cols must be positive")

        self.rows = rows
        self.cols = cols
        self.nodes: Dict[Cell, NodeData] = {
            (r, c): NodeData() for r in range(rows) for c in range(cols)
        }
        self.adjacency: Dict[Cell, Dict[Cell, EdgeData]] = {cell: {} for cell in self.nodes}
        self._init_grid_edges()

    @property
    def min_base_cost(self) -> float:
        """Smallest base_cost currently present on any edge.

        This is the floor any admissible heuristic must respect. We compute
        from current edge state so the guard remains correct even if a future
        residential discount is changed in `_edge_base_cost`.
        """
        best = float("inf")
        for adj in self.adjacency.values():
            for edge in adj.values():
                if edge.base_cost < best:
                    best = edge.base_cost
        return best if best != float("inf") else self.STANDARD_BASE_COST

    def _init_grid_edges(self) -> None:
        for r in range(self.rows):
            for c in range(self.cols):
                current = (r, c)
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if self.in_bounds((nr, nc)) and (nr, nc) not in self.adjacency[current]:
                        self.add_edge(current, (nr, nc), base_cost=1.0)

    def in_bounds(self, cell: Cell) -> bool:
        r, c = cell
        return 0 <= r < self.rows and 0 <= c < self.cols

    def neighbors(
        self,
        cell: Cell,
        include_blocked: bool = False,
        include_inaccessible: bool = False,
    ) -> List[Cell]:
        result: List[Cell] = []
        for neighbor, edge in self.adjacency[cell].items():
            if not include_blocked and edge.blocked:
                continue
            if not include_inaccessible and not self.nodes[neighbor].accessible:
                continue
            result.append(neighbor)
        return result

    def set_location(self, cell: Cell, location_type: LocationType, population: int = 0) -> None:
        self._ensure_cell(cell)
        node = self.nodes[cell]
        node.location_type = location_type
        node.population = max(0, population)
        self._refresh_incident_base_costs(cell)

    def set_cluster(self, cell: Cell, cluster_id: int) -> None:
        self._ensure_cell(cell)
        self.nodes[cell].cluster_id = cluster_id

    def set_risk(self, cell: Cell, risk_level: RiskLevel) -> None:
        self._ensure_cell(cell)
        self.nodes[cell].risk_level = risk_level
        self.nodes[cell].risk_mult = risk_level.multiplier

    def set_accessible(self, cell: Cell, accessible: bool) -> None:
        self._ensure_cell(cell)
        self.nodes[cell].accessible = bool(accessible)

    def set_police_post(self, cell: Cell, value: bool) -> None:
        self._ensure_cell(cell)
        self.nodes[cell].police_post = bool(value)

    def add_edge(self, a: Cell, b: Cell, base_cost: float = 1.0, blocked: bool = False) -> None:
        self._ensure_cell(a)
        self._ensure_cell(b)
        if base_cost <= 0:
            raise ValueError("base_cost must be positive")
        edge = EdgeData(base_cost=base_cost, blocked=blocked)
        self.adjacency[a][b] = edge
        self.adjacency[b][a] = edge

    def set_edge_blocked(self, a: Cell, b: Cell, blocked: bool) -> None:
        self._ensure_edge(a, b)
        self.adjacency[a][b].blocked = blocked

    def effective_cost(self, a: Cell, b: Cell) -> float:
        self._ensure_edge(a, b)
        edge = self.adjacency[a][b]
        if edge.blocked:
            return float("inf")
        if not self.nodes[a].accessible or not self.nodes[b].accessible:
            return float("inf")
        risk_factor = (self.nodes[a].risk_mult + self.nodes[b].risk_mult) / 2.0
        return edge.base_cost * risk_factor

    def hop_distance(self, source: Cell, target: Cell, max_hops: Optional[int] = None) -> Optional[int]:
        """Unweighted shortest hops using BFS. Returns None when unreachable."""
        self._ensure_cell(source)
        self._ensure_cell(target)
        if source == target:
            return 0

        seen: Set[Cell] = {source}
        queue: deque[Tuple[Cell, int]] = deque([(source, 0)])
        while queue:
            node, dist = queue.popleft()
            if max_hops is not None and dist >= max_hops:
                continue
            for n in self.neighbors(node):
                if n in seen:
                    continue
                if n == target:
                    return dist + 1
                seen.add(n)
                queue.append((n, dist + 1))
        return None

    def shortest_path(self, source: Cell, target: Cell) -> Tuple[List[Cell], float]:
        """Dijkstra path on dynamic effective costs."""
        self._ensure_cell(source)
        self._ensure_cell(target)
        if source == target:
            return [source], 0.0

        dist: Dict[Cell, float] = {source: 0.0}
        prev: Dict[Cell, Cell] = {}
        pq: List[Tuple[float, Cell]] = [(0.0, source)]
        visited: Set[Cell] = set()

        while pq:
            cur_cost, node = heapq.heappop(pq)
            if node in visited:
                continue
            visited.add(node)
            if node == target:
                break
            for n in self.neighbors(node):
                new_cost = cur_cost + self.effective_cost(node, n)
                if new_cost < dist.get(n, float("inf")):
                    dist[n] = new_cost
                    prev[n] = node
                    heapq.heappush(pq, (new_cost, n))

        if target not in dist:
            return [], float("inf")

        path: List[Cell] = [target]
        while path[-1] != source:
            path.append(prev[path[-1]])
        path.reverse()
        return path, dist[target]

    def find_by_type(self, location_type: LocationType) -> List[Cell]:
        return [cell for cell, node in self.nodes.items() if node.location_type == location_type]

    def residential_cells(self) -> List[Cell]:
        return self.find_by_type(LocationType.RESIDENTIAL)

    def _ensure_cell(self, cell: Cell) -> None:
        if cell not in self.nodes:
            raise KeyError(f"Unknown cell {cell}")

    def _ensure_edge(self, a: Cell, b: Cell) -> None:
        self._ensure_cell(a)
        self._ensure_cell(b)
        if b not in self.adjacency[a]:
            raise KeyError(f"No edge between {a} and {b}")

    def _refresh_incident_base_costs(self, cell: Cell) -> None:
        for neighbor in self.adjacency[cell]:
            edge = self.adjacency[cell][neighbor]
            edge.base_cost = self._edge_base_cost(cell, neighbor)

    def _edge_base_cost(self, a: Cell, b: Cell) -> float:
        a_type = self.nodes[a].location_type
        b_type = self.nodes[b].location_type
        if a_type == LocationType.RESIDENTIAL or b_type == LocationType.RESIDENTIAL:
            return 0.8
        return 1.0
