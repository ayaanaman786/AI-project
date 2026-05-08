"""Challenge 4: A* routing with admissible Manhattan heuristic."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Dict, List, Tuple

from .city_graph import Cell, CityGraph


@dataclass
class AStarResult:
    path: List[Cell]
    cost: float
    expanded_nodes: int
    found: bool


_ADMISSIBILITY_TOLERANCE: float = 1e-9


class AStarRouter:
    """A* search over the shared CityGraph.

    The heuristic is `Manhattan(a, b) * min_edge_cost`. For admissibility this
    multiplier MUST be <= the smallest possible per-edge cost. The router
    asserts this against `graph.min_base_cost` at search time so that future
    edits to the residential discount cannot silently break optimality.
    """

    def __init__(self, min_edge_cost: float = 0.8) -> None:
        if min_edge_cost <= 0.0:
            raise ValueError("min_edge_cost must be positive")
        self.min_edge_cost = float(min_edge_cost)
        self._tie = 0

    def find_path(self, graph: CityGraph, start: Cell, goal: Cell) -> AStarResult:
        floor = graph.min_base_cost
        if self.min_edge_cost > floor + _ADMISSIBILITY_TOLERANCE:
            raise ValueError(
                "Heuristic floor "
                f"{self.min_edge_cost:.4f} exceeds graph minimum base cost "
                f"{floor:.4f}; the heuristic would be inadmissible. "
                "Lower min_edge_cost to <= the residential discount."
            )
        if start == goal:
            return AStarResult(path=[start], cost=0.0, expanded_nodes=0, found=True)

        g_score: Dict[Cell, float] = {start: 0.0}
        came_from: Dict[Cell, Cell] = {}
        open_heap: List[Tuple[float, int, Cell]] = []
        self._tie = 0
        heapq.heappush(open_heap, (self._heuristic(start, goal), self._next_tie(), start))
        closed = set()
        expanded = 0

        while open_heap:
            _, _, node = heapq.heappop(open_heap)
            if node in closed:
                continue
            closed.add(node)
            expanded += 1

            if node == goal:
                path = self._reconstruct(came_from, goal)
                return AStarResult(path=path, cost=g_score[goal], expanded_nodes=expanded, found=True)

            for nxt in graph.neighbors(node):
                step = graph.effective_cost(node, nxt)
                if not math.isfinite(step):
                    continue
                tentative = g_score[node] + step
                if tentative < g_score.get(nxt, float("inf")):
                    came_from[nxt] = node
                    g_score[nxt] = tentative
                    f = tentative + self._heuristic(nxt, goal)
                    heapq.heappush(open_heap, (f, self._next_tie(), nxt))

        return AStarResult(path=[], cost=float("inf"), expanded_nodes=expanded, found=False)

    def _heuristic(self, a: Cell, b: Cell) -> float:
        return (abs(a[0] - b[0]) + abs(a[1] - b[1])) * self.min_edge_cost

    def _reconstruct(self, came_from: Dict[Cell, Cell], goal: Cell) -> List[Cell]:
        path = [goal]
        while path[-1] in came_from:
            path.append(came_from[path[-1]])
        path.reverse()
        return path

    def _next_tie(self) -> int:
        self._tie += 1
        return self._tie
