"""Challenge 5 follow-up: deploy 10 police officers using predicted risk."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set

from .city_graph import Cell, CityGraph


@dataclass
class PoliceDeploymentResult:
    positions: List[Cell]
    total_covered_risk: float
    coverage_radius: int
    coverage_by_post: Dict[Cell, List[Cell]]


class PoliceDeploymentPlanner:
    """Greedy maximum-coverage deployment driven by node risk multipliers."""

    def __init__(
        self,
        graph: CityGraph,
        num_officers: int = 10,
        coverage_radius: int = 2,
        min_spacing: int = 1,
    ) -> None:
        if num_officers <= 0:
            raise ValueError("num_officers must be positive")
        if coverage_radius < 0:
            raise ValueError("coverage_radius must be non-negative")
        if min_spacing < 0:
            raise ValueError("min_spacing must be non-negative")
        self.graph = graph
        self.num_officers = num_officers
        self.coverage_radius = coverage_radius
        self.min_spacing = min_spacing

    def plan(self, candidates: Optional[Sequence[Cell]] = None) -> PoliceDeploymentResult:
        cells = self._candidate_cells(candidates)
        coverage_map = self._build_coverage_map(cells)
        risk_by_cell = {cell: self.graph.nodes[cell].risk_mult for cell in cells}

        chosen: List[Cell] = []
        covered: Set[Cell] = set()
        forbidden: Set[Cell] = set()
        coverage_by_post: Dict[Cell, List[Cell]] = {}

        for _ in range(self.num_officers):
            best_cell: Optional[Cell] = None
            best_gain = -1.0
            best_tiebreak: float = float("inf")
            for cell in cells:
                if cell in forbidden:
                    continue
                area = coverage_map[cell]
                gain = sum(risk_by_cell[c] for c in area if c not in covered)
                if gain > best_gain or (gain == best_gain and risk_by_cell[cell] > best_tiebreak):
                    best_cell = cell
                    best_gain = gain
                    best_tiebreak = risk_by_cell[cell]
            if best_cell is None or best_gain <= 0:
                break
            chosen.append(best_cell)
            self.graph.set_police_post(best_cell, True)
            new_area = coverage_map[best_cell]
            covered.update(new_area)
            coverage_by_post[best_cell] = sorted(new_area)
            for nearby in self._cells_within_hops(best_cell, self.min_spacing):
                forbidden.add(nearby)

        total_covered = sum(risk_by_cell[c] for c in covered)
        return PoliceDeploymentResult(
            positions=chosen,
            total_covered_risk=total_covered,
            coverage_radius=self.coverage_radius,
            coverage_by_post=coverage_by_post,
        )

    def _candidate_cells(self, candidates: Optional[Sequence[Cell]]) -> List[Cell]:
        if candidates is not None:
            return list(candidates)
        return sorted(
            cell for cell, node in self.graph.nodes.items() if node.accessible
        )

    def _build_coverage_map(self, cells: Sequence[Cell]) -> Dict[Cell, List[Cell]]:
        return {cell: self._cells_within_hops(cell, self.coverage_radius) for cell in cells}

    def _cells_within_hops(self, source: Cell, max_hops: int) -> List[Cell]:
        if max_hops == 0:
            return [source]
        seen: Set[Cell] = {source}
        frontier: List[Cell] = [source]
        for _ in range(max_hops):
            next_frontier: List[Cell] = []
            for node in frontier:
                for neighbor in self.graph.neighbors(node, include_blocked=True):
                    if neighbor in seen:
                        continue
                    seen.add(neighbor)
                    next_frontier.append(neighbor)
            frontier = next_frontier
            if not frontier:
                break
        return sorted(seen)
