"""Challenge 1: city layout planning with CSP + Min-Conflicts fallback."""

from __future__ import annotations

from collections import Counter, deque
import random
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Set, Tuple

from .city_graph import Cell, CityGraph, LocationType


# Stable identifiers for each constraint rule. These are used as keys in
# `LayoutResult.violation_breakdown` so the UI and tests can show per-rule
# counts independent of the human-readable label below.
RULE_INDUSTRIAL_ADJACENT_TO_PROTECTED = "industrial_adjacent_to_protected"
RULE_RESIDENTIAL_HOSPITAL_REACH = "residential_hospital_reach"
RULE_POWER_INDUSTRIAL_REACH = "power_industrial_reach"
RULE_QUOTA_MISMATCH = "quota_mismatch"

RULE_LABELS: Dict[str, str] = {
    RULE_INDUSTRIAL_ADJACENT_TO_PROTECTED: "Industrial next to Hospital/School",
    RULE_RESIDENTIAL_HOSPITAL_REACH: "Residential farther than 3 hops from hospital",
    RULE_POWER_INDUSTRIAL_REACH: "Power plant farther than 2 hops from industrial",
    RULE_QUOTA_MISMATCH: "Quota mismatch",
}


@dataclass
class LayoutResult:
    assignment: Dict[Cell, LocationType]
    success: bool
    violated_rule: Optional[str] = None
    violations: int = 0
    violation_breakdown: Dict[str, int] = field(default_factory=dict)


class LayoutPlannerCSP:
    DEFAULT_INDUSTRIAL_FORBIDDEN_NEIGHBORS: Set[LocationType] = frozenset(
        {LocationType.HOSPITAL, LocationType.SCHOOL}
    )

    def __init__(
        self,
        graph: CityGraph,
        quotas: Optional[Dict[LocationType, int]] = None,
        seed: int = 7,
        hospital_max_hops: int = 3,
        industrial_max_hops_for_power: int = 2,
        industrial_forbidden_neighbors: Optional[Set[LocationType]] = None,
        max_backtrack_steps: int = 15000,
    ) -> None:
        if hospital_max_hops < 1:
            raise ValueError("hospital_max_hops must be >= 1")
        if industrial_max_hops_for_power < 1:
            raise ValueError("industrial_max_hops_for_power must be >= 1")

        self.graph = graph
        self.rng = random.Random(seed)
        self.cells: List[Cell] = list(graph.nodes.keys())
        total_cells = len(self.cells)
        # Keep default quotas feasible for the hospital reach rule.
        industrial_quota = max(6, int(total_cells * 0.15))
        hospital_quota = max(6, int(total_cells * 0.10))
        school_quota = max(2, int(total_cells * 0.05))
        power_quota = max(2, int(total_cells * 0.05))
        depot_quota = 1
        fixed = industrial_quota + hospital_quota + school_quota + power_quota + depot_quota
        residential_quota = max(1, total_cells - fixed)
        self.quotas = quotas or {
            LocationType.HOSPITAL: hospital_quota,
            LocationType.SCHOOL: school_quota,
            LocationType.INDUSTRIAL: industrial_quota,
            LocationType.POWER_PLANT: power_quota,
            LocationType.AMBULANCE_DEPOT: depot_quota,
            LocationType.RESIDENTIAL: residential_quota,
        }
        self.location_types: List[LocationType] = list(self.quotas.keys())
        self.initial_domains: Dict[Cell, Set[LocationType]] = {
            cell: set(self.location_types) for cell in self.cells
        }
        self.hospital_max_hops = hospital_max_hops
        self.industrial_max_hops_for_power = industrial_max_hops_for_power
        self.industrial_forbidden_neighbors: Set[LocationType] = set(
            industrial_forbidden_neighbors
            if industrial_forbidden_neighbors is not None
            else self.DEFAULT_INDUSTRIAL_FORBIDDEN_NEIGHBORS
        )
        self.hospital_reach: Dict[Cell, List[Cell]] = {
            cell: self._cells_within_hops(cell, hospital_max_hops) for cell in self.cells
        }
        self.industrial_reach: Dict[Cell, List[Cell]] = {
            cell: self._cells_within_hops(cell, industrial_max_hops_for_power)
            for cell in self.cells
        }
        self._backtrack_steps = 0
        self.max_backtrack_steps = max_backtrack_steps

    # Backward-compat aliases for the legacy attribute names used by older
    # tests and external callers prior to the rule parameterization.
    @property
    def hospital_reach_3(self) -> Dict[Cell, List[Cell]]:
        return self.hospital_reach

    @hospital_reach_3.setter
    def hospital_reach_3(self, value: Dict[Cell, List[Cell]]) -> None:
        self.hospital_reach = value

    @property
    def industrial_reach_2(self) -> Dict[Cell, List[Cell]]:
        return self.industrial_reach

    @industrial_reach_2.setter
    def industrial_reach_2(self, value: Dict[Cell, List[Cell]]) -> None:
        self.industrial_reach = value

    def solve(self) -> LayoutResult:
        # Full backtracking on 10x10 is expensive; cap it and fall back to min-conflicts.
        self._backtrack_steps = 0
        domains = self._ac3(self.initial_domains)
        assignment = None
        if domains is not None:
            quotas_remaining = dict(self.quotas)
            assignment = self._backtracking({}, domains, quotas_remaining)
        if assignment is not None:
            self._apply_layout(assignment)
            return LayoutResult(
                assignment=assignment,
                success=True,
                violation_breakdown={},
            )

        fallback = self._min_conflicts(max_steps=5000)
        self._apply_layout(fallback.assignment)
        return fallback

    def _backtracking(
        self,
        assignment: Dict[Cell, LocationType],
        domains: Dict[Cell, Set[LocationType]],
        quotas_remaining: Dict[LocationType, int],
    ) -> Optional[Dict[Cell, LocationType]]:
        self._backtrack_steps += 1
        if self._backtrack_steps >= self.max_backtrack_steps:
            return None
        if len(assignment) == len(self.cells):
            violations, _, _ = self._count_violations(assignment)
            return assignment if violations == 0 else None

        cell = self._select_unassigned_cell(assignment, domains)
        for value in self._ordered_values(cell, assignment, domains):
            if quotas_remaining[value] <= 0:
                continue
            assignment[cell] = value
            next_domains = {c: set(vs) for c, vs in domains.items()}
            next_quotas = dict(quotas_remaining)
            next_quotas[value] -= 1
            next_domains[cell] = {value}
            if self._is_partial_valid(assignment, cell) and self._forward_check(
                assignment, next_domains, next_quotas, cell
            ):
                solved = self._backtracking(assignment, next_domains, next_quotas)
                if solved is not None:
                    return solved
            del assignment[cell]
        return None

    def _select_unassigned_cell(
        self,
        assignment: Dict[Cell, LocationType],
        domains: Dict[Cell, Set[LocationType]],
    ) -> Cell:
        unassigned = [c for c in self.cells if c not in assignment]
        if not unassigned:
            raise RuntimeError("No unassigned cells left")
        # MRV + degree tie-break.
        return min(
            unassigned,
            key=lambda c: (
                len(domains[c]),
                -sum(1 for n in self.graph.neighbors(c, include_blocked=True) if n not in assignment),
            ),
        )

    def _ordered_values(
        self,
        cell: Cell,
        assignment: Dict[Cell, LocationType],
        domains: Dict[Cell, Set[LocationType]],
    ) -> List[LocationType]:
        # LCV: value that eliminates the fewest neighbor choices first.
        def elimination_score(value: LocationType) -> int:
            score = 0
            for n in self.graph.neighbors(cell, include_blocked=True):
                if n in assignment:
                    continue
                for neighbor_value in domains[n]:
                    if not self._pair_consistent(value, neighbor_value):
                        score += 1
            return score

        return sorted(domains[cell], key=elimination_score)

    def _ac3(self, domains: Dict[Cell, Set[LocationType]]) -> Optional[Dict[Cell, Set[LocationType]]]:
        domains = {c: set(vs) for c, vs in domains.items()}
        queue: Deque[Tuple[Cell, Cell]] = deque()
        for xi in self.cells:
            for xj in self.graph.neighbors(xi, include_blocked=True):
                queue.append((xi, xj))

        while queue:
            xi, xj = queue.popleft()
            if self._revise(domains, xi, xj):
                if not domains[xi]:
                    return None
                for xk in self.graph.neighbors(xi, include_blocked=True):
                    if xk != xj:
                        queue.append((xk, xi))
        return domains

    def _revise(self, domains: Dict[Cell, Set[LocationType]], xi: Cell, xj: Cell) -> bool:
        revised = False
        to_remove: Set[LocationType] = set()
        for vi in domains[xi]:
            if not any(self._pair_consistent(vi, vj) for vj in domains[xj]):
                to_remove.add(vi)
        if to_remove:
            domains[xi] -= to_remove
            revised = True
        return revised

    def _pair_consistent(self, a: LocationType, b: LocationType) -> bool:
        disallowed = self.industrial_forbidden_neighbors
        if a == LocationType.INDUSTRIAL and b in disallowed:
            return False
        if b == LocationType.INDUSTRIAL and a in disallowed:
            return False
        return True

    def _is_partial_valid(self, assignment: Dict[Cell, LocationType], changed_cell: Cell) -> bool:
        value = assignment[changed_cell]
        if value == LocationType.INDUSTRIAL:
            for n in self.graph.neighbors(changed_cell, include_blocked=True):
                nv = assignment.get(n)
                if nv in self.industrial_forbidden_neighbors:
                    return False
        if value in self.industrial_forbidden_neighbors:
            for n in self.graph.neighbors(changed_cell, include_blocked=True):
                if assignment.get(n) == LocationType.INDUSTRIAL:
                    return False
        if value == LocationType.RESIDENTIAL:
            if not self._has_assigned_or_possible_hospital(changed_cell, assignment):
                return False
        if value == LocationType.POWER_PLANT:
            if not self._has_assigned_or_possible_industrial(changed_cell, assignment):
                return False
        return True

    def _forward_check(
        self,
        assignment: Dict[Cell, LocationType],
        domains: Dict[Cell, Set[LocationType]],
        quotas_remaining: Dict[LocationType, int],
        changed_cell: Cell,
    ) -> bool:
        # Enforce adjacency constraints into neighbor domains.
        assigned_value = assignment[changed_cell]
        for neighbor in self.graph.neighbors(changed_cell, include_blocked=True):
            if neighbor in assignment:
                continue
            domains[neighbor] = {
                val for val in domains[neighbor] if self._pair_consistent(assigned_value, val)
            }
            if not domains[neighbor]:
                return False

        # Enforce quotas by removing exhausted types from unassigned domains.
        exhausted = {t for t, remaining in quotas_remaining.items() if remaining <= 0}
        for cell in self.cells:
            if cell in assignment:
                continue
            if exhausted:
                domains[cell] -= exhausted
            if not domains[cell]:
                return False

        # Global feasibility check: remaining quotas must fit remaining domain capacity.
        unassigned = [c for c in self.cells if c not in assignment]
        for t, remaining in quotas_remaining.items():
            if remaining < 0:
                return False
            possible_slots = sum(1 for c in unassigned if t in domains[c])
            if possible_slots < remaining:
                return False

        # Validate still-feasible reachability constraints for assigned global nodes.
        for cell, t in assignment.items():
            if t == LocationType.RESIDENTIAL and not self._has_assigned_or_possible_hospital(
                cell, assignment, domains
            ):
                return False
            if t == LocationType.POWER_PLANT and not self._has_assigned_or_possible_industrial(
                cell, assignment, domains
            ):
                return False
        return True

    def _has_assigned_or_possible_hospital(
        self,
        residential_cell: Cell,
        assignment: Dict[Cell, LocationType],
        domains: Optional[Dict[Cell, Set[LocationType]]] = None,
    ) -> bool:
        for cell in self.hospital_reach[residential_cell]:
            if assignment.get(cell) == LocationType.HOSPITAL:
                return True
            if domains is not None and cell not in assignment and LocationType.HOSPITAL in domains[cell]:
                return True
        return False

    def _has_assigned_or_possible_industrial(
        self,
        power_cell: Cell,
        assignment: Dict[Cell, LocationType],
        domains: Optional[Dict[Cell, Set[LocationType]]] = None,
    ) -> bool:
        for cell in self.industrial_reach[power_cell]:
            if assignment.get(cell) == LocationType.INDUSTRIAL:
                return True
            if domains is not None and cell not in assignment and LocationType.INDUSTRIAL in domains[cell]:
                return True
        return False

    def _cells_within_hops(self, source: Cell, max_hops: int) -> List[Cell]:
        seen: Set[Cell] = {source}
        queue: Deque[Tuple[Cell, int]] = deque([(source, 0)])
        cells: List[Cell] = [source]
        while queue:
            node, hops = queue.popleft()
            if hops == max_hops:
                continue
            for n in self.graph.neighbors(node, include_blocked=True):
                if n in seen:
                    continue
                seen.add(n)
                cells.append(n)
                queue.append((n, hops + 1))
        return cells

    def _min_conflicts(self, max_steps: int) -> LayoutResult:
        assignment: Dict[Cell, LocationType] = {}
        cells = self.cells[:]
        self.rng.shuffle(cells)

        pool: List[LocationType] = []
        for t, count in self.quotas.items():
            pool.extend([t] * count)
        while len(pool) < len(cells):
            pool.append(LocationType.RESIDENTIAL)
        self.rng.shuffle(pool)
        for cell, t in zip(cells, pool):
            assignment[cell] = t

        for _ in range(max_steps):
            violations, rule, breakdown = self._count_violations(assignment)
            if violations == 0:
                return LayoutResult(
                    assignment=assignment,
                    success=True,
                    violation_breakdown={},
                )

            conflicted = [c for c in cells if self._cell_conflicts(c, assignment)]
            if not conflicted:
                break
            cell = self.rng.choice(conflicted)
            best_swap = cell
            best_score = float("inf")
            for other in cells:
                if other == cell:
                    continue
                assignment[cell], assignment[other] = assignment[other], assignment[cell]
                score, _, _ = self._count_violations(assignment)
                if score < best_score:
                    best_score = score
                    best_swap = other
                assignment[cell], assignment[other] = assignment[other], assignment[cell]
            assignment[cell], assignment[best_swap] = assignment[best_swap], assignment[cell]

        violations, rule, breakdown = self._count_violations(assignment)
        return LayoutResult(
            assignment=assignment,
            success=violations == 0,
            violated_rule=rule,
            violations=violations,
            violation_breakdown=breakdown,
        )

    def _cell_conflicts(self, cell: Cell, assignment: Dict[Cell, LocationType]) -> bool:
        my_type = assignment[cell]
        if my_type == LocationType.INDUSTRIAL:
            for n in self.graph.neighbors(cell, include_blocked=True):
                if assignment.get(n) in self.industrial_forbidden_neighbors:
                    return True
        if my_type in self.industrial_forbidden_neighbors:
            for n in self.graph.neighbors(cell, include_blocked=True):
                if assignment.get(n) == LocationType.INDUSTRIAL:
                    return True
        return False

    def _count_violations(
        self, assignment: Dict[Cell, LocationType]
    ) -> Tuple[int, Optional[str], Dict[str, int]]:
        # Per-rule counters are stable identifiers; a human-readable summary
        # rule is chosen as the dominant violation for backward compatibility.
        breakdown: Dict[str, int] = {}

        def bump(key: str, amount: int = 1) -> None:
            if amount <= 0:
                return
            breakdown[key] = breakdown.get(key, 0) + amount

        for cell, location_type in assignment.items():
            if location_type == LocationType.INDUSTRIAL:
                for n in self.graph.neighbors(cell, include_blocked=True):
                    if assignment.get(n) in self.industrial_forbidden_neighbors:
                        bump(RULE_INDUSTRIAL_ADJACENT_TO_PROTECTED)

        hospitals = {c for c, t in assignment.items() if t == LocationType.HOSPITAL}
        industrial = {c for c, t in assignment.items() if t == LocationType.INDUSTRIAL}
        for cell, location_type in assignment.items():
            if location_type == LocationType.RESIDENTIAL and hospitals:
                if not any(c in hospitals for c in self.hospital_reach[cell]):
                    bump(RULE_RESIDENTIAL_HOSPITAL_REACH)
            if location_type == LocationType.POWER_PLANT and industrial:
                if not any(c in industrial for c in self.industrial_reach[cell]):
                    bump(RULE_POWER_INDUSTRIAL_REACH)

        quota_used = Counter(assignment.values())
        for t, required in self.quotas.items():
            mismatch = abs(quota_used[t] - required)
            if mismatch:
                bump(RULE_QUOTA_MISMATCH, mismatch)

        violations = sum(breakdown.values())
        if not breakdown:
            return 0, None, {}
        dominant_key = max(breakdown.items(), key=lambda kv: kv[1])[0]
        return violations, RULE_LABELS.get(dominant_key, dominant_key), dict(breakdown)

    def _apply_layout(self, assignment: Dict[Cell, LocationType]) -> None:
        for cell, location_type in assignment.items():
            population = self.rng.randint(40, 200) if location_type == LocationType.RESIDENTIAL else 0
            self.graph.set_location(cell, location_type, population=population)
