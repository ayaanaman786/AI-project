"""Challenge 4 mission simulation with dynamic A* replanning."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import random
from typing import List, Optional, Sequence, Set, Tuple

from .astar_router import AStarRouter
from .city_graph import Cell, CityGraph, LocationType, RiskLevel

Edge = Tuple[Cell, Cell]


@dataclass
class MissionState:
    team_pos: Cell
    civilians_ordered: List[Cell]
    target_index: int = 0
    current_path: List[Cell] = field(default_factory=list)
    event_log: List[str] = field(default_factory=list)
    completed: bool = False

    @property
    def current_target(self) -> Optional[Cell]:
        if self.target_index >= len(self.civilians_ordered):
            return None
        return self.civilians_ordered[self.target_index]


class FloodEventManager:
    def __init__(
        self,
        graph: CityGraph,
        flood_probability: float = 0.12,
        seed: int = 101,
        max_blocks_per_tick: int = 1,
    ) -> None:
        if max_blocks_per_tick < 0:
            raise ValueError("max_blocks_per_tick must be non-negative")
        self.graph = graph
        self.flood_probability = flood_probability
        self.rng = random.Random(seed)
        self.max_blocks_per_tick = max_blocks_per_tick

    def apply_tick(self) -> List[Edge]:
        newly_blocked: List[Edge] = []
        if self.max_blocks_per_tick == 0:
            return newly_blocked
        for a in sorted(self.graph.nodes.keys()):
            if len(newly_blocked) >= self.max_blocks_per_tick:
                break
            for b in sorted(self.graph.neighbors(a, include_blocked=True)):
                if a >= b:
                    continue
                edge = self.graph.adjacency[a][b]
                if edge.blocked:
                    continue
                if self.rng.random() < self.flood_probability:
                    self.graph.set_edge_blocked(a, b, True)
                    newly_blocked.append((a, b))
                    if len(newly_blocked) >= self.max_blocks_per_tick:
                        break
        return newly_blocked


class EmergencyMissionRunner:
    def __init__(
        self,
        graph: CityGraph,
        team_start: Cell,
        civilians_ordered: Sequence[Cell],
        flood_probability: float = 0.12,
        router: Optional[AStarRouter] = None,
        seed: int = 101,
        risk_shift_probability: float = 0.35,
        sa_reopt_threshold: int = 2,
        sa_optimizer: Optional[object] = None,
        max_blocks_per_tick: int = 1,
        repair_after_failures: int = 2,
    ) -> None:
        if not civilians_ordered:
            raise ValueError("Mission requires at least one civilian target")
        if repair_after_failures < 1:
            raise ValueError("repair_after_failures must be >= 1")
        self.graph = graph
        self.state = MissionState(team_pos=team_start, civilians_ordered=list(civilians_ordered))
        self.router = router or AStarRouter()
        self.events = FloodEventManager(
            graph,
            flood_probability=flood_probability,
            seed=seed,
            max_blocks_per_tick=max_blocks_per_tick,
        )
        self.risk_shift_probability = risk_shift_probability
        self.sa_reopt_threshold = sa_reopt_threshold
        self.sa_optimizer = sa_optimizer
        self.repair_after_failures = repair_after_failures
        self._failure_streak = 0

    def run(self, max_steps: int = 20) -> MissionState:
        for tick in range(1, max_steps + 1):
            if self.state.completed:
                break
            self.step(tick)
        return self.state

    def step(self, tick: int) -> None:
        blocked_now = self.events.apply_tick()
        for a, b in blocked_now:
            self.state.event_log.append(f"t={tick:02d} flood: edge {a}-{b} blocked")
        risk_updates = self._apply_risk_shifts(blocked_now)
        if risk_updates:
            self.state.event_log.append(f"t={tick:02d} risk updates: {risk_updates} cells shifted")
        if risk_updates >= self.sa_reopt_threshold and self.sa_optimizer is not None:
            result = self.sa_optimizer.optimize()
            self.sa_optimizer.apply_positions(result.positions)
            self.state.event_log.append(f"t={tick:02d} SA re-eval: positions={result.positions}")

        target = self.state.current_target
        if target is None:
            self.state.completed = True
            return

        need_replan = not self.state.current_path or not self._path_is_still_valid(self.state.current_path)
        if need_replan:
            result = self.router.find_path(self.graph, self.state.team_pos, target)
            if not result.found:
                self._failure_streak += 1
                if self._failure_streak == 1:
                    self.state.event_log.append(
                        f"t={tick:02d} replan failed: {self.state.team_pos}->{target} (waiting)"
                    )
                if self._failure_streak >= self.repair_after_failures:
                    repaired = self._attempt_repair(target)
                    if repaired is not None:
                        a, b = repaired
                        self.state.event_log.append(
                            f"t={tick:02d} repair: edge {a}-{b} restored"
                        )
                        self._failure_streak = 0
                return
            self._failure_streak = 0
            self.state.current_path = result.path
            self.state.event_log.append(
                f"t={tick:02d} A* replan: {self.state.team_pos}->{target} cost={result.cost:.2f}"
            )

        if len(self.state.current_path) > 1:
            next_node = self.state.current_path[1]
            self.state.team_pos = next_node
            self.state.current_path = self.state.current_path[1:]
            self.state.event_log.append(f"t={tick:02d} move: team->{next_node}")

        if self.state.team_pos == target:
            self.state.event_log.append(f"t={tick:02d} reached civilian: {target}")
            self.state.target_index += 1
            self.state.current_path = []
            if self.state.current_target is None:
                self.state.completed = True
                self.state.event_log.append(f"t={tick:02d} mission complete")

    def _path_is_still_valid(self, path: Sequence[Cell]) -> bool:
        if len(path) <= 1:
            return False
        a, b = path[0], path[1]
        return b in self.graph.neighbors(a)

    def _bfs_reachable(self, source: Cell) -> Set[Cell]:
        """Cells reachable from source via currently unblocked + accessible edges."""
        seen: Set[Cell] = {source}
        queue: deque = deque([source])
        while queue:
            node = queue.popleft()
            for n in self.graph.neighbors(node):
                if n in seen:
                    continue
                seen.add(n)
                queue.append(n)
        return seen

    def _bfs_hop_count(self, source: Cell, target: Cell) -> Optional[int]:
        if source == target:
            return 0
        seen: Set[Cell] = {source}
        queue: deque = deque([(source, 0)])
        while queue:
            node, dist = queue.popleft()
            for n in self.graph.neighbors(node):
                if n in seen:
                    continue
                if n == target:
                    return dist + 1
                seen.add(n)
                queue.append((n, dist + 1))
        return None

    def _attempt_repair(self, target: Cell) -> Optional[Edge]:
        """Unblock the single blocked edge that minimizes team_pos -> target hops.

        Tie-break by lowest base_cost. Returns the (a, b) repaired edge, or None
        if no candidate edge is on the frontier of the team's reachable set.
        """
        reachable = self._bfs_reachable(self.state.team_pos)

        candidates: List[Edge] = []
        for a in sorted(self.graph.nodes.keys()):
            for b in sorted(self.graph.adjacency[a].keys()):
                if a >= b:
                    continue
                edge = self.graph.adjacency[a][b]
                if not edge.blocked:
                    continue
                a_in = a in reachable
                b_in = b in reachable
                if a_in == b_in:
                    continue
                if not self.graph.nodes[a].accessible or not self.graph.nodes[b].accessible:
                    continue
                candidates.append((a, b))

        if not candidates:
            return None

        best: Optional[Edge] = None
        best_hops = float("inf")
        best_base_cost = float("inf")
        for a, b in candidates:
            edge = self.graph.adjacency[a][b]
            edge.blocked = False
            try:
                hops = self._bfs_hop_count(self.state.team_pos, target)
            finally:
                edge.blocked = True
            if hops is None:
                continue
            if hops < best_hops or (hops == best_hops and edge.base_cost < best_base_cost):
                best = (a, b)
                best_hops = hops
                best_base_cost = edge.base_cost

        if best is None:
            return None

        a, b = best
        self.graph.set_edge_blocked(a, b, False)
        return (a, b)

    def _apply_risk_shifts(self, blocked_now: Sequence[Edge]) -> int:
        changed = 0
        touched: set[Cell] = set()
        for a, b in blocked_now:
            touched.add(a)
            touched.add(b)
        for cell in sorted(touched):
            if self.events.rng.random() >= self.risk_shift_probability:
                continue
            node = self.graph.nodes[cell]
            next_level = self._escalate(node.risk_level)
            if next_level != node.risk_level:
                self.graph.set_risk(cell, next_level)
                changed += 1
        return changed

    def _escalate(self, level: RiskLevel) -> RiskLevel:
        if level == RiskLevel.LOW:
            return RiskLevel.MEDIUM
        if level == RiskLevel.MEDIUM:
            return RiskLevel.HIGH
        return RiskLevel.HIGH


def default_mission(graph: CityGraph, civilian_count: int = 4, seed: int = 17) -> Tuple[Cell, List[Cell]]:
    hospitals = sorted(graph.find_by_type(LocationType.HOSPITAL))
    if not hospitals:
        raise ValueError("No hospital found for mission start")
    civilians = sorted(graph.find_by_type(LocationType.RESIDENTIAL))
    if len(civilians) < civilian_count:
        civilian_count = len(civilians)
    start = hospitals[0]
    if civilian_count == 0:
        return start, []

    # Keep mission generation deterministic but less brittle for short demo
    # horizons: always include the nearest civilian to the starting hospital
    # as the first target, then sample the remaining targets via seeded RNG.
    # This preserves variability across seeds while making the baseline
    # "reach at least one civilian in 20 steps" scenario robust.
    nearest = min(
        civilians,
        key=lambda c: (graph.hop_distance(start, c, max_hops=None) or 10**9, c),
    )
    remaining_pool = [c for c in civilians if c != nearest]
    rng = random.Random(seed)
    remaining_count = max(0, civilian_count - 1)
    remainder = rng.sample(remaining_pool, remaining_count) if remaining_count else []
    ordered = [nearest, *remainder]
    return start, ordered
