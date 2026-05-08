"""Challenge 3: ambulance placement via Simulated Annealing."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
import random
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .city_graph import Cell, CityGraph, LocationType


@dataclass
class AmbulancePlacementResult:
    positions: Tuple[Cell, Cell, Cell]
    worst_case_distance: float
    iterations: int
    best_iteration: int
    temperature_final: float


class AmbulancePlacementSA:
    def __init__(
        self,
        graph: CityGraph,
        num_ambulances: int = 3,
        iterations: int = 1200,
        cooling_factor: float = 0.95,
        cooling_interval: int = 50,
        temp_floor: float = 0.01,
        jump_probability: float = 0.08,
        seed: int = 23,
    ) -> None:
        if num_ambulances != 3:
            raise ValueError("Challenge 3 requires exactly 3 ambulances")
        self.graph = graph
        self.num_ambulances = num_ambulances
        self.iterations = iterations
        self.cooling_factor = cooling_factor
        self.cooling_interval = cooling_interval
        self.temp_floor = temp_floor
        self.jump_probability = jump_probability
        self.rng = random.Random(seed)
        self.current_positions: Optional[Tuple[Cell, Cell, Cell]] = None

        self.all_cells: List[Cell] = sorted(self.graph.nodes.keys())
        self.citizens: List[Cell] = sorted(self.graph.find_by_type(LocationType.RESIDENTIAL))
        if not self.citizens:
            raise ValueError("No residential cells available for Challenge 3 objective")

    def optimize(self) -> AmbulancePlacementResult:
        current = self._initial_state()
        current_cost = self._objective(current)
        best = current
        best_cost = current_cost
        best_iteration = 0

        temperature = self._estimate_initial_temperature(current, sample_count=60)

        for iteration in range(1, self.iterations + 1):
            if temperature < self.temp_floor:
                break
            candidate = self._neighbor(current)
            candidate_cost = self._objective(candidate)
            delta = candidate_cost - current_cost
            if delta <= 0:
                current, current_cost = candidate, candidate_cost
            else:
                accept_prob = math.exp(-delta / max(temperature, 1e-9))
                if self.rng.random() < accept_prob:
                    current, current_cost = candidate, candidate_cost

            if current_cost < best_cost:
                best, best_cost = current, current_cost
                best_iteration = iteration

            if iteration % self.cooling_interval == 0:
                temperature *= self.cooling_factor

        self.current_positions = best
        return AmbulancePlacementResult(
            positions=best,
            worst_case_distance=best_cost,
            iterations=iteration,
            best_iteration=best_iteration,
            temperature_final=temperature,
        )

    def apply_positions(self, positions: Tuple[Cell, Cell, Cell]) -> None:
        self.current_positions = positions

    def _initial_state(self) -> Tuple[Cell, Cell, Cell]:
        # Seed with high-population residential cells for stronger starting point.
        by_population = sorted(
            self.citizens,
            key=lambda c: self.graph.nodes[c].population,
            reverse=True,
        )
        selected: List[Cell] = []
        for cell in by_population:
            if cell not in selected:
                selected.append(cell)
            if len(selected) == self.num_ambulances:
                break
        while len(selected) < self.num_ambulances:
            c = self.rng.choice(self.all_cells)
            if c not in selected:
                selected.append(c)
        return tuple(selected)  # type: ignore[return-value]

    def _neighbor(self, state: Tuple[Cell, Cell, Cell]) -> Tuple[Cell, Cell, Cell]:
        positions = list(state)
        idx = self.rng.randrange(self.num_ambulances)
        current = positions[idx]

        if self.rng.random() < self.jump_probability:
            candidate = self.rng.choice(self.all_cells)
        else:
            neighbors = self.graph.neighbors(current, include_blocked=True)
            candidate = self.rng.choice(neighbors) if neighbors else current

        # Keep tuple entries unique by swapping if candidate already occupied.
        if candidate in positions and candidate != current:
            other_idx = positions.index(candidate)
            positions[other_idx] = current
        positions[idx] = candidate
        return tuple(positions)  # type: ignore[return-value]

    def _estimate_initial_temperature(self, state: Tuple[Cell, Cell, Cell], sample_count: int) -> float:
        base = self._objective(state)
        uphill_deltas: List[float] = []
        trial_state = state
        for _ in range(sample_count):
            trial_neighbor = self._neighbor(trial_state)
            delta = self._objective(trial_neighbor) - base
            if delta > 0 and math.isfinite(delta):
                uphill_deltas.append(delta)
            trial_state = trial_neighbor
        if not uphill_deltas:
            return 1.0
        avg_uphill = sum(uphill_deltas) / len(uphill_deltas)
        # Target initial acceptance ratio ~0.8: exp(-avg_delta/T0)=0.8
        return max(0.1, avg_uphill / abs(math.log(0.8)))

    def _objective(self, state: Tuple[Cell, Cell, Cell]) -> float:
        dist_maps = [self._dijkstra_distances(src) for src in state]
        worst = 0.0
        for citizen in self.citizens:
            nearest = min(dm.get(citizen, float("inf")) for dm in dist_maps)
            if not math.isfinite(nearest):
                return float("inf")
            worst = max(worst, nearest)
        return worst

    def _dijkstra_distances(self, source: Cell) -> Dict[Cell, float]:
        distances: Dict[Cell, float] = {source: 0.0}
        pq: List[Tuple[float, Cell]] = [(0.0, source)]
        visited: Set[Cell] = set()
        while pq:
            cur_dist, node = heapq.heappop(pq)
            if node in visited:
                continue
            visited.add(node)
            for nxt in self.graph.neighbors(node):
                edge_cost = self.graph.effective_cost(node, nxt)
                if not math.isfinite(edge_cost):
                    continue
                nd = cur_dist + edge_cost
                if nd < distances.get(nxt, float("inf")):
                    distances[nxt] = nd
                    heapq.heappush(pq, (nd, nxt))
        return distances
