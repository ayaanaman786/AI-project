"""Challenge 2: road network optimization with a Genetic Algorithm."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import random
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .city_graph import Cell, CityGraph, LocationType

Edge = Tuple[Cell, Cell]
Chromosome = List[int]


@dataclass
class RoadNetworkResult:
    selected_edges: List[Edge]
    total_cost: float
    component_count: int
    has_hospital_depot_redundancy: bool
    best_fitness: float
    generation: int
    generations_run: int = 0
    stopped_early: bool = False


class RoadNetworkGA:
    def __init__(
        self,
        graph: CityGraph,
        population_size: int = 60,
        generations: int = 120,
        mutation_rate: Optional[float] = None,
        connectivity_penalty_weight: float = 10_000.0,
        redundancy_penalty_weight: float = 8_000.0,
        tournament_size: int = 3,
        use_effective_cost: bool = False,
        seed: int = 11,
        patience: int = 25,
        min_generations: int = 30,
    ) -> None:
        if patience < 1:
            raise ValueError("patience must be >= 1")
        if min_generations < 1:
            raise ValueError("min_generations must be >= 1")
        self.graph = graph
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.connectivity_penalty_weight = connectivity_penalty_weight
        self.redundancy_penalty_weight = redundancy_penalty_weight
        self.use_effective_cost = use_effective_cost
        self.patience = patience
        self.min_generations = min_generations
        self.rng = random.Random(seed)

        self.nodes: List[Cell] = sorted(self.graph.nodes.keys())
        self.candidate_edges: List[Edge] = self._build_candidate_edges()
        if not self.candidate_edges:
            raise ValueError("No candidate edges available")
        self.edge_index: Dict[Edge, int] = {edge: i for i, edge in enumerate(self.candidate_edges)}
        self.mutation_rate = mutation_rate if mutation_rate is not None else 1.0 / len(self.candidate_edges)

        hospitals = sorted(self.graph.find_by_type(LocationType.HOSPITAL))
        depots = sorted(self.graph.find_by_type(LocationType.AMBULANCE_DEPOT))
        if not hospitals:
            raise ValueError("No hospital found in graph for Challenge 2")
        if not depots:
            raise ValueError("No ambulance depot found in graph for Challenge 2")
        self.primary_hospital = hospitals[0]
        self.ambulance_depot = depots[0]

    def optimize(self) -> RoadNetworkResult:
        population = self._initial_population()
        best_chromosome = population[0]
        best_fitness = float("-inf")
        best_generation = 0
        last_completed_generation = 0
        stopped_early = False

        for generation in range(self.generations):
            fitnesses = [self._fitness(chrom) for chrom in population]
            for chrom, fit in zip(population, fitnesses):
                if fit > best_fitness:
                    best_fitness = fit
                    best_chromosome = chrom[:]
                    best_generation = generation

            last_completed_generation = generation
            # Stagnation-based early stop. We always run at least
            # min_generations to avoid breaking out of stochastic dips before
            # the search has had a chance to mix; after that, if no new best
            # has been found in `patience` generations we exit early. The
            # exit point is fully determined by the seed, so determinism is
            # preserved for any fixed (seed, patience, min_generations).
            if (
                generation + 1 >= self.min_generations
                and generation - best_generation >= self.patience
                and self._is_feasible(best_chromosome)
            ):
                stopped_early = True
                break

            elite_idx = max(range(len(population)), key=lambda i: fitnesses[i])
            elite = population[elite_idx][:]
            next_population: List[Chromosome] = [elite]

            while len(next_population) < self.population_size:
                p1 = self._tournament_select(population, fitnesses)
                p2 = self._tournament_select(population, fitnesses)
                c1, c2 = self._uniform_crossover(p1, p2)
                self._mutate(c1)
                self._mutate(c2)
                next_population.append(c1)
                if len(next_population) < self.population_size:
                    next_population.append(c2)
            population = next_population

        selected_edges = self._decode_edges(best_chromosome)
        components = self._component_count(selected_edges)
        redundancy = self._has_hospital_depot_redundancy(selected_edges)
        total_cost = self._total_cost(selected_edges)
        return RoadNetworkResult(
            selected_edges=selected_edges,
            total_cost=total_cost,
            component_count=components,
            has_hospital_depot_redundancy=redundancy,
            best_fitness=best_fitness,
            generation=best_generation,
            generations_run=last_completed_generation + 1,
            stopped_early=stopped_early,
        )

    def apply_selected_roads(self, selected_edges: Sequence[Edge]) -> None:
        selected = {self._normalize_edge(a, b) for a, b in selected_edges}
        for a, b in self.candidate_edges:
            self.graph.set_edge_blocked(a, b, self._normalize_edge(a, b) not in selected)

    def _build_candidate_edges(self) -> List[Edge]:
        edges: Set[Edge] = set()
        for a in self.graph.nodes:
            for b in self.graph.neighbors(a, include_blocked=True):
                edges.add(self._normalize_edge(a, b))
        return sorted(edges)

    def _initial_population(self) -> List[Chromosome]:
        half = self.population_size // 2
        seeded = [self._seeded_chromosome() for _ in range(half)]
        randoms = [self._random_chromosome() for _ in range(self.population_size - half)]
        return seeded + randoms

    def _seeded_chromosome(self) -> Chromosome:
        mst_edges = self._mst_edges()
        chrom = [0] * len(self.candidate_edges)
        for edge in mst_edges:
            chrom[self.edge_index[edge]] = 1

        # Add a few cheap extra edges to seed redundancy-friendly individuals.
        remaining = [e for e in self.candidate_edges if chrom[self.edge_index[e]] == 0]
        remaining.sort(key=self._edge_cost)
        extra_count = max(1, len(self.nodes) // 20)
        for edge in remaining[:extra_count]:
            chrom[self.edge_index[edge]] = 1
        return chrom

    def _random_chromosome(self) -> Chromosome:
        return [1 if self.rng.random() < 0.45 else 0 for _ in self.candidate_edges]

    def _mst_edges(self) -> List[Edge]:
        parent: Dict[Cell, Cell] = {n: n for n in self.nodes}

        def find(x: Cell) -> Cell:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: Cell, y: Cell) -> bool:
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            parent[ry] = rx
            return True

        edges_sorted = sorted(self.candidate_edges, key=self._edge_cost)
        mst: List[Edge] = []
        for edge in edges_sorted:
            a, b = edge
            if union(a, b):
                mst.append(edge)
            if len(mst) == len(self.nodes) - 1:
                break
        return mst

    def _fitness(self, chromosome: Chromosome) -> float:
        selected_edges = self._decode_edges(chromosome)
        total_cost = self._total_cost(selected_edges)
        components = self._component_count(selected_edges)
        connectivity_penalty = self.connectivity_penalty_weight * max(0, components - 1)
        redundancy_penalty = 0.0
        if not self._has_hospital_depot_redundancy(selected_edges):
            redundancy_penalty = self.redundancy_penalty_weight
        return -(total_cost + connectivity_penalty + redundancy_penalty)

    def _decode_edges(self, chromosome: Chromosome) -> List[Edge]:
        return [edge for bit, edge in zip(chromosome, self.candidate_edges) if bit == 1]

    def _is_feasible(self, chromosome: Chromosome) -> bool:
        edges = self._decode_edges(chromosome)
        return (
            self._component_count(edges) == 1
            and self._has_hospital_depot_redundancy(edges)
        )

    def _total_cost(self, edges: Iterable[Edge]) -> float:
        total = 0.0
        for a, b in edges:
            total += self._edge_cost((a, b))
        return total

    def _edge_cost(self, edge: Edge) -> float:
        a, b = edge
        return self.graph.effective_cost(a, b) if self.use_effective_cost else self.graph.adjacency[a][b].base_cost

    def _component_count(self, edges: Sequence[Edge]) -> int:
        adj: Dict[Cell, Set[Cell]] = {node: set() for node in self.nodes}
        for a, b in edges:
            adj[a].add(b)
            adj[b].add(a)

        seen: Set[Cell] = set()
        components = 0
        for node in self.nodes:
            if node in seen:
                continue
            components += 1
            stack = [node]
            seen.add(node)
            while stack:
                cur = stack.pop()
                for nxt in adj[cur]:
                    if nxt in seen:
                        continue
                    seen.add(nxt)
                    stack.append(nxt)
        return components

    def _has_hospital_depot_redundancy(self, edges: Sequence[Edge]) -> bool:
        """True iff there exist >=2 edge-disjoint paths between hospital and depot.

        Uses Menger's theorem (k=2): the maximum number of edge-disjoint paths
        between two nodes equals the minimum edge cut between them. We compute
        this with a BFS-based augmenting-path max-flow (Edmonds-Karp), modeling
        each undirected edge as two directed arcs of capacity 1 each. The
        previous heuristic (find any path, then re-search after banning each
        single edge) was strictly weaker than 2-edge-connectivity.
        """
        if self.primary_hospital == self.ambulance_depot:
            return True
        return self._max_flow_at_least_two(
            edges, self.primary_hospital, self.ambulance_depot
        )

    def _max_flow_at_least_two(
        self, edges: Sequence[Edge], src: Cell, dst: Cell
    ) -> bool:
        # Each undirected edge {a,b} -> two directed arcs a->b and b->a, cap=1.
        # Augmenting along an arc decrements its capacity and increments the
        # reverse residual; this is the standard way to permit flow cancellation
        # so that BFS-based Ford-Fulkerson on undirected graphs finds the true
        # maximum number of edge-disjoint paths.
        capacity: Dict[Cell, Dict[Cell, int]] = defaultdict(lambda: defaultdict(int))
        for a, b in edges:
            capacity[a][b] += 1
            capacity[b][a] += 1

        flow = 0
        while flow < 2:
            prev: Dict[Cell, Optional[Cell]] = {src: None}
            queue: deque = deque([src])
            found = False
            while queue and not found:
                u = queue.popleft()
                for v, c in capacity[u].items():
                    if c <= 0 or v in prev:
                        continue
                    prev[v] = u
                    if v == dst:
                        found = True
                        break
                    queue.append(v)
            if not found:
                break

            node = dst
            while prev[node] is not None:
                parent = prev[node]
                capacity[parent][node] -= 1
                capacity[node][parent] += 1
                node = parent
            flow += 1

        return flow >= 2

    def _find_path(
        self,
        edges: Sequence[Edge],
        source: Cell,
        target: Cell,
        banned_edge: Optional[Edge],
    ) -> List[Cell]:
        adj: Dict[Cell, Set[Cell]] = {node: set() for node in self.nodes}
        for a, b in edges:
            e = self._normalize_edge(a, b)
            if banned_edge is not None and e == banned_edge:
                continue
            adj[a].add(b)
            adj[b].add(a)

        queue: List[Cell] = [source]
        prev: Dict[Cell, Cell] = {}
        seen: Set[Cell] = {source}
        while queue:
            node = queue.pop(0)
            if node == target:
                break
            for nxt in adj[node]:
                if nxt in seen:
                    continue
                seen.add(nxt)
                prev[nxt] = node
                queue.append(nxt)
        if target not in seen:
            return []
        path = [target]
        while path[-1] != source:
            path.append(prev[path[-1]])
        path.reverse()
        return path

    def _tournament_select(self, population: Sequence[Chromosome], fitnesses: Sequence[float]) -> Chromosome:
        picks = [self.rng.randrange(len(population)) for _ in range(self.tournament_size)]
        winner = max(picks, key=lambda idx: fitnesses[idx])
        return population[winner][:]

    def _uniform_crossover(self, p1: Chromosome, p2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        c1: Chromosome = []
        c2: Chromosome = []
        for b1, b2 in zip(p1, p2):
            if self.rng.random() < 0.5:
                c1.append(b1)
                c2.append(b2)
            else:
                c1.append(b2)
                c2.append(b1)
        return c1, c2

    def _mutate(self, chromosome: Chromosome) -> None:
        for i in range(len(chromosome)):
            if self.rng.random() < self.mutation_rate:
                chromosome[i] = 1 - chromosome[i]

    @staticmethod
    def _normalize_edge(a: Cell, b: Cell) -> Edge:
        return (a, b) if a <= b else (b, a)
