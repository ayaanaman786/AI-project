# CityMind — An Urban Intelligence System
## Phase 1: Design Document

**Course:** Artificial Intelligence
**Program:** BS Computer Science
**Group Members:**
- 24i-0663 (Lead — CSP, GA, SA, shared graph, simulation/integration)
- 24i-3033 (Co-member — A\*, Machine Learning pipeline, Pygame UI)

**Phase 1 Deadline:** 26th April
**Primary Language:** Python 3

---

## 1. Executive Summary

CityMind is a decision-support system that models a mid-sized city as a graph and helps city authorities make five kinds of decisions: where to place different types of buildings, which roads to build, where to station ambulances, how to route emergency teams in real time when roads fail, and how to predict and incorporate crime risk.

Our core design decision is that **all five modules operate on a single shared `CityGraph` object**. No module keeps its own copy. When a road floods, every module sees the change on its next read. This single source of truth is what lets the 20-step simulation behave coherently.

The system uses six distinct AI techniques drawn from the course:

| # | Challenge | Technique | Course Topic |
|---|-----------|-----------|--------------|
| 1 | Layout planning | CSP (Backtracking + AC-3 + Min-Conflicts) | Constraint Satisfaction |
| 2 | Road network | Genetic Algorithm | Optimization / Evolutionary search |
| 3 | Ambulance placement | Simulated Annealing | Local search |
| 4 | Emergency routing | A\* with dynamic replanning | Informed search |
| 5a | Neighborhood clustering | K-Means | Unsupervised learning |
| 5b | Risk classification | Decision Tree | Supervised learning |

For every challenge we also considered at least one alternative and explain below why we rejected it.

---

## 2. Problem Understanding (In My Own Words)

### 2.1 Challenge 1 — City Layout Planning

We are given an empty grid and a fixed catalogue of location types (residential, hospital, school, industrial, power plant, ambulance depot). We have to decide which cell gets which type, subject to zoning rules:

- No industrial cell can be adjacent (4-neighbor) to a school or hospital.
- Every residential cell must have at least one hospital within **3 road hops** (BFS distance on the grid, not Euclidean).
- Every power plant must have at least one industrial cell within **2 road hops**.
- If the rules cannot all be satisfied simultaneously for the given grid size, the system must report **which rule is violated** and return the assignment with the **smallest number of violated constraints**.

This is a classic constraint satisfaction problem. The variables are the grid cells, the domains are the location types, and the constraints are the zoning rules. The fallback ("minimum conflict solution") is exactly what the Min-Conflicts algorithm produces.

### 2.2 Challenge 2 — Road Network Optimization

Given the placed locations from Challenge 1, we must choose a subset of possible edges (roads between adjacent cells) such that:

- Every location is reachable from every other location (the graph is connected).
- The **total road cost** is minimized (road costs are 1.0 standard, 0.8 through residential).
- There exist **two edge-disjoint paths** between the Primary Hospital and the Ambulance Depot. Removing any single edge must not disconnect them.

The first two requirements alone describe a Minimum Spanning Tree. But the third requirement — edge-disjoint redundancy between two specific nodes — means a tree is not enough, because every edge in a tree is a bridge. This is a **2-edge-connected spanning subgraph** problem, which is NP-hard in general. So we need a heuristic optimizer that can handle soft constraints through fitness penalties.

### 2.3 Challenge 3 — Ambulance Placement

Given the finished road network and the set of "citizen" locations (residential cells weighted by population density), we must place 3 ambulances on grid cells to **minimize the maximum shortest-path distance** from any citizen to their nearest ambulance. This is a **1-center / minimax facility location** problem (well, 3-center).

The objective is non-smooth (it's a maximum of distances), but it has strong local structure: moving one ambulance by one cell usually changes the objective by a small amount. That makes it a good candidate for local search.

### 2.4 Challenge 4 — Emergency Routing Under Changing Conditions

A medical team starts at a known cell and must visit a list of civilian locations **in order**. The graph is dynamic: random edges become blocked during the mission. Every time an edge in the current plan becomes blocked, the system must **replan** from the team's current position to the next target, using only edges that are still valid.

The requirement "always finds the shortest currently available path, not just any path" rules out greedy heuristics. We need an algorithm whose heuristic is admissible so that optimality is guaranteed — which is exactly A\*'s guarantee.

### 2.5 Challenge 5 — Crime Risk Prediction

This challenge has three stages:

1. **Cluster** the city's residential/mixed cells into groups based on features (population density, distance to nearest industrial zone, etc.) without using any labels. This is unsupervised learning.
2. **Generate synthetic crime incident rates** using a justified rule (e.g., "crime likelihood rises with population density and proximity to industrial zones"). Convert these into discrete labels (High / Medium / Low) and use them as training data for a **classification** model. This is supervised learning.
3. **Feed the predicted risk level back** into the shared graph as an edge-cost multiplier, so Challenges 2, 3, and 4 all see higher travel costs through High-risk areas.

The interesting part is that the clustering's purpose is feature engineering (the cluster ID becomes a feature), while the classifier's purpose is decision-support output.

---

## 3. AI Technique Choices and Justifications

For each challenge we state the chosen technique, explain why it fits, and compare it against at least one alternative we explicitly considered.

### 3.1 Challenge 1 — CSP with Backtracking + AC-3 + Min-Conflicts

**Chosen approach:**

- **Variables:** one per grid cell.
- **Domain:** {Residential, Hospital, School, Industrial, Power Plant, Ambulance Depot} plus quotas (e.g., exactly 1 ambulance depot, 2 hospitals, etc., so the problem stays bounded).
- **Constraints:**
  - *Unary/quota:* number of each type matches a target.
  - *Binary adjacency:* industrial cells are not 4-adjacent to school/hospital cells.
  - *Global reachability:* residential cells have a hospital within 3 hops; power plants have an industrial cell within 2 hops.
- **Solver pipeline:**
  1. Run **AC-3** for arc consistency to prune impossible values from adjacency constraints early.
  2. Run **Backtracking search** with MRV (Minimum Remaining Values) and LCV (Least Constraining Value) heuristics.
  3. If backtracking fails (no solution exists), automatically fall back to **Min-Conflicts** starting from a random complete assignment. Min-Conflicts returns the assignment with the fewest violated constraints, along with the identity of the violated rule — this is exactly what the problem asks for.

**Why this fits:** The problem explicitly says "assigning values under constraints" and explicitly asks for a "minimum conflict solution" on failure. The Min-Conflicts heuristic is named in the problem.

**Alternative considered — Genetic Algorithm on layouts:** A GA could encode a layout as a chromosome (one gene per cell) and evolve populations. We rejected it because:

- CSP is exact and will find a valid layout if one exists; GA only gives probabilistic guarantees.
- The constraints are naturally discrete binary checks — a CSP expresses them directly; in a GA we would need ad-hoc fitness penalties that tend to get stuck in infeasible regions.
- Explaining AC-3 propagation in viva is much cleaner than defending a GA's fitness-weight choices.

### 3.2 Challenge 2 — Genetic Algorithm

**Chosen approach:**

- **Chromosome:** a bitstring of length E, where E = number of candidate edges between adjacent cells. Bit = 1 means "build this road."
- **Fitness:** `fitness = - (total_cost + connectivity_penalty + redundancy_penalty)`
  - `connectivity_penalty`: large penalty × (number of components − 1). A BFS from any node tells me how many components the chosen edge set has.
  - `redundancy_penalty`: large penalty if no two edge-disjoint paths exist between Primary Hospital and Ambulance Depot. We can test this by running BFS, removing each edge of the path, and checking if an alternative path still exists; or more efficiently by checking whether the edge between them is a bridge using DFS low-link values.
- **Genetic operators:**
  - **Selection:** tournament selection (size 3).
  - **Crossover:** uniform crossover on edge bits.
  - **Mutation:** flip each bit with a low probability (~1/E).
  - **Seeding:** initialize half the population from an MST of the location subgraph plus random extra edges, so the GA starts from feasible-ish regions rather than pure noise.

**Why this fits:**

- The search space is exponential in the number of edges (2^E). An MST alone does not satisfy the redundancy constraint, so we need something more general.
- 2-edge-connected spanning subgraph is NP-hard; GA is a standard choice for NP-hard graph optimization with multiple soft constraints.
- The fitness function naturally absorbs the hard constraint (redundancy) as a penalty, letting the GA explore feasibility and optimality together.

**Alternative considered — Prim's MST + greedy augmentation:** Compute the MST first, then add the cheapest edge that makes the Hospital–Depot pair 2-edge-connected. We rejected it as the primary choice because:

- It's a heuristic with no optimality guarantee and can get stuck at clearly suboptimal layouts when the cheapest augmenting edge is expensive.
- Nothing in the greedy augmentation explores the tradeoff between tree cost and augmentation cost jointly — a GA does.
- That said, we will **use MST+augmentation as the GA's seed** so we get the best of both worlds.

### 3.3 Challenge 3 — Simulated Annealing

**Chosen approach:**

- **State:** a tuple of three grid cells representing the ambulance positions.
- **Objective (to minimize):** `max over all citizens c of min over all ambulances a of shortest_path_distance(c, a)`, where the shortest path uses the current edge weights (including risk multipliers and blocked edges).
- **Neighborhood:** pick one of the three ambulances at random and move it to a random adjacent cell (or, with lower probability, jump it to a random cell).
- **Annealing schedule:** start temperature T₀ chosen so the initial acceptance ratio is around 0.8, decay geometrically (T ← 0.95 × T every 50 iterations), stop when T falls below a floor.

**Why this fits:**

- The neighborhood is naturally defined on the grid.
- The objective is a max-of-mins — not smooth, not differentiable, but cheap to evaluate with precomputed shortest-path trees from each ambulance.
- SA's probabilistic uphill moves help escape the many local minima typical of minimax placement problems.
- The problem hint about "randomness" and "efficiency without examining every possibility" matches SA directly.

**Alternative considered — Genetic Algorithm:** We could encode an ambulance triple as a chromosome. We rejected it as the primary choice because:

- The search space here (cells³) is much smaller than Challenge 2's edge bitstring, and the neighborhood structure is strong — conditions where SA is known to outperform GA.
- Maintaining a population is overkill for a 3-variable problem.
- Using a different technique from Challenge 2 also demonstrates broader AI coverage (the rubric explicitly rewards breadth).

### 3.4 Challenge 4 — A\* with Dynamic Replanning

**Chosen approach:**

- **Subgoal structure:** the team has a list of civilians `[c1, c2, ..., ck]` to visit in order. We treat this as k consecutive A\* searches from the current position to `c_i`.
- **Heuristic:** Manhattan distance × minimum possible edge cost. Because the grid is 4-connected and costs are ≥ 0.8 (the residential discount), `h(n) = |Δx| + |Δy|) × 0.8` is admissible (it never overestimates true cost) and consistent.
- **Replanning trigger:** during travel along the planned path, a separate event handler marks edges as blocked when flooding events occur. After every simulation tick, if the team's next planned edge is now blocked, A\* is invoked again from the team's current node to the current target. Already-visited civilians are not revisited.
- **Efficiency note:** we will use a binary heap (Python's `heapq`) as the priority queue.

**Why this fits:**

- A\* with an admissible heuristic **guarantees** the shortest path, which is required by the problem ("always finds the shortest currently available path, not just any path").
- Replanning from scratch after each block is simple, correct, and fast enough on a 10×10 grid — no need for more complex incremental algorithms like D\* Lite for this grid size.

**Alternative considered — Dijkstra's algorithm (UCS):** Dijkstra is correct but explores uniformly in all directions; on a grid with a good heuristic, A\* explores far fewer nodes. We also considered **Greedy Best-First Search**: rejected because it's not optimal, and the problem explicitly forbids "just any path."

### 3.5 Challenge 5 — K-Means + Decision Tree

**Step 1 — Clustering (unsupervised):**

- **Algorithm:** K-Means, k = 3 (to roughly correspond to Low/Medium/High archetypes).
- **Features:** population density; Manhattan distance to nearest industrial zone; number of industrial cells within 3 hops; location type one-hot.
- **Output:** a cluster ID per neighborhood cell. This becomes a feature for the classifier.

**Step 2 — Synthetic data generation:**

- **Rule for incident rate:** `rate = w1 × pop_density + w2 × industrial_proximity_score + w3 × cluster_risk_bias + noise`. Weights are justified in the design as "population-adjacent and industrial-proximate areas are empirically associated with higher urban crime." The noise term is Gaussian to simulate real-world variance.
- **Labeling:** bin incident rates into terciles: bottom 33% = Low, middle 33% = Medium, top 33% = High. This guarantees balanced classes for training.

**Step 3 — Classification (supervised):**

- **Algorithm:** Decision Tree classifier (scikit-learn).
- **Features:** everything from Step 1 plus cluster ID.
- **Target:** {Low, Medium, High}.
- **Evaluation:** 80/20 train/test split, report accuracy and confusion matrix in the simulation log.

**Feedback into shared graph:**

- Each cell's predicted label maps to a multiplier: Low → 1.0, Medium → 1.5, High → 2.0.
- Every edge's effective cost becomes `base_cost × avg(risk_multiplier_of_endpoints)`.
- Because Challenges 2, 3, and 4 all read edge costs from the shared graph at the moment they run, the risk feedback propagates automatically.

**Why these fit:**

- The first step has **no labels**, so unsupervised learning is the only option. K-Means is the simplest, most defensible clustering algorithm in the course.
- The second step has **labels we generate**, so supervised classification applies. Decision Tree was chosen over alternatives because its output is directly interpretable — in viva we can literally read the splits and explain why a cell was labeled High.

**Alternative considered — k-Nearest Neighbors:** kNN would also work for Step 3. We rejected it because:

- kNN has no explicit model to inspect; a Decision Tree gives an explainable set of rules, which is a stronger answer for the viva.
- kNN's prediction cost scales with training set size; Decision Tree prediction is O(depth).

**Note on course coverage:** Supervised learning was not yet covered in class as of the design document date. 24i-3033, who owns Challenge 5, will study Decision Trees from the course reference material and standard scikit-learn examples before the implementation phase. The Decision Tree algorithm (recursive splitting on the feature with highest information gain) is self-contained and can be understood in a single reading session.

---

## 4. Shared City Graph Architecture

### 4.1 The Single Source of Truth

All five modules read from and write to a single `CityGraph` instance. No module maintains a private copy. This is the single most important architectural rule in the system; it is what makes "changes in one part of the system are immediately visible to all other parts" true by construction.

### 4.2 Data Model

```python
class CityGraph:
    # Nodes: cell (row, col) -> dict of properties
    #   type:          one of six location types
    #   population:    int (0 for non-residential)
    #   risk_level:    'Low' | 'Medium' | 'High'  (set by Challenge 5)
    #   risk_mult:     float derived from risk_level
    #   accessible:    bool
    #   cluster_id:    int (set by Challenge 5.1)
    #
    # Edges: adjacency[node][neighbor] -> dict
    #   base_cost:   1.0 or 0.8
    #   blocked:     bool
    #   effective_cost (property): base_cost × avg risk_mult of endpoints,
    #                              or infinity if blocked
```

Internally the graph is a dict-of-dicts adjacency list. We deliberately choose **not to use NetworkX** because for the viva each owner needs to defend every algorithm from first principles, and hiding the graph behind a library would make that harder. The dict-of-dicts is ~50 lines of code and can be explained line-by-line.

### 4.3 Module Interaction Diagram

```
                    +------------------------+
                    |     CityGraph          |
                    |  (nodes, edges,        |
                    |   risk_mult, blocked)  |
                    +-----------+------------+
                                |
   +----------+  writes layout  |  reads for all pathfinding
   |   CSP    |-----------------+-----------------+-----------+
   | (Ch. 1)  |                 |                 |           |
   +----------+        +--------v--------+        |           |
                       |       GA        |        |           |
   +----------+ writes |  roads (Ch. 2)  |        |           |
   |    ML    |<-------+--------+--------+        |           |
   |  K-means |                 |                 |           |
   |  + Tree  | writes          |                 |           |
   | (Ch. 5)  | risk_mult       |                 |           |
   +----------+                 v                 v           v
                       +----------------+  +-------------+  +---------+
                       |      SA        |  |     A*      |  |  EVENT  |
                       | ambulance pos  |  |  routing    |  | MANAGER |
                       |   (Ch. 3)      |  |  (Ch. 4)    |  | floods  |
                       +----------------+  +-------------+  +---------+
                                                                 |
                                                                 | blocks edges
                                                                 +--> CityGraph
```

**Order of execution in the 20-step simulation:**

1. **Step 0 (setup):**
   a. CSP places location types on the grid.
   b. K-Means clusters cells; Decision Tree predicts risk labels; `risk_mult` is written back to the graph.
   c. GA builds the road network using the now-updated edge costs.
   d. SA places 3 ambulances.
   e. A team mission is generated (start + ordered civilian list).
2. **Steps 1–20 (tick loop):**
   a. Event manager may block one or more edges (flood probability per tick).
   b. Risk multipliers may shift for cells affected by recent events; if they shift meaningfully, SA re-evaluates ambulance positions.
   c. If the medical team's current planned edge was blocked this tick, A\* replans.
   d. The team advances one edge along the current plan.
   e. The event log records every decision and change.

### 4.4 Concurrency and Consistency

The simulation is single-threaded and tick-based. All module calls within a tick run sequentially, so there are no race conditions. This keeps the code simple and makes the viva easier — either owner can trace any tick's decisions deterministically.

---

## 5. User Interface Wireframe

The UI is built with **Pygame**. The window is split into three regions.

```
+----------------------------------------------------------+
|  CityMind — Urban Intelligence System       Step: 07/20  |
+--------------------------------------+-------------------+
|                                      | VIEW TOGGLES      |
|                                      |  [x] Roads        |
|   10 x 10 CITY GRID                  |  [x] Risk Heatmap |
|                                      |  [ ] Amb. Coverage|
|   H . . R R R . . S .                |  [ ] Population   |
|   . . R R R R R . . .                |                   |
|   . R R R R R R R . .                | LEGEND            |
|   R R R . . . R R R .                |  H = Hospital     |
|   R R . I I . . R R .                |  S = School       |
|   . . . I I . A . . .                |  I = Industrial   |
|   . . . . . . . . P .                |  A = Ambulance    |
|   . S . . . . . . . .                |  P = Power Plant  |
|   . . . . . R R R R .                |  R = Residential  |
|   . . . . . . . . . .                |                   |
|                                      | CONTROLS          |
|   (red edges = blocked)              |  [ Step ]         |
|   (orange cells = High risk)         |  [ Run ]  [ Pause]|
|                                      |  [ Reset ]        |
+--------------------------------------+-------------------+
|  EVENT LOG                                               |
|  t=05  flood: edge (3,4)-(3,5) blocked                   |
|  t=05  A* replan: (2,3) -> (4,6)  cost 5.6 -> 6.8        |
|  t=06  team moved (2,3) -> (2,4)                         |
|  t=07  risk update: cell (5,5) Medium -> High            |
|  t=07  SA re-eval: ambulance #2 (5,5) -> (5,6)           |
+----------------------------------------------------------+
```

**Visual encoding:**

- Each cell is a 40×40 pixel square colored by location type.
- Roads are drawn as lines between cell centers; thickness encodes cost, red means blocked.
- Risk heatmap toggle overlays a semi-transparent red/yellow/green shading by predicted risk.
- Ambulance coverage toggle colors each cell by which ambulance is nearest, with a brightness proportional to distance.
- The medical team is rendered as a moving icon with a trailing path.
- The event log scrolls automatically and persists across the run.

---

## 6. Work Breakdown and Timeline

This is a two-member project. We divided the work so that each member owns a coherent block of modules they can develop and defend end-to-end, rather than both of us touching every file. The `CityGraph` is designed first and jointly reviewed so both members agree on the interface before any module builds on top of it.

### 6.1 Responsibility Matrix

| Component | Owner | Notes |
|-----------|-------|-------|
| Shared `CityGraph` class + utilities (BFS, hop distance) | **24i-0663** | Foundation; built in Week 1 and reviewed with partner before module work begins |
| Challenge 1 — CSP (Backtracking + AC-3 + Min-Conflicts) | **24i-0663** | |
| Challenge 2 — Genetic Algorithm (road network) | **24i-0663** | |
| Challenge 3 — Simulated Annealing (ambulance placement) | **24i-0663** | |
| Challenge 4 — A\* + dynamic replanning | **24i-3033** | Consumes graph; called by the simulation loop whenever an edge becomes blocked |
| Challenge 5.1 — K-Means clustering | **24i-3033** | |
| Challenge 5.2 — Synthetic crime data + Decision Tree classifier | **24i-3033** | |
| Challenge 5.3 — Risk-multiplier write-back into graph | **24i-3033** | Writes `risk_mult` onto the shared graph |
| Pygame UI (grid rendering, view toggles, event-log panel) | **24i-3033** | Reads from the shared graph; purely display layer |
| 20-step simulation loop + flooding event manager | **24i-0663** | Orchestrator that calls every module each tick |
| Cross-module integration + final testing | **Both, jointly** | Final week of Phase 2 |

### 6.2 Why this split

- Each member owns at least one module that uses a distinct major AI concept category, so both can confidently answer viva questions on techniques they personally implemented.
- 24i-3033 owns the full **machine learning pipeline** (Challenge 5) plus the most visual search algorithm (A\*), so her contributions are demo-friendly and directly visible on screen — she can walk the examiner through concrete, visible output.
- 24i-0663 owns the **infrastructure plus the three optimization/search modules** (CSP, GA, SA), which sit below the UI and drive the city's structural decisions.
- The `CityGraph` is the only shared surface; by defining its interface early, each member can work independently without blocking the other.

### 6.3 Timeline

**Week 1 — Design and foundation (April 20–26)**

| Day | 24i-0663 | 24i-3033 |
|-----|----------|----------|
| Apr 22 | Finalize design document (both review together) | Review design document; set up Python environment |
| Apr 23 | Begin `CityGraph` class | Sketch Pygame window, cell-rendering helper |
| Apr 24 | Finish `CityGraph` + BFS/hop-distance utilities | Confirm graph interface works from UI side |
| Apr 25 | Unit tests for graph; hand interface to partner | Draft static grid renderer against `CityGraph` |
| Apr 26 | **Design document submission (joint)** | |

**Week 2 — Implementation (April 27 – May 10)**

| Day(s) | 24i-0663 | 24i-3033 |
|--------|----------|----------|
| Apr 27–28 | Challenge 1 — CSP solver | Challenge 4 — A\* baseline (static graph) |
| Apr 29–30 | Challenge 1 — Min-Conflicts fallback | Challenge 4 — dynamic replanning on edge blocks |
| May 1–2 | Challenge 2 — GA (chromosome, fitness, operators) | Challenge 5.1 — K-Means on cell features |
| May 3 | Challenge 2 — GA tuning; MST-seeded initialization | Challenge 5.2 — synthetic data + Decision Tree training |
| May 4 | Challenge 3 — Simulated Annealing | Challenge 5.3 — write risk multipliers to graph; UI risk-heatmap overlay |
| May 5 | Simulation loop + flooding event manager | UI — view toggles (roads / risk / ambulance coverage) |
| May 6 | Wire all modules into the tick loop | UI — event-log panel; animated medical team |
| May 7 | Integration testing (joint) | Integration testing (joint) |
| May 8 | Bug-fix day (joint) | Bug-fix day (joint) |
| May 9 | Viva prep: rehearse algorithm walk-throughs | Viva prep: rehearse algorithm walk-throughs |
| May 10 | **Phase 2 submission (joint)** | |

**Week 3 — Demo & Defense**

- Both members rehearse explaining their owned algorithms from first principles.
- We prepare for modification-challenge scenarios, e.g.:
  - "Change the industrial-to-hospital distance rule" → one-line change in the CSP constraint function (24i-0663 demonstrates).
  - "Make the medical team prefer lower-risk routes" → scale the heuristic or edge costs (24i-3033 demonstrates).

### 6.4 Risk Mitigation

- **Risk: GA for Challenge 2 takes too long to converge.** Mitigation: seed the GA with MST+augmentation; cap generations at a reasonable limit and always keep the best-so-far solution.
- **Risk: CSP has no valid assignment for the chosen grid size.** Mitigation: by design, the Min-Conflicts fallback returns the best partial layout with violations reported — this is what the problem asks for.
- **Risk: Supervised learning not yet covered in class.** Mitigation: Decision Trees are self-studyable from the course text plus scikit-learn docs. 24i-3033 will study the algorithm (recursive splitting on the feature with highest information gain) in the first week of Phase 2 before implementing it.
- **Risk: Interface drift between owners.** Mitigation: `CityGraph` is finalized and unit-tested in Week 1 before any module work begins; both owners agree on the interface in writing.

---

## 7. AI-Concept Coverage Checklist (Rubric §3)

The rubric requires at least 4 distinct AI techniques, correctly applied. Our system uses 6:

1. ✅ **Constraint Satisfaction** — Ch. 1 (Backtracking, AC-3, Min-Conflicts)
2. ✅ **Informed Search** — Ch. 4 (A\* with admissible heuristic)
3. ✅ **Evolutionary / Genetic Search** — Ch. 2 (GA on edge bitstrings)
4. ✅ **Local Search** — Ch. 3 (Simulated Annealing)
5. ✅ **Unsupervised Learning** — Ch. 5.1 (K-Means)
6. ✅ **Supervised Learning** — Ch. 5.3 (Decision Tree)

---

## 8. Summary of Design Decisions

- **Single shared `CityGraph`** — mandatory for the rubric's integration requirement.
- **Custom graph class, not NetworkX** — so we can defend every algorithm in viva from first principles.
- **Grid size 10×10** — meaningful enough for all five challenges without blowing up GA/SA runtimes.
- **Six distinct AI techniques** — exceeds the rubric minimum of 4.
- **Tick-based, single-threaded simulation** — keeps the event ordering deterministic and debuggable.
- **Pygame UI** — supports real-time animation, which is the natural fit for this problem.

The design is intentionally conservative in its individual choices (each algorithm is a textbook one) and ambitious only in how the pieces integrate. This gives us the best chance of a working system we can defend end-to-end.

---

*End of Design Document.*
