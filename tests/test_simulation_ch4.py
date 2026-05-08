from citymind.layout_csp import LayoutPlannerCSP
from citymind.road_ga import RoadNetworkGA
from citymind.simulation import EmergencyMissionRunner, FloodEventManager, default_mission
from citymind.city_graph import CityGraph, RiskLevel


def _build_ready_graph() -> CityGraph:
    graph = CityGraph(8, 8)
    assert LayoutPlannerCSP(graph, seed=7).solve().success is True
    ga = RoadNetworkGA(graph, population_size=40, generations=80, seed=11)
    roads = ga.optimize()
    ga.apply_selected_roads(roads.selected_edges)
    return graph


def test_ordered_target_completion() -> None:
    graph = _build_ready_graph()
    start, civilians = default_mission(graph, civilian_count=3, seed=5)
    runner = EmergencyMissionRunner(graph, start, civilians, flood_probability=0.0, seed=2)
    state = runner.run(max_steps=60)
    assert state.completed is True
    assert state.target_index == len(civilians)


def test_replan_trigger_on_blocked_next_edge() -> None:
    graph = _build_ready_graph()
    start, civilians = default_mission(graph, civilian_count=2, seed=8)
    runner = EmergencyMissionRunner(graph, start, civilians, flood_probability=0.0, seed=2)

    target = civilians[0]
    first_plan = runner.router.find_path(graph, start, target)
    assert first_plan.found is True
    assert len(first_plan.path) > 1
    edge_a, edge_b = first_plan.path[0], first_plan.path[1]
    graph.set_edge_blocked(edge_a, edge_b, True)

    runner.step(tick=1)
    assert any("A* replan" in line or "replan failed" in line for line in runner.state.event_log)


def test_no_revisit_behavior_for_completed_targets() -> None:
    graph = _build_ready_graph()
    start, civilians = default_mission(graph, civilian_count=3, seed=9)
    runner = EmergencyMissionRunner(graph, start, civilians, flood_probability=0.0, seed=2)
    state = runner.run(max_steps=60)
    reached_logs = [line for line in state.event_log if "reached civilian" in line]
    assert len(reached_logs) == len(civilians)


def test_simulation_smoke_generates_logs() -> None:
    graph = _build_ready_graph()
    start, civilians = default_mission(graph, civilian_count=4, seed=17)
    runner = EmergencyMissionRunner(graph, start, civilians, flood_probability=0.05, seed=101)
    state = runner.run(max_steps=20)
    assert len(state.event_log) > 0


class _StubSA:
    def __init__(self) -> None:
        self.optimize_calls = 0
        self.applied = None

    def optimize(self):
        self.optimize_calls += 1
        return type("R", (), {"positions": ((0, 0), (0, 1), (1, 0))})()

    def apply_positions(self, positions):
        self.applied = positions


def test_flood_cap_per_tick_limits_new_blocks() -> None:
    graph = CityGraph(3, 3)
    events = FloodEventManager(graph, flood_probability=1.0, seed=1, max_blocks_per_tick=1)
    blocked = events.apply_tick()
    assert len(blocked) == 1
    blocked_again = events.apply_tick()
    assert len(blocked_again) == 1


def test_flood_cap_zero_blocks_nothing() -> None:
    graph = CityGraph(3, 3)
    events = FloodEventManager(graph, flood_probability=1.0, seed=1, max_blocks_per_tick=0)
    blocked = events.apply_tick()
    assert blocked == []


def _isolate_team_at_origin(graph: CityGraph) -> None:
    """Block every edge incident to (0, 0) so the team cannot move anywhere."""
    for neighbor in list(graph.neighbors((0, 0), include_blocked=True)):
        graph.set_edge_blocked((0, 0), neighbor, True)


def test_replan_failure_does_not_spam_log() -> None:
    graph = CityGraph(2, 2)
    _isolate_team_at_origin(graph)
    runner = EmergencyMissionRunner(
        graph,
        team_start=(0, 0),
        civilians_ordered=[(1, 1)],
        flood_probability=0.0,
        seed=2,
        repair_after_failures=10,
    )

    for tick in range(1, 6):
        runner.step(tick=tick)

    failure_lines = [line for line in runner.state.event_log if "replan failed" in line]
    assert len(failure_lines) == 1
    assert "(waiting)" in failure_lines[0]
    assert all("repair" not in line for line in runner.state.event_log)
    assert runner.state.team_pos == (0, 0)


def test_repair_restores_connectivity_and_team_progresses() -> None:
    graph = CityGraph(2, 2)
    _isolate_team_at_origin(graph)
    runner = EmergencyMissionRunner(
        graph,
        team_start=(0, 0),
        civilians_ordered=[(1, 1)],
        flood_probability=0.0,
        seed=2,
        repair_after_failures=2,
    )

    runner.step(tick=1)
    assert runner._failure_streak == 1
    assert runner.state.team_pos == (0, 0)

    runner.step(tick=2)
    repair_lines = [line for line in runner.state.event_log if "repair" in line]
    assert len(repair_lines) == 1
    assert runner._failure_streak == 0

    unblocked_neighbors = graph.neighbors((0, 0))
    assert len(unblocked_neighbors) >= 1

    runner.step(tick=3)
    assert runner.state.team_pos != (0, 0)


def test_full_demo_run_reaches_at_least_one_civilian() -> None:
    from citymind.ui import SetupPipeline

    artifacts = SetupPipeline(rows=10, cols=10, civilian_count=3, flood_probability=0.0).run()
    state = artifacts.runner.run(max_steps=20)
    assert state.target_index >= 1


def test_apply_risk_shifts_escalates_touched_cells_low_to_high() -> None:
    """Direct unit test of _apply_risk_shifts escalation chain.

    With risk_shift_probability=1.0, every cell incident to a blocked edge
    should escalate by exactly one level per call: Low -> Medium -> High,
    and stay at High thereafter."""
    graph = CityGraph(2, 2)
    runner = EmergencyMissionRunner(
        graph,
        team_start=(0, 0),
        civilians_ordered=[(1, 1)],
        flood_probability=0.0,
        seed=2,
        risk_shift_probability=1.0,
    )

    edge = ((0, 0), (0, 1))
    assert graph.nodes[(0, 0)].risk_level == RiskLevel.LOW
    assert graph.nodes[(0, 1)].risk_level == RiskLevel.LOW

    changed = runner._apply_risk_shifts([edge])
    assert changed == 2
    assert graph.nodes[(0, 0)].risk_level == RiskLevel.MEDIUM
    assert graph.nodes[(0, 1)].risk_level == RiskLevel.MEDIUM

    changed = runner._apply_risk_shifts([edge])
    assert changed == 2
    assert graph.nodes[(0, 0)].risk_level == RiskLevel.HIGH
    assert graph.nodes[(0, 1)].risk_level == RiskLevel.HIGH

    # Already at High; no further escalation possible -> changed count == 0.
    changed = runner._apply_risk_shifts([edge])
    assert changed == 0


def test_apply_risk_shifts_skips_cells_when_probability_zero() -> None:
    graph = CityGraph(2, 2)
    runner = EmergencyMissionRunner(
        graph,
        team_start=(0, 0),
        civilians_ordered=[(1, 1)],
        flood_probability=0.0,
        seed=2,
        risk_shift_probability=0.0,
    )
    edge = ((0, 0), (0, 1))
    changed = runner._apply_risk_shifts([edge])
    assert changed == 0
    assert graph.nodes[(0, 0)].risk_level == RiskLevel.LOW
    assert graph.nodes[(0, 1)].risk_level == RiskLevel.LOW


def test_risk_shift_and_sa_reeval_triggered_on_flooded_edges() -> None:
    graph = _build_ready_graph()
    start, civilians = default_mission(graph, civilian_count=2, seed=8)
    sa = _StubSA()
    runner = EmergencyMissionRunner(
        graph,
        start,
        civilians,
        flood_probability=1.0,
        seed=2,
        risk_shift_probability=1.0,
        sa_reopt_threshold=1,
        sa_optimizer=sa,
    )
    before = graph.nodes[(0, 0)].risk_level
    runner.step(tick=1)
    after = graph.nodes[(0, 0)].risk_level
    assert after in {RiskLevel.MEDIUM, RiskLevel.HIGH} or before == RiskLevel.HIGH
    assert sa.optimize_calls >= 1
    assert sa.applied is not None
    assert any("SA re-eval" in line for line in runner.state.event_log)
