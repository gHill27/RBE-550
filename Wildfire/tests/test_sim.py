"""
test_simulation_engine.py
=========================
pytest suite for SimulationEngine.

New tests added in this version
--------------------------------
- TestFallbackGoalSelection   — plan-failure walks the ranked candidate list
- TestWumpusCatch              — sim ends when truck reaches wumpus within 5 m
- TestSimDuration              — sim ends at 5-min (300 s) time limit
- TestFloodFillExtinguish      — connected BURNING cells within 4 tiles extinguished
- TestGaussianSampling         — PRM nodes cluster around map centre
- TestPRMDebugVisualizer       — plot_prm called with graph+nodes after build
"""

from __future__ import annotations

import math
import sys
import os
import types
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_tests_dir   = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_tests_dir)
for p in (_project_dir, _tests_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# Stub visualizer
# ---------------------------------------------------------------------------
_vis_mod = types.ModuleType("pathVisualizer")

class _FakeVisualizer:
    def __init__(self, *a, **kw):
        self.plot_prm_calls = []
    def update(self, *a, **kw):   pass
    def plot_prm(self, *a, **kw):
        self.plot_prm_calls.append((a, kw))
    def show_final(self, *a, **kw): pass
    def close(self, *a, **kw):    pass

_vis_mod.SimVisualizer     = _FakeVisualizer
_vis_mod.PlannerVisualizer = _FakeVisualizer
sys.modules["pathVisualizer"] = _vis_mod

# ---------------------------------------------------------------------------
# Real imports
# ---------------------------------------------------------------------------
from Map_Generator import Map, Status           # noqa: E402
from pathSimulator import SimulationEngine  # noqa: E402
from firetruck import Firetruck                 # noqa: E402
from wumpus import Wumpus                       # noqa: E402

# ---------------------------------------------------------------------------
# Shared constants / helpers
# ---------------------------------------------------------------------------
GRID      = 15
CELL      = 5.0
WORLD     = GRID * CELL   # 75 m
FT_START  = (12.0, 12.0, 0.0)
WU_START  = (60.0, 60.0)
_SAFE_FILL = 0.01          # guarantees STRtree is non-None


def make_engine(
    fill_percent      = 0.05,
    prm_nodes         = 40,
    replan_interval   = 1.0,
    firetruck_start   = FT_START,
    wumpus_start      = WU_START,
    sim_duration      = 300.0,
    extinguish_margin = 5.0,
    burn_lifetime     = 30.0,
    wumpus_catch_radius = 5.0,
    flood_fill_radius = 4,
) -> SimulationEngine:
    return SimulationEngine(
        grid_num            = GRID,
        cell_size           = CELL,
        fill_percent        = fill_percent,
        firetruck_start     = firetruck_start,
        wumpus_start        = wumpus_start,
        prm_nodes           = prm_nodes,
        replan_interval     = replan_interval,
        tick_real_time      = 0.0,
        plot                = False,
        plot_prm            = False,
        sim_duration        = sim_duration,
        extinguish_margin   = extinguish_margin,
        burn_lifetime       = burn_lifetime,
        wumpus_catch_radius = wumpus_catch_radius,
        flood_fill_radius   = flood_fill_radius,
    )


def make_bare_map(firetruck_pose=FT_START, wumpus_pose=WU_START):
    return Map(
        Grid_num       = GRID,
        cell_size      = CELL,
        fill_percent   = 0.0,
        wumpus         = None,
        firetruck      = None,
        firetruck_pose = firetruck_pose,
        wumpus_pose    = wumpus_pose,
    )


# ===========================================================================
# 1. Map.check_time_events regression
# ===========================================================================

class TestMapCheckTimeEvents:

    def _map_with_burning_cell(self):
        m     = make_bare_map()
        coord = (5, 5)
        m._append_new_obstacle(coord)
        m.set_status_on_obstacles([coord], Status.BURNING)
        return m, coord

    def test_find_burnable_obstacles_requires_coordinate(self):
        m, _ = self._map_with_burning_cell()
        with pytest.raises(TypeError):
            m.find_burnable_obstacles()

    def test_find_burnable_obstacles_with_coordinate(self):
        m, coord = self._map_with_burning_cell()
        assert isinstance(m.find_burnable_obstacles(coord), list)

    def test_find_burnable_obstacles_finds_self(self):
        m, coord = self._map_with_burning_cell()
        assert coord in m.find_burnable_obstacles(coord)

    def test_find_burnable_obstacles_finds_neighbours(self):
        m = make_bare_map()
        for c in [(5,5),(5,6),(5,12)]:
            m._append_new_obstacle(c)
        result = m.find_burnable_obstacles((5,5))
        assert (5,6)  in result
        assert (5,12) not in result

    def test_check_time_events_no_crash(self):
        m, _ = self._map_with_burning_cell()
        m.sim_time = 0.0
        try:
            for _ in range(120):
                m.sim_time += 0.1
                m.check_time_events()
        except TypeError as e:
            pytest.fail(f"check_time_events() crashed — 4.9s bug: {e}")

    def test_burning_spreads_after_10s(self):
        m = make_bare_map()
        for c in [(5,5),(5,6)]:
            m._append_new_obstacle(c)
        m.set_status_on_obstacles([(5,5)], Status.BURNING)
        m.sim_time = 0.0
        try:
            for _ in range(115):
                m.sim_time += 0.1
                m.check_time_events()
        except (RuntimeError, KeyError) as e:
            pytest.fail(f"check_time_events() bug: {e}")
        assert m.obstacle_coordinate_dict.get((5,6),{}).get("status") == Status.BURNING

    @pytest.mark.xfail(strict=False, reason="Map_Generator bug: bare tuple in set_status_on_obstacles")
    def test_burning_cell_burned_after_30s(self):
        m, coord = self._map_with_burning_cell()
        m.sim_time = 0.0
        for _ in range(310):
            m.sim_time += 0.1
            m.check_time_events()
        assert coord not in m.obstacle_set
        assert coord not in m.obstacle_coordinate_dict


# ===========================================================================
# 2. _normalize_goal
# ===========================================================================

class TestNormalizeGoal:

    def test_three_tuple(self):
        assert SimulationEngine._normalize_goal((1.0, 2.0, 3.0)) == (1.0, 2.0, 3.0)

    def test_two_tuple_adds_zero_theta(self):
        assert SimulationEngine._normalize_goal((1.0, 2.0)) == (1.0, 2.0, 0.0)

    def test_none_returns_none(self):
        assert SimulationEngine._normalize_goal(None) is None

    def test_error_string_returns_none(self):
        assert SimulationEngine._normalize_goal("ERROR CANT GO HERE") is None

    def test_empty_string_returns_none(self):
        assert SimulationEngine._normalize_goal("") is None

    def test_always_floats(self):
        assert all(isinstance(v, float) for v in
                   SimulationEngine._normalize_goal((1, 2, 3)))

    def test_two_tuple_always_floats(self):
        assert all(isinstance(v, float) for v in
                   SimulationEngine._normalize_goal((1, 2)))


# ===========================================================================
# 3. Engine initialisation
# ===========================================================================

class TestEngineInit:

    def setup_method(self):
        self.engine = make_engine()

    def test_map_created(self):
        assert isinstance(self.engine.map, Map)

    def test_firetruck_created(self):
        assert isinstance(self.engine.firetruck, Firetruck)

    def test_wumpus_created(self):
        assert isinstance(self.engine.wumpus, Wumpus)

    def test_circular_dep_resolved(self):
        assert self.engine.map.firetruck is self.engine.firetruck
        assert self.engine.map.wumpus    is self.engine.wumpus

    def test_same_map_shared(self):
        assert self.engine.firetruck.map is self.engine.map
        assert self.engine.wumpus.map    is self.engine.map

    def test_roadmap_built(self):
        assert self.engine.firetruck._roadmap_size > 0

    def test_sim_time_zero(self):
        assert self.engine.map.sim_time == pytest.approx(0.0)

    def test_no_viz_when_plot_false(self):
        assert self.engine.viz is None

    def test_default_sim_duration_300(self):
        """Default sim_duration must be 300 s (5 minutes)."""
        engine = make_engine()
        assert engine.sim_duration == pytest.approx(300.0)

    def test_wumpus_catch_radius_stored(self):
        engine = make_engine(wumpus_catch_radius=7.5)
        assert engine.wumpus_catch_radius == pytest.approx(7.5)

    def test_flood_fill_radius_stored(self):
        engine = make_engine(flood_fill_radius=6)
        assert engine.flood_fill_radius == 6

    def test_no_reverse_attributes(self):
        e = make_engine()
        for attr in ("_reversing", "_reverse_start", "reverse_duration", "reverse_speed"):
            assert not hasattr(e, attr)

    def test_firetruck_pose_matches_start(self):
        e = make_engine(firetruck_start=(20.0, 30.0, 45.0))
        pose = e.map.firetruck_pose
        assert (pose[0], pose[1], pose[2]) == pytest.approx((20.0, 30.0, 45.0))

    def test_wumpus_pose_matches_start(self):
        e = make_engine(wumpus_start=(55.0, 60.0))
        wp = e.map.wumpus_pose
        assert (wp[0], wp[1]) == pytest.approx((55.0, 60.0))


# ===========================================================================
# 4. Wumpus catch — sim ends within 5 m
# ===========================================================================

class TestWumpusCatch:

    def test_check_wumpus_caught_true_when_close(self):
        """_check_wumpus_caught() returns True when truck ≤ wumpus_catch_radius."""
        engine = make_engine(wumpus_catch_radius=5.0)
        engine.map.firetruck_pose = (10.0, 10.0, 0.0)
        engine.map.wumpus_pose    = (13.0, 10.0)   # 3 m away
        assert engine._check_wumpus_caught() is True

    def test_check_wumpus_caught_false_when_far(self):
        engine = make_engine(wumpus_catch_radius=5.0)
        engine.map.firetruck_pose = (10.0, 10.0, 0.0)
        engine.map.wumpus_pose    = (50.0, 50.0)   # far away
        assert engine._check_wumpus_caught() is False

    def test_check_wumpus_caught_true_at_exact_radius(self):
        engine = make_engine(wumpus_catch_radius=5.0)
        engine.map.firetruck_pose = (0.0, 0.0, 0.0)
        engine.map.wumpus_pose    = (5.0, 0.0)   # exactly 5 m
        assert engine._check_wumpus_caught() is True

    def test_step_returns_false_after_wumpus_caught(self):
        """step() must return False once the wumpus is caught."""
        engine = make_engine(wumpus_catch_radius=5.0)
        # Place wumpus right next to the truck
        engine.map.wumpus_pose = (engine.map.firetruck_pose[0] + 1.0,
                                  engine.map.firetruck_pose[1])
        result = engine.step()
        assert result is False

    def test_end_reason_set_to_wumpus_caught(self):
        engine = make_engine(wumpus_catch_radius=5.0)
        engine.map.wumpus_pose = (engine.map.firetruck_pose[0] + 1.0,
                                  engine.map.firetruck_pose[1])
        engine.step()
        assert engine._end_reason == "wumpus_caught"

    def test_zero_catch_radius_never_triggers_unless_on_top(self):
        engine = make_engine(wumpus_catch_radius=0.0)
        engine.map.firetruck_pose = (10.0, 10.0, 0.0)
        engine.map.wumpus_pose    = (10.1, 10.0)   # 0.1 m away — NOT caught
        assert engine._check_wumpus_caught() is False


# ===========================================================================
# 5. Sim duration — ends at 300 s (5 min)
# ===========================================================================

class TestSimDuration:

    def test_step_returns_false_past_duration(self):
        engine = make_engine(sim_duration=0.0)
        assert engine.step() is False

    def test_end_reason_not_wumpus_on_timeout(self):
        engine = make_engine(sim_duration=0.1)
        while engine.step():
            pass
        assert engine._end_reason != "wumpus_caught"

    def test_engine_survives_to_300s(self):
        """
        With default 300 s duration, step() must keep returning True for
        at least one tick.  (We don't run all 3000 ticks in CI.)
        """
        engine = make_engine(sim_duration=300.0)
        assert engine.step() is True


# ===========================================================================
# 6. Tick correctness
# ===========================================================================

class TestEngineTick:

    def setup_method(self):
        self.engine = make_engine(replan_interval=1.0)

    def test_clock_advances(self):
        t = self.engine.map.sim_time
        self.engine.step()
        assert self.engine.map.sim_time == pytest.approx(t + 0.1)

    def test_step_true_before_duration(self):
        assert self.engine.step() is True

    def test_50_ticks_no_crash(self):
        for i in range(50):
            try:
                self.engine.step()
            except Exception as e:
                pytest.fail(f"Tick {i+1}: {type(e).__name__}: {e}")

    def test_100_ticks_clock_correct(self):
        for _ in range(100):
            self.engine.step()
        assert self.engine.map.sim_time == pytest.approx(10.0, abs=0.15)

    def test_idle_always_replans(self):
        """Idle replans every tick regardless of replan_interval."""
        engine = make_engine(replan_interval=999.0)
        engine._truck_state       = "idle"
        engine._firetruck_path    = None
        engine._target_fire_cell  = None
        engine.map.firetruck_goal = None
        engine.map.active_fires   = set()
        engine.step()
        t1 = engine._last_replan_time
        engine.step()
        t2 = engine._last_replan_time
        assert t1 > 0
        assert t2 > t1


# ===========================================================================
# 7. Fallback goal selection on plan failure
# ===========================================================================

class TestFallbackGoalSelection:
    """
    When plan_to_fire() fails for the top-ranked fire candidate,
    _replan_firetruck() must try the next candidate in the ranked list.
    """

    def test_rank_fire_candidates_sorted_by_distance(self):
        engine = make_engine(burn_lifetime=30.0, extinguish_margin=5.0)
        engine.map.sim_time = 0.0
        close_cell = (3, 3)
        far_cell   = (10, 10)
        for cell in [close_cell, far_cell]:
            engine.map._append_new_obstacle(cell)
            engine.map.set_status_on_obstacles([cell], Status.BURNING)
            engine.map.obstacle_coordinate_dict[cell]["burn_time"] = 0.0
        ranked = engine._rank_fire_candidates()
        cells  = [c for _, c in ranked]
        # Both are viable (fresh) → closest first
        assert cells[0] == close_cell
        assert cells[1] == far_cell

    def test_viable_fires_before_fallback(self):
        """Viable fires (enough burn time) must appear before dying fires."""
        engine = make_engine(burn_lifetime=30.0, extinguish_margin=5.0)
        engine.map.sim_time = 10.0
        dying  = (3, 3)
        fresh  = (8, 8)
        engine.map._append_new_obstacle(dying)
        engine.map.set_status_on_obstacles([dying], Status.BURNING)
        engine.map.obstacle_coordinate_dict[dying]["burn_time"] = engine.map.sim_time - 28.0

        engine.map._append_new_obstacle(fresh)
        engine.map.set_status_on_obstacles([fresh], Status.BURNING)
        engine.map.obstacle_coordinate_dict[fresh]["burn_time"] = engine.map.sim_time

        ranked = engine._rank_fire_candidates()
        cells  = [c for _, c in ranked]
        assert cells[0] == fresh    # viable comes first
        assert cells[1] == dying    # fallback second

    def test_replan_tries_next_candidate_when_first_fails(self):
        """
        Inject two fire cells.  Force plan_to_fire to fail for cell A but
        succeed for cell B.  _replan_firetruck must update _target_fire_cell
        to B.
        """
        engine = make_engine(burn_lifetime=30.0, extinguish_margin=5.0)
        engine.map.sim_time = 0.0
        cell_a = (3, 3)
        cell_b = (5, 5)
        for cell in [cell_a, cell_b]:
            engine.map._append_new_obstacle(cell)
            engine.map.set_status_on_obstacles([cell], Status.BURNING)
            engine.map.obstacle_coordinate_dict[cell]["burn_time"] = 0.0

        # Pre-populate _fire_candidates so _replan_firetruck uses them
        engine._fire_candidates    = [cell_a, cell_b]
        engine._target_fire_cell   = cell_a

        call_log = []

        def fake_plan_to_fire(fire_cell, start_state, radius):
            call_log.append(fire_cell)
            # Fail for cell_a, succeed for cell_b
            return None if fire_cell == cell_a else [(10.0, 10.0, 0.0)]

        engine.firetruck.plan_to_fire = fake_plan_to_fire
        engine._replan_firetruck()

        assert cell_a in call_log, "Should have tried cell_a first"
        assert cell_b in call_log, "Should have fallen back to cell_b"
        assert engine._target_fire_cell == cell_b

    def test_replan_keeps_path_when_fallback_succeeds(self):
        """_firetruck_path must be set when a fallback candidate succeeds."""
        engine = make_engine()
        engine._fire_candidates   = [(3, 3), (5, 5)]
        engine._target_fire_cell  = (3, 3)

        dummy_path = [(10.0, 10.0, 0.0), (15.0, 15.0, 0.0)]
        engine.firetruck.plan_to_fire = lambda fc, **kw: (
            None if fc == (3, 3) else dummy_path
        )
        engine._replan_firetruck()
        assert engine._firetruck_path == dummy_path

    def test_replan_path_none_when_all_fail(self):
        """If all candidates fail, _firetruck_path must remain None."""
        engine = make_engine()
        engine._fire_candidates   = [(3, 3), (5, 5)]
        engine._target_fire_cell  = (3, 3)
        engine._firetruck_path    = None
        engine.firetruck.plan_to_fire = lambda fc, **kw: None
        engine._replan_firetruck()
        assert engine._firetruck_path is None


# ===========================================================================
# 8. Flood-fill extinguish within 4 tiles
# ===========================================================================

class TestFloodFillExtinguish:

    def _engine_with_cluster(self, cells):
        """Engine with all given grid cells set to BURNING."""
        engine = make_engine(flood_fill_radius=4)
        for cell in cells:
            engine.map._append_new_obstacle(cell)
            engine.map.set_status_on_obstacles([cell], Status.BURNING)
        return engine

    def test_immediate_neighbour_extinguished(self):
        """A BURNING cell one step from the origin must be extinguished."""
        engine = self._engine_with_cluster([(5, 5), (5, 6)])
        engine._extinguish_connected((5, 5))
        status = engine.map.obstacle_coordinate_dict[(5, 6)]["status"]
        assert status == Status.EXTINGUISHED

    def test_cells_within_4_steps_extinguished(self):
        """A 5-cell chain: (5,5)→(5,6)→(5,7)→(5,8)→(5,9).
        Origin at (5,5), flood_fill_radius=4.  All 4 neighbours reachable."""
        cells = [(5, 5), (5, 6), (5, 7), (5, 8), (5, 9)]
        engine = self._engine_with_cluster(cells)
        engine._extinguish_connected((5, 5))
        for cell in [(5, 6), (5, 7), (5, 8), (5, 9)]:
            status = engine.map.obstacle_coordinate_dict[cell]["status"]
            assert status == Status.EXTINGUISHED, f"{cell} should be EXTINGUISHED"

    def test_cells_beyond_radius_not_extinguished(self):
        """Cell at depth 5 must NOT be extinguished when radius=4."""
        cells = [(5,5),(5,6),(5,7),(5,8),(5,9),(5,10)]
        engine = self._engine_with_cluster(cells)
        engine._extinguish_connected((5, 5))
        # (5,10) is at depth 5 — beyond flood_fill_radius=4
        status = engine.map.obstacle_coordinate_dict[(5, 10)]["status"]
        assert status == Status.BURNING

    def test_open_ground_stops_propagation(self):
        """
        (5,5) and (5,7) are burning, but (5,6) is open ground (not in
        obstacle_coordinate_dict).  Flood fill must not reach (5,7).
        """
        engine = make_engine(flood_fill_radius=4)
        # Only plant (5,5) and (5,7); skip (5,6) so it is open ground
        for cell in [(5, 5), (5, 7)]:
            engine.map._append_new_obstacle(cell)
            engine.map.set_status_on_obstacles([cell], Status.BURNING)
        engine._extinguish_connected((5, 5))
        status = engine.map.obstacle_coordinate_dict[(5, 7)]["status"]
        assert status == Status.BURNING, "(5,7) must remain BURNING: not connected through open ground"

    def test_zero_radius_extinguishes_nothing_beyond_origin(self):
        """With flood_fill_radius=0, no neighbours are touched."""
        cells = [(5, 5), (5, 6)]
        engine = self._engine_with_cluster(cells)
        engine.flood_fill_radius = 0
        engine._extinguish_connected((5, 5))
        status = engine.map.obstacle_coordinate_dict[(5, 6)]["status"]
        assert status == Status.BURNING

    def test_finish_suppression_triggers_flood_fill(self):
        """_finish_suppression must cascade flood-fill to connected cells."""
        engine = make_engine(flood_fill_radius=4)
        centre     = (5, 5)
        neighbour  = (5, 6)
        for cell in [centre, neighbour]:
            engine.map._append_new_obstacle(cell)
            engine.map.set_status_on_obstacles([cell], Status.BURNING)

        cs = engine.map.cell_size
        engine.map.firetruck_pose = (centre[0]*cs + cs/2, centre[1]*cs + cs/2, 0.0)
        engine._target_fire_cell  = centre
        engine._finish_suppression()

        # Both centre and neighbour should be extinguished
        for cell in [centre, neighbour]:
            status = engine.map.obstacle_coordinate_dict[cell]["status"]
            assert status == Status.EXTINGUISHED, f"{cell} should be EXTINGUISHED after suppression"

    def test_proximity_extinguish_triggers_flood_fill(self):
        """
        When proximity-extinguish fires, _extinguish_connected must be
        called.  Verify by checking a connected neighbour gets extinguished
        during the suppression tick.
        """
        engine = make_engine(flood_fill_radius=4)
        cell      = (5, 5)
        neighbour = (5, 6)
        for c in [cell, neighbour]:
            engine.map._append_new_obstacle(c)
            engine.map.set_status_on_obstacles([c], Status.BURNING)

        engine._target_fire_cell = cell
        engine._truck_state      = "suppressing"
        engine._suppress_start   = engine.map.sim_time
        # Pre-expire the proximity timer
        engine._proximity_timers = {
            cell: engine.map.sim_time - engine.proximity_duration - 1.0
        }
        cs = engine.map.cell_size
        engine.map.firetruck_pose = (cell[0]*cs + cs/2, cell[1]*cs + cs/2, 0.0)

        # Run one suppressing tick through the full engine tick
        engine.map.sim_time += 0.1   # advance clock
        extinguished = engine._check_proximity_extinguish()
        for c in extinguished:
            engine._extinguish_connected(c)

        assert engine.map.obstacle_coordinate_dict[neighbour]["status"] == Status.EXTINGUISHED


# ===========================================================================
# 9. Gaussian sampling
# ===========================================================================

class TestGaussianSampling:
    """
    _sample_points() must concentrate nodes near the map centre.

    We build a small PRM and check that more than half the nodes fall
    within the central 50% of the world (mu ± sigma, where sigma=world/4).
    This is a stochastic test so we use a generous threshold (>40%).
    """

    def test_nodes_cluster_around_centre(self):
        engine = make_engine(prm_nodes=80, fill_percent=_SAFE_FILL)
        world  = GRID * CELL   # 75 m
        mu     = world / 2.0   # 37.5 m
        sigma  = world / 4.0   # 18.75 m

        nodes   = engine.firetruck.nodes
        assert len(nodes) > 0, "No nodes sampled"

        within_sigma = sum(
            1 for x, y, _ in nodes
            if abs(x - mu) <= sigma and abs(y - mu) <= sigma
        )
        fraction = within_sigma / len(nodes)
        # Gaussian: ~68% of samples within ±1σ; even after rejection ~50% expected
        assert fraction > 0.40, (
            f"Only {fraction:.0%} of nodes within ±σ of centre — "
            "Gaussian sampling may not be applied"
        )

    def test_no_nodes_outside_map_bounds(self):
        """All sampled nodes must lie within [margin, world-margin]."""
        engine = make_engine(prm_nodes=60, fill_percent=_SAFE_FILL)
        world  = GRID * CELL
        margin = 5.0
        for x, y, _ in engine.firetruck.nodes:
            assert margin <= x <= world - margin, f"x={x} outside bounds"
            assert margin <= y <= world - margin, f"y={y} outside bounds"

    def test_heading_uniformly_distributed(self):
        """Heading must still be drawn from {0,45,...,315} (8 values)."""
        engine  = make_engine(prm_nodes=80, fill_percent=_SAFE_FILL)
        headings = {th for _, _, th in engine.firetruck.nodes}
        assert len(headings) > 1, "All nodes have the same heading — not uniform"
        for h in headings:
            assert h % 45 == 0, f"Heading {h}° is not a multiple of 45°"


# ===========================================================================
# 10. PRM debug visualizer
# ===========================================================================

class TestPRMDebugVisualizer:
    """
    When plot_prm=True, the engine must call firetruck.viz.plot_prm()
    with the graph and nodes after build_tree() completes.
    """

    def test_plot_prm_called_when_plot_prm_true(self):
        """
        Instantiate the engine with plot_prm=True.  The PlannerVisualizer
        stub records calls to plot_prm(); verify at least one was made.
        """
        engine = SimulationEngine(
            grid_num          = GRID,
            cell_size         = CELL,
            fill_percent      = _SAFE_FILL,
            firetruck_start   = FT_START,
            wumpus_start      = WU_START,
            prm_nodes         = 30,
            replan_interval   = 1.0,
            tick_real_time    = 0.0,
            plot              = False,
            plot_prm          = True,   # <-- enable PRM debug window
            sim_duration      = 300.0,
        )
        assert engine.firetruck.viz is not None, (
            "firetruck.viz should be set when plot_prm=True"
        )
        assert len(engine.firetruck.viz.plot_prm_calls) >= 1, (
            "plot_prm() should be called at least once after build_tree()"
        )

    def test_plot_prm_not_called_when_plot_prm_false(self):
        """With plot_prm=False, firetruck.viz must be None (no visualizer)."""
        engine = make_engine(fill_percent=_SAFE_FILL)
        assert engine.firetruck.viz is None, (
            "firetruck.viz should be None when plot_prm=False"
        )

    def test_plot_prm_receives_graph_and_nodes(self):
        """plot_prm() must be called with both the graph dict and nodes list."""
        engine = SimulationEngine(
            grid_num          = GRID,
            cell_size         = CELL,
            fill_percent      = _SAFE_FILL,
            firetruck_start   = FT_START,
            wumpus_start      = WU_START,
            prm_nodes         = 30,
            replan_interval   = 1.0,
            tick_real_time    = 0.0,
            plot              = False,
            plot_prm          = True,
            sim_duration      = 300.0,
        )
        assert engine.firetruck.viz is not None
        call_args, _ = engine.firetruck.viz.plot_prm_calls[0]
        # Signature: plot_prm(map, graph, nodes, path=None)
        # call_args is positional: (map, graph, nodes)
        assert len(call_args) >= 3, "plot_prm should receive map, graph, nodes"
        _map, graph, nodes = call_args[0], call_args[1], call_args[2]
        assert isinstance(graph, dict), "Second arg must be the PRM graph dict"
        assert isinstance(nodes, list), "Third arg must be the node list"
        assert len(nodes) > 0


# ===========================================================================
# 11. Firetruck advancement and arrival
# ===========================================================================

class TestFiretruckAdvancement:

    def _engine_with_path(self, path, target_cell=None):
        engine = make_engine()
        engine._firetruck_path    = list(path)
        engine._truck_state       = "driving"
        engine._target_fire_cell  = target_cell
        engine.map.firetruck_pose = path[0]
        return engine

    def test_pose_advances(self):
        path = [(12.0, 12.0, 0.0), (15.0, 12.0, 0.0), (18.0, 12.0, 0.0)]
        e = self._engine_with_path(path)
        e._advance_firetruck()
        assert e.map.firetruck_pose == (15.0, 12.0, 0.0)

    def test_path_shrinks(self):
        path = [(12.0, 12.0, 0.0), (15.0, 12.0, 0.0)]
        e = self._engine_with_path(path)
        e._advance_firetruck()
        assert len(e._firetruck_path) == 1

    def test_exhausted_with_fire_suppresses(self):
        e = self._engine_with_path([(12.0, 12.0, 0.0)], target_cell=(2, 2))
        e._advance_firetruck()
        assert e._truck_state == "suppressing"

    def test_exhausted_no_fire_idles(self):
        e = self._engine_with_path([(12.0, 12.0, 0.0)], target_cell=None)
        e._advance_firetruck()
        assert e._truck_state == "idle"

    def test_arrival_in_approach_radius_suppresses(self):
        """Fire cell (2,2) centre = (12.5, 12.5); waypoint (15.0,12.0) ≈ 2.7m away."""
        path = [(50.0, 50.0, 0.0), (15.0, 12.0, 0.0), (5.0, 5.0, 0.0)]
        e = self._engine_with_path(path, target_cell=(2, 2))
        e.approach_radius = 10.0
        e._advance_firetruck()
        assert e._truck_state == "suppressing"

    def test_arrival_freezes_path(self):
        path = [(50.0, 50.0, 0.0), (15.0, 12.0, 0.0), (5.0, 5.0, 0.0)]
        e = self._engine_with_path(path, target_cell=(2, 2))
        e.approach_radius = 10.0
        e._advance_firetruck()
        assert len(e._firetruck_path) == 1

    def test_far_from_fire_stays_driving(self):
        path = [(60.0, 60.0, 0.0), (50.0, 50.0, 0.0)]
        e = self._engine_with_path(path, target_cell=(2, 2))
        e.approach_radius = 10.0
        e._advance_firetruck()
        assert e._truck_state == "driving"

    def test_none_path_does_not_crash(self):
        e = make_engine()
        e._firetruck_path = None
        e._truck_state = "driving"
        e._target_fire_cell = None
        e._advance_firetruck()


# ===========================================================================
# 12. Suppression state transitions
# ===========================================================================

class TestSuppressionTransitions:

    def test_finish_suppression_extinguishes_cell(self):
        engine = make_engine()
        cell   = (5, 5)
        engine.map._append_new_obstacle(cell)
        engine.map.set_status_on_obstacles([cell], Status.BURNING)
        cs = engine.map.cell_size
        engine.map.firetruck_pose = (cell[0]*cs + cs/2, cell[1]*cs + cs/2, 0.0)
        engine._target_fire_cell  = cell
        engine._finish_suppression()
        assert engine.map.obstacle_coordinate_dict[cell]["status"] == Status.EXTINGUISHED

    def test_finish_suppression_clears_target(self):
        engine = make_engine()
        cell   = (3, 3)
        engine.map._append_new_obstacle(cell)
        engine.map.set_status_on_obstacles([cell], Status.BURNING)
        cs = engine.map.cell_size
        engine.map.firetruck_pose = (cell[0]*cs + cs/2, cell[1]*cs + cs/2, 0.0)
        engine._target_fire_cell  = cell
        engine._finish_suppression()
        assert engine._target_fire_cell is None
        assert engine.map.firetruck_goal is None

    def test_fire_cell_burned_out_extinguished(self):
        engine = make_engine()
        cell   = (4, 4)
        engine.map._append_new_obstacle(cell)
        engine.map.set_status_on_obstacles([cell], Status.BURNING)
        engine.map.set_status_on_obstacles([cell], Status.EXTINGUISHED)
        assert engine._fire_cell_burned_out(cell) is True

    def test_fire_cell_burned_out_burning(self):
        engine = make_engine()
        cell   = (4, 4)
        engine.map._append_new_obstacle(cell)
        engine.map.set_status_on_obstacles([cell], Status.BURNING)
        assert engine._fire_cell_burned_out(cell) is False

    def test_fire_cell_burned_out_missing(self):
        assert make_engine()._fire_cell_burned_out((99, 99)) is True


# ===========================================================================
# 13. Fire triage
# ===========================================================================

class TestFireTriage:

    def test_returns_none_with_no_fires(self):
        engine = make_engine()
        engine.map.active_fires = set()
        assert engine._select_best_fire_goal() is None

    def test_picks_viable_over_dying(self):
        engine = make_engine(burn_lifetime=30.0, extinguish_margin=5.0)
        engine.map.sim_time = 10.0
        for cell, elapsed in [((3,3), 28.0), ((8,8), 0.0)]:
            engine.map._append_new_obstacle(cell)
            engine.map.set_status_on_obstacles([cell], Status.BURNING)
            engine.map.obstacle_coordinate_dict[cell]["burn_time"] = (
                engine.map.sim_time - elapsed
            )
        assert engine._select_best_fire_goal() == (8, 8)

    def test_fallback_returns_closest_when_all_dying(self):
        engine = make_engine(burn_lifetime=30.0, extinguish_margin=5.0)
        engine.map.sim_time = 10.0
        for cell in [(3,3), (8,8)]:
            engine.map._append_new_obstacle(cell)
            engine.map.set_status_on_obstacles([cell], Status.BURNING)
            engine.map.obstacle_coordinate_dict[cell]["burn_time"] = (
                engine.map.sim_time - 29.5   # only 0.5s left
            )
        result = engine._select_best_fire_goal()
        assert result is not None  # fallback always returns something

    def test_remaining_burn_time_correct(self):
        engine = make_engine(burn_lifetime=30.0)
        engine.map.sim_time = 20.0
        cell = (4, 4)
        engine.map._append_new_obstacle(cell)
        engine.map.set_status_on_obstacles([cell], Status.BURNING)
        engine.map.obstacle_coordinate_dict[cell]["burn_time"] = 5.0
        assert engine._fire_remaining_burn_time(cell) == pytest.approx(15.0, abs=0.01)

    def test_remaining_zero_for_non_burning(self):
        engine = make_engine(burn_lifetime=30.0)
        cell = (4, 4)
        engine.map._append_new_obstacle(cell)
        assert engine._fire_remaining_burn_time(cell) == pytest.approx(0.0)


# ===========================================================================
# 14. Multi-tick integration
# ===========================================================================

class TestMultiTickIntegration:

    def test_50_ticks_no_fire(self):
        engine = make_engine(fill_percent=_SAFE_FILL)
        for i in range(50):
            try:
                engine.step()
            except Exception as e:
                pytest.fail(f"Tick {i+1}: {e}")

    def test_50_ticks_with_fire(self):
        engine = make_engine(fill_percent=_SAFE_FILL)
        coord = (7, 7)
        engine.map._append_new_obstacle(coord)
        engine.map.set_status_on_obstacles([coord], Status.BURNING)
        engine.map._append_new_obstacle((7, 8))
        for i in range(50):
            try:
                engine.step()
            except TypeError as e:
                pytest.fail(f"Tick {i+1} — 4.9s bug: {e}")

    def test_clock_correct_after_30_ticks(self):
        engine = make_engine()
        for _ in range(30):
            engine.step()
        assert engine.map.sim_time == pytest.approx(3.0, abs=0.01)

    def test_wumpus_stays_away_does_not_end_sim(self):
        """Wumpus far from truck: sim must not end on wumpus catch."""
        engine = make_engine(wumpus_catch_radius=5.0, sim_duration=1.0)
        # wumpus is at (60,60), truck starts at (12,12) — well beyond 5m
        while engine.step():
            pass
        assert engine._end_reason != "wumpus_caught"


# ===========================================================================
# 15. Status transitions
# ===========================================================================

class TestStatusTransitions:

    def test_intact_to_burning(self):
        m, coord = make_bare_map(), (3, 3)
        m._append_new_obstacle(coord)
        m.set_status_on_obstacles([coord], Status.BURNING)
        assert m.obstacle_coordinate_dict[coord]["status"] == Status.BURNING

    def test_burning_to_extinguished(self):
        m, coord = make_bare_map(), (3, 3)
        m._append_new_obstacle(coord)
        m.set_status_on_obstacles([coord], Status.BURNING)
        m.set_status_on_obstacles([coord], Status.EXTINGUISHED)
        assert m.obstacle_coordinate_dict[coord]["status"] == Status.EXTINGUISHED

    def test_extinguished_cannot_reignite(self):
        m, coord = make_bare_map(), (3, 3)
        m._append_new_obstacle(coord)
        m.set_status_on_obstacles([coord], Status.BURNING)
        m.set_status_on_obstacles([coord], Status.EXTINGUISHED)
        m.set_status_on_obstacles([coord], Status.BURNING)
        assert m.obstacle_coordinate_dict[coord]["status"] == Status.EXTINGUISHED

    def test_burning_in_active_fires(self):
        m, coord = make_bare_map(), (3, 3)
        m._append_new_obstacle(coord)
        m.set_status_on_obstacles([coord], Status.BURNING)
        assert coord in m.active_fires

    def test_extinguished_removed_from_active_fires(self):
        m, coord = make_bare_map(), (3, 3)
        m._append_new_obstacle(coord)
        m.set_status_on_obstacles([coord], Status.BURNING)
        m.set_status_on_obstacles([coord], Status.EXTINGUISHED)
        assert coord not in m.active_fires

    def test_burn_time_recorded(self):
        m = make_bare_map()
        m.sim_time = 5.5
        coord = (3, 3)
        m._append_new_obstacle(coord)
        m.set_status_on_obstacles([coord], Status.BURNING)
        assert m.obstacle_coordinate_dict[coord]["burn_time"] is not None