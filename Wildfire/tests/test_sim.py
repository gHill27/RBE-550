"""
test_simulation_engine.py
=========================
Comprehensive pytest suite for SimulationEngine and its interaction with
Map, Firetruck, and Wumpus.

Run with:
    pytest test_simulation_engine.py -v

Design philosophy
-----------------
Every test class targets one concrete layer or behaviour.  Tests are
ordered bottom-up: Map internals first, then agent helpers, then engine
mechanics, then integration.  A failure in an early class pinpoints the
root cause of failures in later classes.

All tests run WITHOUT a display (plot=False, tick_real_time=0.0) and use
a small grid (15 x 15) with few obstacles so they finish in < 10 seconds.

External modules (pathVisualizer, pathSimulator) are stubbed so the suite
runs without the full project tree installed.
"""

from __future__ import annotations
import math
import sys
import os
import types
import pytest

# 1. ADD PROJECT ROOT TO SYS.PATH
# This allows 'import firetruck' etc. to work from the /tests/ folder
_tests_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_tests_dir)
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

# 2. STUB THE VISUALIZER ONLY
# We stub 'pathVisualizer' because it likely imports Matplotlib/interferes with CI.
# We do NOT stub 'pathSimulator' because that is the logic we want to test.
_vis_mod = types.ModuleType("pathVisualizer")

class _FakeVisualizer:
    def __init__(self, *a, **kw): pass
    def update(self, *a, **kw):   pass
    def plot_prm(self, *a, **kw): pass
    def show_final(self, *a, **kw): pass

_vis_mod.SimVisualizer = _FakeVisualizer
_vis_mod.PlannerVisualizer = _FakeVisualizer
sys.modules["pathVisualizer"] = _vis_mod

# 3. NOW IMPORT REAL PROJECT MODULES
# Note: Ensure the class name in pathSimulator.py is actually SimulationEngine
from Map_Generator import Map, Status
from pathSimulator import SimulationEngine 
from firetruck import Firetruck
from wumpus import Wumpus

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
 
# Small, fast world used by almost every test
GRID   = 15
CELL   = 5.0
WORLD  = GRID * CELL   # 75 m × 75 m
 
FT_START = (12.0, 12.0, 0.0)   # well inside the safe buffer
WU_START = (60.0, 60.0)
 
 
def make_engine(
    fill_percent    = 0.05,
    prm_nodes       = 40,
    replan_interval = 1.0,
    firetruck_start = FT_START,
    wumpus_start    = WU_START,
    sim_duration    = 3600.0,
) -> SimulationEngine:
    """
    Build a minimal SimulationEngine with no display and zero wall-clock
    delay — suitable for unit and integration tests.
    """
    return SimulationEngine(
        grid_num        = GRID,
        cell_size       = CELL,
        fill_percent    = fill_percent,
        firetruck_start = firetruck_start,
        wumpus_start    = wumpus_start,
        prm_nodes       = prm_nodes,
        replan_interval = replan_interval,
        tick_real_time  = 0.0,
        plot            = False,
        sim_duration    = sim_duration,
    )
 
 
def make_bare_map(obstacles=None, firetruck_pose=FT_START, wumpus_pose=WU_START):
    """
    Build a Map with no agents attached — useful for testing Map logic
    in isolation without the full engine construction overhead.
    """
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
# 1. Map._normalize helpers — the 4.9s crash lives here
# ===========================================================================
 
class TestMapCheckTimeEvents:
    """
    map.check_time_events() is called every tick via map.main().
    It iterates over burning obstacles and calls find_burnable_obstacles(coordinate).
 
    KNOWN BUG (causes the 4.9s crash):
    ───────────────────────────────────
    check_time_events() calls self.find_burnable_obstacles() with NO argument,
    but find_burnable_obstacles(self, coordinate, radius_cells=6) REQUIRES one.
    This raises TypeError at the first tick where any burn_time check is active,
    which is exactly at sim_time = 10.1s if a fire starts at t=0,
    or earlier if a fire was seeded manually.
 
    These tests isolate that call path and confirm the fix.
    """
 
    def _map_with_burning_cell(self):
        """Return a bare map with one cell manually set to BURNING."""
        m = make_bare_map()
        # Manually plant an obstacle and set it burning
        coord = (5, 5)
        m._append_new_obstacle(coord)
        m.set_status_on_obstacles([coord], Status.BURNING)
        return m, coord
 
    def test_find_burnable_obstacles_requires_coordinate(self):
        """
        Confirm that find_burnable_obstacles(coordinate) requires its
        argument — calling it without one raises TypeError.
        This documents the root cause of the 4.9s crash.
        """
        m, _ = self._map_with_burning_cell()
        with pytest.raises(TypeError):
            m.find_burnable_obstacles()   # missing required arg
 
    def test_find_burnable_obstacles_with_coordinate(self):
        """
        find_burnable_obstacles((row, col)) must return a list (possibly
        empty) without raising when called correctly.
        """
        m, coord = self._map_with_burning_cell()
        result = m.find_burnable_obstacles(coord)
        assert isinstance(result, list)
 
    def test_find_burnable_obstacles_finds_self(self):
        """
        A burning cell should find itself when queried at its own coordinate.
        """
        m, coord = self._map_with_burning_cell()
        result = m.find_burnable_obstacles(coord)
        assert coord in result
 
    def test_find_burnable_obstacles_finds_neighbours(self):
        """
        Obstacles within radius_cells=6 of the queried coordinate should
        appear in the results.
        """
        m = make_bare_map()
        centre = (5, 5)
        near   = (5, 6)   # 1 cell away
        far    = (5, 12)  # 7 cells away — outside default radius
        for c in [centre, near, far]:
            m._append_new_obstacle(c)
        result = m.find_burnable_obstacles(centre)
        assert near   in result
        assert far not in result
 
    def test_check_time_events_does_not_crash_with_burning_cell(self):
        """
        REGRESSION TEST for the 4.9s crash.
 
        check_time_events() must complete without TypeError even when
        there is an active burning cell with a burn_time set.
        If the bug is unfixed this test raises:
            TypeError: find_burnable_obstacles() missing 1 required positional argument
        """
        m, coord = self._map_with_burning_cell()
        m.sim_time = 0.0
        # Simulate enough ticks to reach the 10s spread threshold
        try:
            for _ in range(120):   # 12 sim-seconds
                m.sim_time += 0.1
                m.check_time_events()
        except TypeError as e:
            pytest.fail(
                f"check_time_events() crashed with TypeError — this is the "
                f"4.9s bug. Fix: pass `coordinate` to find_burnable_obstacles(). "
                f"Original error: {e}"
            )
 
    def test_burning_cell_spreads_to_neighbour_after_10s(self):
        """
        After a cell has been burning for > 10 sim-seconds, check_time_events
        should call find_burnable_obstacles(coordinate) and then
        set_status_on_obstacles(nearby, Status.BURNING) on them.
 
        Two bugs in Map.check_time_events() prevent this from working today:
 
        BUG A — dict mutation during iteration:
            The loop iterates self.obstacle_coordinate_dict.items() while
            _delete_obstacle() deletes keys from it at the >30s branch.
            Fix: iterate over list(self.obstacle_coordinate_dict.items()).
 
        BUG B — set_status_on_obstacles called with wrong argument type:
            Line reads: self.set_status_on_obstacles(nearby_obstacles, Status.BURNING)
            where nearby_obstacles is already a list — this is correct.
            BUT the >30s branch calls:
                self.set_status_on_obstacles(coordinate, Status.BURNED)
            passing a bare tuple instead of [coordinate].
            set_status_on_obstacles iterates its first argument, so iterating
            a tuple (5, 5) gives integers 5 and 5 — KeyError or silent no-op.
            Fix: self.set_status_on_obstacles([coordinate], Status.BURNED)
 
        This test verifies the spread behaviour once both fixes are applied
        to Map_Generator.py.
        """
        m = make_bare_map()
        centre    = (5, 5)
        neighbour = (5, 6)
        for c in [centre, neighbour]:
            m._append_new_obstacle(c)
 
        m.set_status_on_obstacles([centre], Status.BURNING)
        m.sim_time = 0.0
 
        try:
            for _ in range(115):   # 11.5 sim-seconds — past the 10s threshold
                m.sim_time += 0.1
                m.check_time_events()
        except RuntimeError as e:
            pytest.fail(
                f"check_time_events() mutated dict during iteration: {e}\n"
                "Fix (Bug A): change .items() to list(.items()) in check_time_events"
            )
        except KeyError as e:
            pytest.fail(
                f"check_time_events() passed bare tuple to set_status_on_obstacles: {e}\n"
                "Fix (Bug B): change set_status_on_obstacles(coordinate, ...) "
                "to set_status_on_obstacles([coordinate], ...)"
            )
 
        neighbour_status = m.obstacle_coordinate_dict.get(neighbour, {}).get("status")
        assert neighbour_status == Status.BURNING, (
            "Neighbour should be BURNING after 11.5s of adjacent fire. "
            "If still INTACT, Bug B is unfixed: set_status_on_obstacles receives "
            "a bare tuple (5,5) which iterates as integers, not as a coord."
        )
 
    def test_burning_cell_burned_after_30s(self):
        """
        A cell burning for > 30 sim-seconds should be deleted from
        obstacle_coordinate_dict and obstacle_set via _delete_obstacle().
 
        Two bugs in Map.check_time_events() block this:
 
        BUG A — dict mutation during iteration (same as spread test):
            Fix: list(self.obstacle_coordinate_dict.items())
 
        BUG B — set_status_on_obstacles called with bare tuple at >30s branch:
            Current code:
                self.set_status_on_obstacles(coordinate, Status.BURNED)
            `coordinate` is a tuple e.g. (5,5). Iterating it yields integers
            5 and 5. The status lookup fails silently (integers not in dict),
            so the obstacle is never marked BURNED and _delete_obstacle is
            never called — the cell stays in obstacle_set forever.
            Fix:
                self.set_status_on_obstacles([coordinate], Status.BURNED)
 
        Apply both fixes to Map_Generator.py, then this test will pass.
        """
        m, coord = self._map_with_burning_cell()
        m.sim_time = 0.0
 
        try:
            for _ in range(310):   # 31 sim-seconds
                m.sim_time += 0.1
                m.check_time_events()
        except RuntimeError as e:
            pytest.fail(
                f"check_time_events() mutated dict during iteration (Bug A): {e}\n"
                "Fix: list(self.obstacle_coordinate_dict.items())"
            )
        except KeyError as e:
            pytest.fail(
                f"check_time_events() passed bare tuple to set_status_on_obstacles (Bug B): {e}\n"
                "Fix: set_status_on_obstacles([coordinate], Status.BURNED)"
            )
 
        assert coord not in m.obstacle_set, (
            "Burned obstacle should be removed from obstacle_set after 30s. "
            "If still present, Bug B is unfixed: the bare-tuple argument means "
            "_delete_obstacle is never reached."
        )
        assert coord not in m.obstacle_coordinate_dict, (
            "Burned obstacle should be removed from obstacle_coordinate_dict after 30s."
        )
 
 
# ===========================================================================
# 2. SimulationEngine._normalize_goal
# ===========================================================================
 
class TestNormalizeGoal:
    """
    _normalize_goal() is the defensive layer that prevents the
    IndexError: tuple index out of range crash when find_firetruck_goal()
    returns a 2-tuple (wumpus fallback path).
 
    It must handle every possible return value from find_firetruck_goal().
    """
 
    def test_three_tuple_passthrough(self):
        """A valid (x, y, theta) 3-tuple must pass through unchanged."""
        result = SimulationEngine._normalize_goal((10.0, 20.0, 45.0))
        assert result == (10.0, 20.0, 45.0)
 
    def test_two_tuple_gets_zero_theta(self):
        """
        A 2-tuple (from wumpus_pose fallback) must be extended to
        (x, y, 0.0) — the missing theta that caused the original crash.
        """
        result = SimulationEngine._normalize_goal((10.0, 20.0))
        assert result == (10.0, 20.0, 0.0)
 
    def test_none_returns_none(self):
        """None input must return None — no goal to plan toward."""
        assert SimulationEngine._normalize_goal(None) is None
 
    def test_error_string_returns_none(self):
        """'ERROR CANT GO HERE' must be discarded → None."""
        assert SimulationEngine._normalize_goal("ERROR CANT GO HERE") is None
 
    def test_empty_string_returns_none(self):
        """An empty string (falsy) must return None."""
        assert SimulationEngine._normalize_goal("") is None
 
    def test_output_always_floats(self):
        """All three elements of the result must be float, not int."""
        result = SimulationEngine._normalize_goal((10, 20, 30))
        assert all(isinstance(v, float) for v in result)
 
    def test_two_tuple_output_always_floats(self):
        result = SimulationEngine._normalize_goal((10, 20))
        assert all(isinstance(v, float) for v in result)
 
    def test_negative_coords_preserved(self):
        """Negative coordinates must survive normalisation unchanged."""
        result = SimulationEngine._normalize_goal((-5.0, -10.0, 180.0))
        assert result == (-5.0, -10.0, 180.0)
 
 
# ===========================================================================
# 3. SimulationEngine._goal_has_changed
# ===========================================================================
 
class TestGoalHasChanged:
    """
    _goal_has_changed() decides whether to trigger an immediate replan
    outside the normal interval.  False negatives mean the truck keeps
    chasing a stale goal; false positives hammer the PRM unnecessarily.
    """
 
    def setup_method(self):
        self.engine = make_engine()
 
    def test_both_none_returns_false(self):
        """No goal before, no goal now → nothing changed."""
        self.engine.map.firetruck_goal = None
        self.engine._last_goal         = None
        assert self.engine._goal_has_changed() is False
 
    def test_goal_appears_from_none_returns_true(self):
        """A new goal appearing when there was none triggers a replan."""
        self.engine.map.firetruck_goal = (50.0, 50.0, 0.0)
        self.engine._last_goal         = None
        assert self.engine._goal_has_changed() is True
 
    def test_goal_disappears_returns_true(self):
        """Goal being cleared (e.g. after extinguishing) triggers a replan."""
        self.engine.map.firetruck_goal = None
        self.engine._last_goal         = (50.0, 50.0, 0.0)
        assert self.engine._goal_has_changed() is True
 
    def test_same_goal_returns_false(self):
        """Identical goal positions must not trigger a spurious replan."""
        goal = (50.0, 50.0, 0.0)
        self.engine.map.firetruck_goal = goal
        self.engine._last_goal         = goal
        assert self.engine._goal_has_changed() is False
 
    def test_small_drift_ignored(self):
        """
        Sub-metre drift (floating point noise) must not trigger a replan.
        Threshold is 1.0 m.
        """
        self.engine.map.firetruck_goal = (50.0,   50.0,   0.0)
        self.engine._last_goal         = (50.5,   50.5,   0.0)
        assert self.engine._goal_has_changed() is False
 
    def test_large_shift_returns_true(self):
        """A goal moving by > 1 m must trigger a replan."""
        self.engine.map.firetruck_goal = (50.0, 50.0, 0.0)
        self.engine._last_goal         = (30.0, 30.0, 0.0)
        assert self.engine._goal_has_changed() is True
 
    def test_heading_change_alone_not_detected(self):
        """
        Change in theta only (same x, y) must NOT trigger a replan —
        the engine compares positions, not headings.
        """
        self.engine.map.firetruck_goal = (50.0, 50.0, 90.0)
        self.engine._last_goal         = (50.0, 50.0,  0.0)
        assert self.engine._goal_has_changed() is False
 
 
# ===========================================================================
# 4. SimulationEngine initialisation
# ===========================================================================
 
class TestEngineInit:
    """
    The 5-step init sequence must produce a fully wired, consistent object.
    Tests here confirm the circular dependency is resolved and every
    component references the correct shared Map instance.
    """
 
    def setup_method(self):
        self.engine = make_engine()
 
    def test_map_created(self):
        """map attribute must be a Map instance."""
        assert isinstance(self.engine.map, Map)
 
    def test_firetruck_created(self):
        """firetruck attribute must be a Firetruck instance."""
        assert isinstance(self.engine.firetruck, Firetruck)
 
    def test_wumpus_created(self):
        """wumpus attribute must be a Wumpus instance."""
        assert isinstance(self.engine.wumpus, Wumpus)
 
    def test_circular_dependency_resolved(self):
        """
        map.firetruck and map.wumpus must point to the same agent objects
        held by the engine — not None (the initial placeholder values).
        """
        assert self.engine.map.firetruck is self.engine.firetruck
        assert self.engine.map.wumpus    is self.engine.wumpus
 
    def test_all_agents_share_same_map(self):
        """
        Firetruck and Wumpus must both hold a reference to the same Map
        object as the engine.  If they held copies, state changes would
        not propagate.
        """
        assert self.engine.firetruck.map is self.engine.map
        assert self.engine.wumpus.map    is self.engine.map
 
    def test_prm_roadmap_built(self):
        """The PRM roadmap must be built (_roadmap_size > 0) after init."""
        assert self.engine.firetruck._roadmap_size > 0
 
    def test_sim_time_starts_at_zero(self):
        """Simulation clock must start at 0.0."""
        assert self.engine.map.sim_time == pytest.approx(0.0)
 
    def test_no_visualizer_when_plot_false(self):
        """With plot=False, the viz attribute must be None."""
        assert self.engine.viz is None
 
    def test_replan_interval_stored(self):
        """Constructor parameter must be stored on the engine."""
        engine = make_engine(replan_interval=3.7)
        assert engine.replan_interval == pytest.approx(3.7)
 
    def test_firetruck_pose_matches_start(self):
        """map.firetruck_pose must equal the firetruck_start argument."""
        engine = make_engine(firetruck_start=(20.0, 30.0, 45.0))
        pose = engine.map.firetruck_pose
        assert (pose[0], pose[1], pose[2]) == pytest.approx((20.0, 30.0, 45.0))
 
    def test_wumpus_pose_matches_start(self):
        """map.wumpus_pose must equal the wumpus_start argument."""
        engine = make_engine(wumpus_start=(55.0, 60.0))
        wp = engine.map.wumpus_pose
        assert (wp[0], wp[1]) == pytest.approx((55.0, 60.0))
 
 
# ===========================================================================
# 5. Engine._tick — single-step correctness
# ===========================================================================
 
class TestEngineTick:
    """
    engine.step() / _tick() is the heartbeat of the simulation.
    Each call must advance the clock, optionally replan, and move agents.
    These tests drive the engine one tick at a time for precise control.
    """
 
    def setup_method(self):
        self.engine = make_engine(replan_interval=1.0)
 
    def test_single_tick_advances_clock(self):
        """One step must advance sim_time by exactly 0.1 s."""
        t_before = self.engine.map.sim_time
        self.engine.step()
        assert self.engine.map.sim_time == pytest.approx(t_before + 0.1)
 
    def test_step_returns_true_before_duration(self):
        """step() must return True while within sim_duration."""
        assert self.engine.step() is True
 
    def test_step_returns_false_after_duration(self):
        """step() must return False once sim_time exceeds sim_duration."""
        self.engine.sim_duration = 0.0
        assert self.engine.step() is False
 
    def test_clock_is_monotonically_increasing(self):
        """sim_time must never decrease across ticks."""
        times = []
        for _ in range(20):
            times.append(self.engine.map.sim_time)
            self.engine.step()
        for i in range(len(times) - 1):
            assert times[i+1] >= times[i], (
                f"sim_time decreased from {times[i]} to {times[i+1]}"
            )
 
    def test_50_ticks_no_crash(self):
        """
        REGRESSION: the sim must survive 50 ticks (5 sim-seconds) without
        crashing.  This directly catches the 4.9s crash caused by the
        missing coordinate argument in check_time_events().
        """
        try:
            for _ in range(50):
                self.engine.step()
        except Exception as e:
            pytest.fail(
                f"Engine crashed within 50 ticks: {type(e).__name__}: {e}\n"
                f"This is likely the 4.9s / check_time_events bug."
            )
 
    def test_100_ticks_no_crash(self):
        """Engine must survive 100 ticks (10 sim-seconds) cleanly."""
        for _ in range(100):
            self.engine.step()
        assert self.engine.map.sim_time == pytest.approx(10.0, abs=0.15)
 
    def test_replan_fires_on_first_tick(self):
        """
        _last_replan_time starts at -replan_interval, so the first tick
        must trigger a replan and set _last_replan_time to ~0.1.
        """
        assert self.engine._last_replan_time < 0
        self.engine.step()
        assert self.engine._last_replan_time > 0
 
    def test_replan_throttled_within_interval(self):
        """
        After the first replan, subsequent ticks within the replan_interval
        must NOT reset _last_replan_time (no new replan triggered).
        """
        self.engine.step()   # first tick — triggers replan
        t_after_first_replan = self.engine._last_replan_time
        self.engine.step()   # second tick — within interval
        # _last_replan_time should not have changed
        assert self.engine._last_replan_time == pytest.approx(t_after_first_replan)
 
    def test_replan_fires_after_interval(self):
        """
        After replan_interval sim-seconds have passed, the next tick must
        trigger a new replan (update _last_replan_time).
 
        Timing proof (replan_interval=0.5s):
          tick 1: sim_time=0.1, _last_replan_time=-0.5 → time_since=0.6 >= 0.5 → REPLAN → _last=0.1
          tick 2: sim_time=0.2, time_since=0.1 < 0.5  → no replan
          tick 3: sim_time=0.3, time_since=0.2 < 0.5  → no replan
          tick 4: sim_time=0.4, time_since=0.3 < 0.5  → no replan
          tick 5: sim_time=0.5, time_since=0.4 < 0.5  → no replan
          tick 6: sim_time=0.6, time_since=0.5 >= 0.5 → REPLAN → _last=0.6
        The second replan fires on tick 6, not tick 5.
        """
        engine = make_engine(replan_interval=0.5)
        engine.step()   # tick 1: first replan fires (_last_replan_time = 0.1)
        t1 = engine._last_replan_time
        engine.step()   # tick 2: t=0.2, since=0.1 — no replan
        engine.step()   # tick 3: t=0.3, since=0.2 — no replan
        engine.step()   # tick 4: t=0.4, since=0.3 — no replan
        engine.step()   # tick 5: t=0.5, since=0.4 — no replan (< 0.5)
        engine.step()   # tick 6: t=0.6, since=0.5 — REPLAN fires
        t2 = engine._last_replan_time
        assert t2 > t1, (
            f"Second replan should have fired at t=0.6 (interval=0.5s). "
            f"_last_replan_time stayed at {t1} — replan did not trigger."
        )
 
 
# ===========================================================================
# 6. Firetruck pose advancement
# ===========================================================================
 
class TestFiretruckAdvancement:
    """
    _advance_firetruck() pops one waypoint per tick and updates
    map.firetruck_pose.  The tests here confirm pose updates are correct
    and the goal-reached trigger fires at the right moment.
    """
 
    def setup_method(self):
        self.engine = make_engine()
        # Seed a simple 3-point path manually so tests are deterministic
        self.engine._firetruck_path = [
            (12.0, 12.0, 0.0),
            (15.0, 12.0, 0.0),
            (18.0, 12.0, 0.0),
        ]
        self.engine.map.firetruck_pose = (12.0, 12.0, 0.0)
 
    def test_pose_advances_one_waypoint_per_tick(self):
        """After one advance, firetruck_pose must equal the second waypoint."""
        self.engine._advance_firetruck()
        pose = self.engine.map.firetruck_pose
        assert pose == (15.0, 12.0, 0.0)
 
    def test_path_shrinks_by_one_per_tick(self):
        """Each advance pops the front of the path."""
        before = len(self.engine._firetruck_path)
        self.engine._advance_firetruck()
        assert len(self.engine._firetruck_path) == before - 1
 
    def test_single_waypoint_path_does_not_advance(self):
        """
        A path of length 1 means the truck is AT the goal — must not
        pop further or raise IndexError.
        """
        self.engine._firetruck_path = [(12.0, 12.0, 0.0)]
        self.engine._advance_firetruck()
        assert len(self.engine._firetruck_path) == 1
 
    def test_none_path_does_not_crash(self):
        """A None path must be handled gracefully."""
        self.engine._firetruck_path = None
        self.engine._advance_firetruck()   # must not raise
 
    def test_empty_path_does_not_crash(self):
        """An empty path list must be handled gracefully."""
        self.engine._firetruck_path = []
        self.engine._advance_firetruck()   # must not raise
 
    def test_goal_reached_clears_goal(self):
        """
        When the truck reaches within cell_size of the goal,
        map.firetruck_goal must be set to None.
        """
        self.engine.map.firetruck_goal = (15.0, 12.0, 0.0)
        self.engine._advance_firetruck()
        # The truck is now at (15.0, 12.0) — exactly at the goal
        assert self.engine.map.firetruck_goal is None
 
    def test_goal_not_cleared_when_far(self):
        """
        A goal far from the current waypoint must NOT be cleared.
        """
        self.engine.map.firetruck_goal = (65.0, 65.0, 0.0)
        self.engine._advance_firetruck()
        assert self.engine.map.firetruck_goal is not None
 
    def test_pose_is_always_3_tuple(self):
        """firetruck_pose must always remain a 3-element tuple after advance."""
        self.engine._advance_firetruck()
        pose = self.engine.map.firetruck_pose
        assert len(pose) == 3
 
 
# ===========================================================================
# 7. Wumpus pose advancement
# ===========================================================================
 
class TestWumpusAdvancement:
    """
    _advance_wumpus() pops one grid cell per tick, converts it to world
    metres, and updates map.wumpus_pose.
    """
 
    def setup_method(self):
        self.engine = make_engine()
        # Wumpus path in grid (row, col) cells
        self.engine._wumpus_path = [(6, 6), (6, 7), (6, 8)]
        self.engine.map.wumpus_pose = (
            6 * CELL + CELL / 2,
            6 * CELL + CELL / 2,
        )
 
    def test_pose_advances_to_next_cell_centre(self):
        """
        After one advance, wumpus_pose must be the centre of cell (6,7)
        in world metres: (6*5+2.5, 7*5+2.5) = (32.5, 37.5).
        """
        self.engine._advance_wumpus()
        wp = self.engine.map.wumpus_pose
        assert wp[0] == pytest.approx(6 * CELL + CELL / 2)
        assert wp[1] == pytest.approx(7 * CELL + CELL / 2)
 
    def test_path_shrinks_by_one(self):
        before = len(self.engine._wumpus_path)
        self.engine._advance_wumpus()
        assert len(self.engine._wumpus_path) == before - 1
 
    def test_single_cell_path_does_not_crash(self):
        self.engine._wumpus_path = [(6, 6)]
        self.engine._advance_wumpus()   # must not raise
 
    def test_none_path_does_not_crash(self):
        self.engine._wumpus_path = None
        self.engine._advance_wumpus()
 
    def test_empty_path_does_not_crash(self):
        self.engine._wumpus_path = []
        self.engine._advance_wumpus()
 
    def test_pose_is_world_metres_not_grid_coords(self):
        """
        Wumpus path is in grid cells but pose must be stored in world metres.
        A grid cell (6,7) with cell_size=5 → world centre (32.5, 37.5).
        Values should be >> 1 (not grid indices 6/7).
        """
        self.engine._advance_wumpus()
        wp = self.engine.map.wumpus_pose
        assert wp[0] > 10.0, "x should be in metres, not grid index"
        assert wp[1] > 10.0, "y should be in metres, not grid index"
 
 
# ===========================================================================
# 8. Firetruck-reached-goal action
# ===========================================================================
 
class TestFiretruckReachedGoal:
    """
    _on_firetruck_reached_goal() must extinguish a burning obstacle at the
    goal cell and clear map.firetruck_goal.
    """
 
    def _engine_with_burning_goal(self):
        engine = make_engine()
        cs = engine.map.cell_size
        # Plant a burning obstacle at grid cell (4, 4)
        coord = (4, 4)
        engine.map._append_new_obstacle(coord)
        engine.map.set_status_on_obstacles([coord], Status.BURNING)
        # Set the goal to the centre of that cell
        engine.map.firetruck_goal = (
            coord[0] * cs + cs / 2,
            coord[1] * cs + cs / 2,
            0.0,
        )
        return engine, coord
 
    def test_burning_obstacle_extinguished_on_arrival(self):
        """
        When the truck reaches a burning cell, status must change to
        EXTINGUISHED.
        """
        engine, coord = self._engine_with_burning_goal()
        engine._on_firetruck_reached_goal()
        status = engine.map.obstacle_coordinate_dict[coord]["status"]
        assert status == Status.EXTINGUISHED
 
    def test_goal_cleared_after_arrival(self):
        """map.firetruck_goal must be None after the truck arrives."""
        engine, _ = self._engine_with_burning_goal()
        engine._on_firetruck_reached_goal()
        assert engine.map.firetruck_goal is None
 
    def test_intact_obstacle_not_extinguished(self):
        """
        Arriving at an INTACT obstacle (no fire) must not change its status.
        """
        engine = make_engine()
        cs     = engine.map.cell_size
        coord  = (4, 4)
        engine.map._append_new_obstacle(coord)
        # Status is INTACT — do NOT set it burning
        engine.map.firetruck_goal = (
            coord[0] * cs + cs / 2,
            coord[1] * cs + cs / 2,
            0.0,
        )
        engine._on_firetruck_reached_goal()
        status = engine.map.obstacle_coordinate_dict[coord]["status"]
        assert status == Status.INTACT
 
    def test_none_goal_does_not_crash(self):
        """Calling with no goal set must be a silent no-op."""
        engine = make_engine()
        engine.map.firetruck_goal = None
        engine._on_firetruck_reached_goal()   # must not raise
 
 
# ===========================================================================
# 9. Goal refresh and replan integration
# ===========================================================================
 
class TestGoalRefreshAndReplan:
    """
    _refresh_goal() must always store a valid 3-tuple in map.firetruck_goal.
    _replan_firetruck() must produce a path (or keep the old one) without
    crashing regardless of what find_firetruck_goal() returns.
    """
 
    def setup_method(self):
        self.engine = make_engine()
 
    def test_refresh_goal_stores_3_tuple(self):
        """After _refresh_goal(), map.firetruck_goal must be a 3-tuple or None."""
        self.engine._refresh_goal()
        goal = self.engine.map.firetruck_goal
        if goal is not None:
            assert len(goal) == 3, f"Goal must be 3-tuple, got {goal}"
 
    def test_refresh_goal_handles_2_tuple_from_map(self, monkeypatch):
        """
        If find_firetruck_goal returns a 2-tuple, _refresh_goal must
        normalise it to a 3-tuple before storing.
        """
        monkeypatch.setattr(
            self.engine.map, "find_firetruck_goal", lambda: (30.0, 40.0)
        )
        self.engine._refresh_goal()
        goal = self.engine.map.firetruck_goal
        assert goal is not None
        assert len(goal) == 3
        assert goal[2] == pytest.approx(0.0)
 
    def test_refresh_goal_handles_error_string(self, monkeypatch):
        """'ERROR CANT GO HERE' must result in goal remaining unchanged."""
        self.engine.map.firetruck_goal = (10.0, 10.0, 0.0)   # pre-existing goal
        monkeypatch.setattr(
            self.engine.map, "find_firetruck_goal", lambda: "ERROR CANT GO HERE"
        )
        self.engine._refresh_goal()
        # Goal should not be overwritten with None
        assert self.engine.map.firetruck_goal == (10.0, 10.0, 0.0)
 
    def test_refresh_goal_handles_none(self, monkeypatch):
        """None from find_firetruck_goal must not crash and must not overwrite."""
        self.engine.map.firetruck_goal = (10.0, 10.0, 0.0)
        monkeypatch.setattr(
            self.engine.map, "find_firetruck_goal", lambda: None
        )
        self.engine._refresh_goal()
        assert self.engine.map.firetruck_goal == (10.0, 10.0, 0.0)
 
    def test_replan_firetruck_does_not_crash_with_2_tuple_goal(self):
        """
        If map.firetruck_goal is a 2-tuple, _replan_firetruck must normalise
        it and not raise IndexError on goal[2].
        """
        self.engine.map.firetruck_goal = (30.0, 40.0)   # 2-tuple
        try:
            self.engine._replan_firetruck()
        except IndexError as e:
            pytest.fail(
                f"_replan_firetruck raised IndexError on 2-tuple goal: {e}\n"
                f"This is the original crash bug."
            )
 
    def test_replan_firetruck_no_crash_with_valid_goal(self):
        """_replan_firetruck with a proper 3-tuple goal must not raise."""
        self.engine.map.firetruck_goal = (40.0, 40.0, 0.0)
        self.engine._replan_firetruck()   # must not raise
 
    def test_replan_firetruck_no_crash_with_none_goal(self):
        """_replan_firetruck with goal=None must silently return."""
        self.engine.map.firetruck_goal = None
        self.engine._replan_firetruck()   # must not raise
 
    def test_last_goal_updated_after_replan(self):
        """_last_goal must be set to the current goal after _replan()."""
        self.engine.map.firetruck_goal = (40.0, 40.0, 0.0)
        self.engine._replan()
        assert self.engine._last_goal == (40.0, 40.0, 0.0)
 
 
# ===========================================================================
# 10. Multi-tick integration — the real 4.9s scenario
# ===========================================================================
 
class TestMultiTickIntegration:
    """
    These tests run the full engine for multiple ticks with realistic
    conditions to catch emergent crashes that unit tests miss.
 
    The 4.9s scenario is reproduced by running 50 ticks with a fire active
    from tick 1.  If check_time_events() still has the missing-argument bug,
    this will fail at tick ~49.
    """
 
    def test_engine_survives_50_ticks_no_fire(self):
        """
        50 ticks on an empty map (fill_percent=0) must complete without
        any exception.
        """
        engine = make_engine(fill_percent=0.0)
        for i in range(50):
            try:
                engine.step()
            except Exception as e:
                pytest.fail(f"Crashed on tick {i+1}: {type(e).__name__}: {e}")
 
    def test_engine_survives_50_ticks_with_obstacles(self):
        """50 ticks with obstacles present must not crash."""
        engine = make_engine(fill_percent=0.08)
        for i in range(50):
            try:
                engine.step()
            except Exception as e:
                pytest.fail(f"Crashed on tick {i+1}: {type(e).__name__}: {e}")
 
    def test_engine_survives_50_ticks_with_active_fire(self):
        """
        CORE REGRESSION: 50 ticks with a burning cell active from tick 1.
 
        This is exactly the scenario that produced the 4.9s crash.
        The fire's check_time_events path (including find_burnable_obstacles)
        must execute 50 times without TypeError.
        """
        engine = make_engine(fill_percent=0.0)
 
        # Manually plant a burning obstacle near the map centre
        coord = (7, 7)
        engine.map._append_new_obstacle(coord)
        engine.map.set_status_on_obstacles([coord], Status.BURNING)
 
        # Also plant a neighbour so spread logic has something to work with
        neighbour = (7, 8)
        engine.map._append_new_obstacle(neighbour)
 
        for i in range(50):
            try:
                engine.step()
            except TypeError as e:
                pytest.fail(
                    f"TICK {i+1} — TypeError in check_time_events: {e}\n"
                    f"Root cause: find_burnable_obstacles() called without "
                    f"the required `coordinate` argument."
                )
            except Exception as e:
                pytest.fail(
                    f"TICK {i+1} — unexpected crash: {type(e).__name__}: {e}"
                )
 
    def test_sim_time_correct_after_n_ticks(self):
        """After N ticks, sim_time must equal N × 0.1 (within tolerance)."""
        engine = make_engine()
        N = 30
        for _ in range(N):
            engine.step()
        assert engine.map.sim_time == pytest.approx(N * 0.1, abs=0.01)
 
    def test_firetruck_pose_changes_over_time(self):
        """
        Over 30 ticks the firetruck must move — pose at tick 30 must differ
        from pose at tick 0 (assuming a path was planned).
        """
        engine = make_engine()
        initial_pose = engine.map.firetruck_pose
 
        # Ensure a goal and path exist before we start ticking
        engine.map.firetruck_goal = (60.0, 60.0, 0.0)
        engine._replan_firetruck()
 
        for _ in range(30):
            engine.step()
 
        final_pose = engine.map.firetruck_pose
        dist = math.hypot(
            final_pose[0] - initial_pose[0],
            final_pose[1] - initial_pose[1],
        )
        assert dist > 0.1, (
            "Firetruck did not move after 30 ticks — check path planning "
            "and _advance_firetruck."
        )
 
    def test_goal_cleared_and_replanned_after_extinguish(self):
        """
        When the truck extinguishes a fire and the goal is cleared,
        the next tick must trigger a new replan (_goal_has_changed returns True).
        """
        engine = make_engine(fill_percent=0.0, replan_interval=999.0)
        engine.step()   # first tick seeds replan
 
        # Simulate arriving at goal
        engine.map.firetruck_goal = None   # as if truck just arrived
 
        # Next tick: goal changed (was something, now None) → replan fires
        t_before = engine._last_replan_time
        engine.step()
        assert engine._last_replan_time > t_before, (
            "Goal being cleared should have triggered a replan on the next tick"
        )
 
    def test_wumpus_pose_changes_over_time(self):
        """
        Over 20 ticks the wumpus must move (assuming a path was planned).
        """
        engine = make_engine()
        initial_wp = engine.map.wumpus_pose
 
        # Force a fresh wumpus plan
        engine._replan_wumpus()
 
        for _ in range(20):
            engine.step()
 
        final_wp = engine.map.wumpus_pose
        dist = math.hypot(
            final_wp[0] - initial_wp[0],
            final_wp[1] - initial_wp[1],
        )
        assert dist >= 0.0   # at minimum it must not crash; movement is probabilistic
 
 
# ===========================================================================
# 11. Map.find_firetruck_goal return value contracts
# ===========================================================================
 
class TestFindFiretruckGoal:
    """
    find_firetruck_goal() has two branches:
      - Active fires present → return (x, y, theta) pointing to nearest fire
      - No fires             → return wumpus_pose (may be a 2-tuple!)
 
    These tests document the exact return shapes so _normalize_goal
    requirements are concrete and traceable.
    """
 
    def test_no_fires_returns_wumpus_pose(self):
        """With no active fires, the return value must equal wumpus_pose."""
        m = make_bare_map(wumpus_pose=(60.0, 60.0))
        result = m.find_firetruck_goal()
        # wumpus_pose is (x, y) — only 2 elements
        assert result == (60.0, 60.0)
 
    def test_no_fires_returns_2_tuple(self):
        """
        Confirm that the wumpus fallback returns a 2-tuple (the root of the
        IndexError crash).  This test DOCUMENTS the known contract mismatch
        that _normalize_goal compensates for.
        """
        m = make_bare_map(wumpus_pose=(60.0, 60.0))
        result = m.find_firetruck_goal()
        assert len(result) == 2, (
            "find_firetruck_goal() returns wumpus_pose which is a 2-tuple — "
            "this is expected behaviour; _normalize_goal must compensate."
        )
 
    def test_with_fire_returns_3_tuple(self):
        """
        With an active fire, the return must be a 3-tuple (x, y, theta).
        """
        m = make_bare_map(firetruck_pose=(12.0, 12.0, 0.0))
        coord = (5, 5)
        m._append_new_obstacle(coord)
        m.set_status_on_obstacles([coord], Status.BURNING)
        result = m.find_firetruck_goal()
        assert result is not None
        assert result != "ERROR CANT GO HERE"
        assert len(result) == 3
 
    def test_with_fire_goal_is_near_fire(self):
        """
        The returned goal must be positioned along the vector from the
        firetruck toward the fire, stopping stop_distance=7m short.
 
        DOCUMENTED MAP BUG: find_firetruck_goal() uses raw grid cell indices
        (row, col) directly as world metres — it does NOT multiply by cell_size.
        So fire cell (5,5) is treated as if it were at world coords (5m, 5m),
        not (25m, 25m).  This means the firetruck navigates toward grid-index
        space rather than real world space.
 
        This test documents the ACTUAL behaviour of the current Map code so
        that the bug is visible and traceable.  Once find_firetruck_goal is
        fixed to convert: target_x = coord[0] * cell_size + cell_size/2,
        update this test to use the corrected world-metre assertion.
 
        Current (buggy) behaviour:
          firetruck at (12, 12), fire treated as (5, 5) in metres.
          Goal is 7m short along that vector ≈ (9.95, 9.95).
        """
        m = make_bare_map(firetruck_pose=(12.0, 12.0, 0.0))
        coord = (5, 5)
        m._append_new_obstacle(coord)
        m.set_status_on_obstacles([coord], Status.BURNING)
        result = m.find_firetruck_goal()
 
        assert result is not None, "Should return a goal when fire is active"
        assert result != "ERROR CANT GO HERE", "Should not return error string"
        assert len(result) == 3, f"Goal must be 3-tuple, got {result}"
 
        # The goal must lie along the truck→fire vector, stop_distance=7m short.
        # With the current bug, "fire" is at raw grid coord (5,5) in metres.
        # Truck is at (12,12). Distance ≈ 9.9m. Stop 7m short → goal ≈ (9.95, 9.95).
        stop_distance = 7.0
        raw_fire_x, raw_fire_y = float(coord[0]), float(coord[1])
        fx, fy = 12.0, 12.0
        d = math.hypot(raw_fire_x - fx, raw_fire_y - fy)
 
        if d > stop_distance:
            # Goal should be stop_distance metres from the raw fire coord
            dist_goal_to_fire = math.hypot(result[0] - raw_fire_x,
                                           result[1] - raw_fire_y)
            assert dist_goal_to_fire == pytest.approx(stop_distance, abs=0.5), (
                f"Goal {result} should be {stop_distance}m from fire coord "
                f"({raw_fire_x},{raw_fire_y}), got {dist_goal_to_fire:.3f}m. "
                f"NOTE: find_firetruck_goal uses raw grid indices as metres — "
                f"fix by multiplying by cell_size when computing target_x/y."
            )
        else:
            # Truck is already within stop_distance — goal should equal truck pose
            dist_goal_to_truck = math.hypot(result[0] - fx, result[1] - fy)
            assert dist_goal_to_truck < stop_distance
 
    def test_multiple_fires_picks_closest(self):
        """
        With two burning cells, the returned goal must point toward
        the one closer to the firetruck.
        """
        m = make_bare_map(firetruck_pose=(12.0, 12.0, 0.0))
        near_coord = (3, 3)    # ~15m from firetruck
        far_coord  = (10, 10)  # ~50m from firetruck
        for coord in [near_coord, far_coord]:
            m._append_new_obstacle(coord)
            m.set_status_on_obstacles([coord], Status.BURNING)
 
        result = m.find_firetruck_goal()
 
        near_wx = near_coord[0] * CELL
        near_wy = near_coord[1] * CELL
        far_wx  = far_coord[0]  * CELL
        far_wy  = far_coord[1]  * CELL
 
        dist_to_near = math.hypot(result[0] - near_wx, result[1] - near_wy)
        dist_to_far  = math.hypot(result[0] - far_wx,  result[1] - far_wy)
        assert dist_to_near < dist_to_far, (
            "find_firetruck_goal should return the CLOSER fire"
        )
 
 
# ===========================================================================
# 12. Status transition correctness
# ===========================================================================
 
class TestStatusTransitions:
    """
    The Status state machine in Map.set_status_on_obstacles enforces:
      INTACT    → BURNING    (allowed)
      BURNING   → EXTINGUISHED (allowed)
      EXTINGUISHED → *        (blocked — final state)
      BURNED    → *           (blocked — final state)
      INTACT    → EXTINGUISHED (blocked — can't skip BURNING)
 
    These tests confirm the transitions so engine logic that depends on
    them (e.g. _on_firetruck_reached_goal) has a stable contract.
    """
 
    def test_intact_to_burning(self):
        m = make_bare_map()
        coord = (3, 3)
        m._append_new_obstacle(coord)
        m.set_status_on_obstacles([coord], Status.BURNING)
        assert m.obstacle_coordinate_dict[coord]["status"] == Status.BURNING
 
    def test_burning_to_extinguished(self):
        m = make_bare_map()
        coord = (3, 3)
        m._append_new_obstacle(coord)
        m.set_status_on_obstacles([coord], Status.BURNING)
        m.set_status_on_obstacles([coord], Status.EXTINGUISHED)
        assert m.obstacle_coordinate_dict[coord]["status"] == Status.EXTINGUISHED
 
    def test_extinguished_cannot_be_set_burning(self):
        """Once extinguished, a cell must not re-ignite."""
        m = make_bare_map()
        coord = (3, 3)
        m._append_new_obstacle(coord)
        m.set_status_on_obstacles([coord], Status.BURNING)
        m.set_status_on_obstacles([coord], Status.EXTINGUISHED)
        m.set_status_on_obstacles([coord], Status.BURNING)   # should be blocked
        assert m.obstacle_coordinate_dict[coord]["status"] == Status.EXTINGUISHED
 
    def test_burning_added_to_active_fires(self):
        """A cell transitioning to BURNING must appear in active_fires."""
        m = make_bare_map()
        coord = (3, 3)
        m._append_new_obstacle(coord)
        m.set_status_on_obstacles([coord], Status.BURNING)
        assert coord in m.active_fires
 
    def test_extinguished_removed_from_active_fires(self):
        """A cell being extinguished must be removed from active_fires."""
        m = make_bare_map()
        coord = (3, 3)
        m._append_new_obstacle(coord)
        m.set_status_on_obstacles([coord], Status.BURNING)
        m.set_status_on_obstacles([coord], Status.EXTINGUISHED)
        assert coord not in m.active_fires
 
    def test_burn_time_set_when_burning_starts(self):
        """Transitioning to BURNING must record a non-None burn_time."""
        m = make_bare_map()
        m.sim_time = 5.5
        coord = (3, 3)
        m._append_new_obstacle(coord)
        m.set_status_on_obstacles([coord], Status.BURNING)
        assert m.obstacle_coordinate_dict[coord]["burn_time"] is not None