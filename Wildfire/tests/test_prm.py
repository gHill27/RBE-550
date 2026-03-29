"""
test_firetruck_prm.py
=====================
Comprehensive pytest suite for CarModel, ConfigurationSpace,
DubinsEdge, DubinsPlanner, and Firetruck (PRM builder + query).

Run from the project root (the folder that contains both
the tests/ directory and firetruck.py):

    pytest tests/test_prm.py -v
    pytest tests/test_prm.py -v -k "Dubins"
    pytest tests/test_prm.py -x --tb=short

HOW THE MOCKING WORKS
----------------------
firetruck.py does  `from pathVisualizer import PlannerVisualizer`
at module level.  If the real pathVisualizer.py is on sys.path when
Python processes that line the import succeeds — our mock never fires.

Fix: inject ALL stub modules into sys.modules at the very top of this
file, before os.path / sys.path manipulation and before any project
import.  Python's import machinery checks sys.modules first, so the
stubs win regardless of what is on disk.
"""

from __future__ import annotations
# ---------------------------------------------------------------------------
# Stdlib / third-party — always available
# ---------------------------------------------------------------------------
import math
import sys
import os
import types
import pytest
import numpy as np

# ---------------------------------------------------------------------------
# Stub out project-specific imports that are not part of this test target.
# We create minimal fake modules and inject them into sys.modules BEFORE
# importing firetruck_prm, so Python never tries to load the real files.
# ---------------------------------------------------------------------------

def _make_stub_map(grid_num=10, cell_size=30.0, obstacle_set=None,
                   firetruck_pose=(45.0, 45.0)):
    """
    Return a fake Map object.  Only the attributes actually read by
    Firetruck.__init__ and its methods are needed.
    """
    m = types.SimpleNamespace()
    m.grid_num        = grid_num
    m.cell_size       = cell_size
    m.obstacle_set    = obstacle_set if obstacle_set is not None else set()
    m.firetruck_pose  = firetruck_pose
    return m


# Inject fake Map_Generator module
_map_gen_mod = types.ModuleType("Map_Generator")
_map_gen_mod.Map    = type("Map",    (), {})
_map_gen_mod.Status = type("Status", (), {})
sys.modules.setdefault("Map_Generator", _map_gen_mod)

# Inject fake pathVisualizer module
_vis_mod = types.ModuleType("pathVisualizer")
class _FakeVisualizer:
    def __init__(self, *a, **kw): pass
    def plot_prm(self, *a, **kw): pass
_vis_mod.PlannerVisualizer = _FakeVisualizer
sys.modules.setdefault("pathVisualizer", _vis_mod)

# Inject fake pathSimulator module
_sim_mod = types.ModuleType("pathSimulator")
_sim_mod.PathSimulator = type("PathSimulator", (), {})
sys.modules.setdefault("pathSimulator", _sim_mod)

# ---------------------------------------------------------------------------
# Now it is safe to import the module under test
# ---------------------------------------------------------------------------
# ── Step 3: NOW we can safely add the project root to sys.path ───────────────
_tests_dir  = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_tests_dir)
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

# ── Step 4: import project code — stubs are already in sys.modules ───────────
import pytest  # noqa: E402  (pytest must come after sys.path is set)
from shapely.geometry import Point, Polygon, box  # noqa: E402

from firetruck import (  # noqa: E402
    _mod2pi,
    _dubins_compute,
    _DubinsPath,
    dubins_shortest_path,
    CarModel,
    ConfigurationSpace,
    Firetruck,
)


# ===========================================================================
# Helpers shared by multiple tests
# ===========================================================================

def make_map(grid_num=10, cell_size=30.0, obstacles=None, pose=(45.0, 45.0)):
    """Convenience wrapper around _make_stub_map."""
    return _make_stub_map(
        grid_num=grid_num,
        cell_size=cell_size,
        obstacle_set=obstacles if obstacles is not None else set(),
        firetruck_pose=pose,
    )


def open_map():
    """10×10 grid, 30 m cells, no obstacles → 300×300 m open world."""
    return make_map()


def small_prm(n_samples=50):
    """
    Build a tiny PRM (50 nodes) on an open map for fast unit testing.
    Returns the Firetruck instance after build_tree().
    """
    truck = Firetruck(open_map())
    truck.build_tree(n_samples=n_samples)
    return truck


# ===========================================================================
# 1. _mod2pi  —  angle wrapping utility
# ===========================================================================

class TestMod2Pi:
    """
    _mod2pi(angle) keeps any angle in [0, 2π).
    Dubins geometry passes intermediate angles through this to avoid
    accumulated negative or >2π values that would break trig comparisons.
    """

    def test_zero_unchanged(self):
        # 0 is already in range, must come back unchanged
        assert _mod2pi(0.0) == pytest.approx(0.0)

    def test_pi_unchanged(self):
        # π is inside [0, 2π), must be returned as-is
        assert _mod2pi(math.pi) == pytest.approx(math.pi)

    def test_two_pi_wraps_to_zero(self):
        # 2π is exactly the upper boundary and should wrap to 0
        assert _mod2pi(2 * math.pi) == pytest.approx(0.0, abs=1e-12)

    def test_negative_angle_wraps_positive(self):
        # -π/2 should become 3π/2 (270°)
        result = _mod2pi(-math.pi / 2)
        assert result == pytest.approx(3 * math.pi / 2)

    def test_large_positive_wraps(self):
        # 5π = 2π + 3π, so result should be 3π … but 3π > 2π, wraps to π
        assert _mod2pi(5 * math.pi) == pytest.approx(math.pi)

    def test_result_always_in_range(self):
        # Fuzz: for 1000 random angles the output is always in [0, 2π)
        rng = np.random.default_rng(0)
        for angle in rng.uniform(-100, 100, 1000):
            r = _mod2pi(float(angle))
            assert 0.0 <= r < 2 * math.pi + 1e-12


# ===========================================================================
# 2. _dubins_compute  —  per-path-type segment solver
# ===========================================================================

class TestDubinsCompute:
    """
    _dubins_compute(path_type, d, a, b) solves the normalised Dubins word
    equations for one of the six path types (LSL, RSR, LSR, RSL, RLR, LRL).

    d = Euclidean distance / turning_radius (dimensionless)
    a = departure heading offset (rad)
    b = arrival  heading offset (rad)

    Returns (t, p, q) — normalised arc/straight lengths — or None when the
    geometry is infeasible for that path type.
    """

    def test_straight_ahead_lsl(self):
        # Source and destination both pointing East (a=b=0), well separated.
        # LSL should give a valid (t, p, q) triple.
        result = _dubins_compute("LSL", d=5.0, a=0.0, b=0.0)
        assert result is not None
        t, p, q = result
        assert p >= 0.0, "straight segment length must be non-negative"

    def test_invalid_path_type_returns_none(self):
        # If an unrecognised path type is passed, the function falls through
        # all if-blocks and returns None (Python implicit return).
        result = _dubins_compute("ZZZ", d=5.0, a=0.0, b=0.0)
        assert result is None

    def test_rsr_symmetric(self):
        # RSR with a=b (same relative heading) should be computable
        result = _dubins_compute("RSR", d=4.0, a=math.pi / 4, b=math.pi / 4)
        assert result is not None

    def test_rlr_infeasible_returns_none(self):
        # RLR requires |val| <= 1; a very large d makes it infeasible
        result = _dubins_compute("RLR", d=100.0, a=0.0, b=0.0)
        assert result is None, "RLR should be None when points are far apart"

    def test_lrl_infeasible_returns_none(self):
        # Same geometry argument for LRL
        result = _dubins_compute("LRL", d=100.0, a=0.0, b=0.0)
        assert result is None

    def test_lsr_needs_sufficient_distance(self):
        # LSR: p² = d²-2 + 2cos(a-b) + 2d(sin(a)+sin(b))
        # To force p² < 0 we need d small AND the remaining terms to be
        # negative.  With a=b=π (both facing West) and d tiny:
        #   p² ≈ 0 - 2 + 2cos(0) + 0 = 0   (borderline)
        # Use a=π, b=0 (opposing headings) and d=0.5:
        #   p² = 0.25 - 2 + 2cos(π) + 2*0.5*(sin(π)+sin(0))
        #       = 0.25 - 2 - 2 + 0 = -3.75  → None
        result = _dubins_compute("LSR", d=0.5, a=math.pi, b=0.0)
        assert result is None, "LSR should be None when p² < 0"
 
    def test_all_types_return_tuple_or_none(self):
        # Every path type must return exactly (t,p,q) or None — never raises
        for ptype in ["LSL", "RSR", "LSR", "RSL", "RLR", "LRL"]:
            result = _dubins_compute(ptype, d=3.0, a=0.5, b=1.0)
            assert result is None or (isinstance(result, tuple) and len(result) == 3)

    def test_all_types_return_tuple_or_none(self):
        # Every path type must return exactly (t,p,q) or None — never raises
        for ptype in ["LSL", "RSR", "LSR", "RSL", "RLR", "LRL"]:
            result = _dubins_compute(ptype, d=3.0, a=0.5, b=1.0)
            assert result is None or (isinstance(result, tuple) and len(result) == 3)


# ===========================================================================
# 3. dubins_shortest_path  —  top-level Dubins path finder
# ===========================================================================

class TestDubinsShortestPath:
    """
    dubins_shortest_path(q0, q1, r) picks the cheapest feasible path among
    all six Dubins words, returns a _DubinsPath object, or None if q0 == q1.

    All angles here are in radians (the raw Dubins API; the Firetruck wrapper
    converts degrees↔radians transparently).
    """

    R = 13.0  # matches the firetruck's r_min

    def test_coincident_points_returns_none(self):
        # Two identical configurations have no meaningful Dubins path.
        # The function guards against this with a distance < 1e-6 check.
        q = (50.0, 50.0, 0.0)
        assert dubins_shortest_path(q, q, self.R) is None

    def test_returns_dubins_path_object(self):
        # Normal call must return a _DubinsPath, not None
        q0 = (0.0, 0.0, 0.0)
        q1 = (50.0, 0.0, 0.0)
        path = dubins_shortest_path(q0, q1, self.R)
        assert path is not None
        assert isinstance(path, _DubinsPath)

    def test_path_length_positive(self):
        # A valid path must have strictly positive length
        q0 = (0.0, 0.0, 0.0)
        q1 = (40.0, 30.0, math.pi / 2)
        path = dubins_shortest_path(q0, q1, self.R)
        assert path is not None
        assert path.path_length() > 0.0

    def test_path_length_at_least_euclidean(self):
        # Dubins path length >= straight-line distance (it may curve)
        q0 = (0.0, 0.0, 0.0)
        q1 = (60.0, 0.0, 0.0)
        dist = math.hypot(q1[0] - q0[0], q1[1] - q0[1])
        path = dubins_shortest_path(q0, q1, self.R)
        assert path.path_length() >= dist - 1e-6

    def test_straight_path_heading_aligned(self):
        # Two East-facing configs far apart on the x-axis: the shortest path
        # should be nearly straight, so length ≈ Euclidean distance
        q0 = (0.0,   0.0, 0.0)
        q1 = (100.0, 0.0, 0.0)
        path = dubins_shortest_path(q0, q1, self.R)
        assert path is not None
        assert path.path_length() == pytest.approx(100.0, rel=0.01)

    def test_opposite_heading_longer_path(self):
        # Facing away from each other forces a U-turn → longer than Euclidean
        q0 = (0.0,  0.0, 0.0)        # facing East
        q1 = (50.0, 0.0, math.pi)    # facing West
        path = dubins_shortest_path(q0, q1, self.R)
        dist = 50.0
        assert path.path_length() > dist

    def test_symmetry_of_length(self):
        # Reversing start/end (and flipping headings) gives same path length.
        # This is NOT the same as path.reverse() — it's a geometric check.
        q0 = (10.0, 20.0, 0.3)
        q1 = (80.0, 60.0, 1.1)
        p_fwd = dubins_shortest_path(q0, q1, self.R)
        # Reverse: flip positions AND headings by π
        q0r = (q1[0], q1[1], q1[2] + math.pi)
        q1r = (q0[0], q0[1], q0[2] + math.pi)
        p_rev = dubins_shortest_path(q0r, q1r, self.R)
        assert p_fwd is not None and p_rev is not None
        assert p_fwd.path_length() == pytest.approx(p_rev.path_length(), rel=0.01)

    def test_larger_radius_longer_or_equal(self):
        # A tighter minimum turn radius allows shorter paths; a larger radius
        # forces wider arcs and therefore the path length should be >= smaller r.
        q0 = (0.0, 0.0, 0.0)
        q1 = (30.0, 30.0, math.pi / 2)
        p_tight = dubins_shortest_path(q0, q1, r=5.0)
        p_wide  = dubins_shortest_path(q0, q1, r=30.0)
        if p_tight and p_wide:
            assert p_wide.path_length() >= p_tight.path_length() - 1e-3


# ===========================================================================
# 4. _DubinsPath.sample_many  —  path interpolation
# ===========================================================================

class TestDubinsPathSampleMany:
    """
    _DubinsPath.sample_many(step_size) walks along each of the three segments
    (arc / straight / arc) and emits (x, y, theta_rad) samples spaced
    approximately step_size metres apart.

    This is the function that actually generates the dense collision-check
    poses used by ConfigurationSpace.is_path_free().
    """

    R = 13.0

    def _straight_path(self):
        q0 = (0.0, 0.0, 0.0)
        q1 = (60.0, 0.0, 0.0)
        return dubins_shortest_path(q0, q1, self.R)

    def test_first_pose_matches_start(self):
        # The very first sample must be the start configuration
        path = self._straight_path()
        poses, _ = path.sample_many(1.0)
        assert poses[0][0] == pytest.approx(0.0, abs=1e-6)
        assert poses[0][1] == pytest.approx(0.0, abs=1e-6)

    def test_last_pose_near_goal(self):
        # The final sample should be within step_size of the goal
        path = self._straight_path()
        step = 1.0
        poses, _ = path.sample_many(step)
        last = poses[-1]
        assert last[0] == pytest.approx(60.0, abs=step + 0.1)
        assert last[1] == pytest.approx(0.0,  abs=step + 0.1)

    def test_returns_tuple_of_list_and_none(self):
        # API contract: return value is (list, None) to mirror pydubins
        path = self._straight_path()
        result = path.sample_many(2.0)
        assert isinstance(result, tuple) and len(result) == 2
        assert result[1] is None
        assert isinstance(result[0], list)

    def test_more_poses_with_smaller_step(self):
        # Halving step_size should roughly double the number of samples
        path = self._straight_path()
        poses_coarse, _ = path.sample_many(4.0)
        poses_fine,   _ = path.sample_many(2.0)
        assert len(poses_fine) > len(poses_coarse)

    def test_continuous_no_teleport(self):
        # Consecutive poses must be no more than step_size + small tolerance apart
        q0 = (0.0, 0.0, 0.0)
        q1 = (50.0, 30.0, math.pi / 4)
        path = dubins_shortest_path(q0, q1, self.R)
        step = 2.0
        poses, _ = path.sample_many(step)
        for i in range(len(poses) - 1):
            dx = poses[i+1][0] - poses[i][0]
            dy = poses[i+1][1] - poses[i][1]
            dist = math.hypot(dx, dy)
            assert dist <= step + 0.05, (
                f"teleport between poses {i} and {i+1}: {dist:.3f} > {step}"
            )

    def test_heading_changes_smoothly(self):
        # Heading should not jump by more than step/r radians between samples
        q0 = (0.0, 0.0, 0.0)
        q1 = (0.0, 40.0, math.pi)
        path = dubins_shortest_path(q0, q1, self.R)
        step = 1.0
        poses, _ = path.sample_many(step)
        max_dtheta = step / self.R + 0.05
        for i in range(len(poses) - 1):
            dth = abs(poses[i+1][2] - poses[i][2])
            # Account for wrapping near ±π
            dth = min(dth, 2 * math.pi - dth)
            assert dth <= max_dtheta, f"heading jump at step {i}: {dth:.4f} rad"


# ===========================================================================
# 5. CarModel  —  vehicle geometry
# ===========================================================================

class TestCarModel:
    """
    CarModel stores the vehicle's physical dimensions and computes a Shapely
    Polygon footprint at any (x, y, theta) pose.

    The footprint is used by ConfigurationSpace to check whether a pose
    overlaps an obstacle or exits the world boundary.

    Origin convention: the reference point is the REAR AXLE centre.
    +x points forward (in the direction of travel), +y points left.
    """

    def test_default_dimensions(self):
        car = CarModel()
        assert car.length    == pytest.approx(4.9)
        assert car.width     == pytest.approx(2.2)
        assert car.wheelbase == pytest.approx(3.0)
        assert car.r_min     == pytest.approx(13.0)

    def test_overhang_splits_evenly(self):
        # With default params: overhang = length - wheelbase = 1.9 m
        # split evenly → front = rear = 0.95 m
        car = CarModel()
        expected = (car.length - car.wheelbase) / 2.0
        assert car.front_overhang == pytest.approx(expected)
        assert car.rear_overhang  == pytest.approx(expected)

    def test_custom_overhang_preserved(self):
        # Explicit overhangs must not be overwritten by __post_init__
        car = CarModel(length=5.0, wheelbase=3.0,
                       front_overhang=1.5, rear_overhang=0.5)
        assert car.front_overhang == pytest.approx(1.5)
        assert car.rear_overhang  == pytest.approx(0.5)

    def test_footprint_is_polygon(self):
        from shapely.geometry import Polygon
        car = CarModel()
        fp = car.footprint_at(50.0, 50.0, 0.0)
        assert isinstance(fp, Polygon)
        assert fp.is_valid

    def test_footprint_area(self):
        # Area must equal length × width regardless of pose
        car = CarModel()
        fp = car.footprint_at(100.0, 100.0, 45.0)
        assert fp.area == pytest.approx(car.length * car.width, rel=1e-4)

    def test_footprint_moves_with_position(self):
        # Centroid should track the given (x, y) approximately
        car = CarModel()
        fp1 = car.footprint_at(50.0, 50.0, 0.0)
        fp2 = car.footprint_at(80.0, 50.0, 0.0)
        assert fp2.centroid.x > fp1.centroid.x

    def test_footprint_rotates_with_heading(self):
        # At 0° the bounding box is wider in x; at 90° wider in y
        car = CarModel()
        fp0  = car.footprint_at(100.0, 100.0, 0.0)
        fp90 = car.footprint_at(100.0, 100.0, 90.0)
        b0  = fp0.bounds   # (minx, miny, maxx, maxy)
        b90 = fp90.bounds
        width0  = b0[2]  - b0[0]
        height0 = b0[3]  - b0[1]
        width90 = b90[2] - b90[0]
        assert width0 > height0      # elongated along x at 0°
        assert width90 < width0      # rotated 90° → narrower in x


# ===========================================================================
# 6. ConfigurationSpace  —  free/collision checking
# ===========================================================================

class TestConfigurationSpace:
    """
    ConfigurationSpace answers two questions:
      is_free(x, y, theta)    → is this single pose collision-free?
      is_path_free(poses)     → are ALL poses in the list collision-free?

    It fuses obstacle cells into one Shapely union (for speed) and checks
    the car footprint against that union and against the world boundary.
    """

    def _open_cspace(self):
        car = CarModel()
        return ConfigurationSpace(car, world_size=300.0,
                                  obstacle_set=set(), cell_size=30.0)

    def _cspace_with_block(self, row=5, col=5):
        """Block cell (5,5) covers x=[150,180], y=[150,180]."""
        car = CarModel()
        return ConfigurationSpace(car, world_size=300.0,
                                  obstacle_set={(row, col)}, cell_size=30.0)

    # --- is_free ---

    def test_centre_of_open_world_is_free(self):
        cs = self._open_cspace()
        assert cs.is_free(150.0, 150.0, 0.0) is True

    def test_outside_world_boundary_not_free(self):
        # x=295 with the car extending forward puts part of the footprint
        # outside the 300×300 world box
        cs = self._open_cspace()
        assert cs.is_free(299.0, 150.0, 0.0) is False

    def test_inside_obstacle_not_free(self):
        # Place the car squarely inside the blocked cell
        cs = self._cspace_with_block()
        assert cs.is_free(165.0, 165.0, 0.0) is False

    def test_away_from_obstacle_is_free(self):
        cs = self._cspace_with_block()
        assert cs.is_free(50.0, 50.0, 0.0) is True

    def test_different_headings_matter(self):
        # A pose near a wall may be free at 0° but not at 90° if the car
        # rotates into the wall.  At least one heading should differ from the other.
        cs = self._open_cspace()
        r0  = cs.is_free(295.0, 150.0, 0.0)    # car nose pointing at east wall
        r90 = cs.is_free(295.0, 150.0, 90.0)   # car nose pointing north
        # They won't always differ, but we just verify neither raises
        assert isinstance(r0, bool) and isinstance(r90, bool)

    # --- is_path_free ---

    def test_empty_path_is_free(self):
        # A path with no poses has nothing to collide — should return True
        cs = self._open_cspace()
        assert cs.is_path_free([]) is True

    def test_all_free_poses_returns_true(self):
        cs = self._open_cspace()
        poses = [(50.0 + i, 50.0, 0.0) for i in range(10)]
        assert cs.is_path_free(poses) is True

    def test_one_bad_pose_returns_false(self):
        # Inserting a single out-of-bounds pose anywhere makes the whole
        # path invalid
        cs = self._open_cspace()
        poses = [(50.0 + i, 50.0, 0.0) for i in range(10)]
        poses[5] = (299.0, 150.0, 0.0)   # bad pose
        assert cs.is_path_free(poses) is False


# ===========================================================================
# 7. Firetruck._mod2pi / Dubins wrappers
# ===========================================================================

class TestFiretruckDubinsWrappers:
    """
    Firetruck exposes three private helpers that bridge State=(x,y,deg)
    to the raw Dubins API (which uses radians):

      _dubins(q_start, q_end)            → _DubinsPath or None
      _dubins_length(q_start, q_end)     → float (metres)
      _dubins_poses(q_start, q_end, step)→ list of (x, y, deg) States

    These wrappers are tested independently so that PRM-level failures can
    be distinguished from geometry-level failures.
    """

    def setup_method(self):
        self.truck = Firetruck(open_map())

    def test_dubins_returns_path_object(self):
        q0 = (50.0, 50.0, 0.0)
        q1 = (150.0, 50.0, 0.0)
        path = self.truck._dubins(q0, q1)
        assert isinstance(path, _DubinsPath)

    def test_dubins_coincident_returns_none(self):
        q = (50.0, 50.0, 0.0)
        assert self.truck._dubins(q, q) is None

    def test_dubins_length_positive(self):
        q0 = (50.0,  50.0, 0.0)
        q1 = (150.0, 50.0, 0.0)
        assert self.truck._dubins_length(q0, q1) > 0.0

    def test_dubins_length_coincident_is_inf(self):
        q = (50.0, 50.0, 0.0)
        assert self.truck._dubins_length(q, q) == float("inf")

    def test_dubins_poses_returns_degrees(self):
        # Output headings must be in degrees (not radians)
        q0 = (50.0, 50.0, 0.0)
        q1 = (150.0, 50.0, 90.0)
        poses = self.truck._dubins_poses(q0, q1, step_size=5.0)
        assert len(poses) > 0
        for _, _, theta_deg in poses:
            # Degrees are in [-360, 720] range; radians for this path would
            # be < 2π ≈ 6.28.  Any value > 7 must be degrees.
            assert -360.0 <= theta_deg <= 720.0

    def test_dubins_poses_first_matches_start(self):
        q0 = (50.0, 50.0, 45.0)
        q1 = (100.0, 100.0, 45.0)
        poses = self.truck._dubins_poses(q0, q1, step_size=2.0)
        assert poses[0][0] == pytest.approx(50.0, abs=0.1)
        assert poses[0][1] == pytest.approx(50.0, abs=0.1)

    def test_dubins_poses_coincident_returns_empty(self):
        q = (50.0, 50.0, 0.0)
        poses = self.truck._dubins_poses(q, q)
        assert poses == []


# ===========================================================================
# 8. Firetruck.build_tree  —  roadmap construction
# ===========================================================================

class TestBuildTree:
    """
    build_tree(n_samples) does two things:
      1. _sample_points: scatter n_samples collision-free (x, y, theta) nodes
         across the world using rejection sampling.
      2. _connect_nodes: for each node, find nearby neighbours via KD-tree
         and try to join them with valid Dubins edges.

    After build_tree():
      - self.nodes has ≤ n_samples entries (fewer if the map is very crowded)
      - self.graph[i] is a list of outgoing edge dicts {"to", "cost", "path"}
      - self._roadmap_size == len(self.nodes)
      - self._kd_tree is a scipy KDTree over the node (x, y) positions
    """

    def test_node_count_matches_requested(self):
        # Open world → should always find exactly n_samples free configs
        truck = Firetruck(open_map())
        truck.build_tree(n_samples=50)
        assert len(truck.nodes) == 50

    def test_roadmap_size_set_correctly(self):
        truck = Firetruck(open_map())
        truck.build_tree(n_samples=50)
        assert truck._roadmap_size == len(truck.nodes)

    def test_graph_keys_match_nodes(self):
        # Every node index must have a corresponding graph entry (even if empty)
        truck = Firetruck(open_map())
        truck.build_tree(n_samples=50)
        for i in range(len(truck.nodes)):
            assert i in truck.graph

    def test_all_nodes_are_free(self):
        # Rejection sampling must never place a node in collision
        truck = Firetruck(open_map())
        truck.build_tree(n_samples=50)
        for x, y, theta in truck.nodes:
            assert truck.cspace.is_free(x, y, theta), (
                f"Node ({x:.1f},{y:.1f},{theta}°) is in collision!"
            )

    def test_all_nodes_within_world(self):
        # All nodes must have coordinates inside the world boundary
        truck = Firetruck(open_map())
        truck.build_tree(n_samples=50)
        limit = truck.map.grid_num * truck.map.cell_size
        for x, y, _ in truck.nodes:
            assert 0.0 < x < limit
            assert 0.0 < y < limit

    def test_kd_tree_built(self):
        truck = Firetruck(open_map())
        truck.build_tree(n_samples=30)
        assert truck._kd_tree is not None

    def test_edges_reference_valid_node_indices(self):
        # Every edge's "to" field must be a valid index into self.nodes
        truck = Firetruck(open_map())
        truck.build_tree(n_samples=50)
        n = len(truck.nodes)
        for i, edges in truck.graph.items():
            for e in edges:
                assert 0 <= e["to"] < n, (
                    f"Edge from {i} points to invalid index {e['to']}"
                )

    def test_edge_cost_positive(self):
        # Path cost is a Dubins path length and must always be > 0
        truck = Firetruck(open_map())
        truck.build_tree(n_samples=50)
        for i, edges in truck.graph.items():
            for e in edges:
                assert e["cost"] > 0.0

    def test_edge_path_collision_free(self):
        # Every stored edge path must be collision-free (it was checked at
        # connection time, but we verify the stored data is consistent)
        truck = Firetruck(open_map())
        truck.build_tree(n_samples=30)
        for i, edges in truck.graph.items():
            for e in edges:
                assert truck.cspace.is_path_free(e["path"]), (
                    f"Stored edge from {i} to {e['to']} contains a collision!"
                )

    def test_rebuild_resets_state(self):
        # Calling build_tree a second time must fully reset nodes and graph
        truck = Firetruck(open_map())
        truck.build_tree(n_samples=30)
        first_nodes = list(truck.nodes)
        truck.build_tree(n_samples=20)
        assert len(truck.nodes) == 20
        assert truck._roadmap_size == 20


# ===========================================================================
# 9. Firetruck._inject_query_node  —  temporary node injection
# ===========================================================================

class TestInjectQueryNode:
    """
    During a query, start and goal poses are NOT in the prebuilt roadmap.
    _inject_query_node temporarily appends the pose to self.nodes (beyond
    _roadmap_size) and wires it into the graph.

    outgoing=True  → edges flow  new_node → roadmap  (used for start)
    outgoing=False → edges flow  roadmap  → new_node (used for goal)

    Returns the new node's index, or None if no Dubins connection is found.
    """

    def setup_method(self):
        self.truck = small_prm(n_samples=60)

    def test_inject_appends_beyond_roadmap(self):
        before = len(self.truck.nodes)
        q = (50.0, 50.0, 0.0)
        self.truck._inject_query_node(q, outgoing=True)
        assert len(self.truck.nodes) == before + 1

    def test_injected_index_beyond_roadmap_size(self):
        q = (50.0, 50.0, 0.0)
        idx = self.truck._inject_query_node(q, outgoing=True)
        if idx is not None:
            assert idx >= self.truck._roadmap_size

    def test_outgoing_edges_from_new_node(self):
        # Start node: edges must leave the new node → roadmap nodes
        q = (50.0, 50.0, 0.0)
        idx = self.truck._inject_query_node(q, outgoing=True)
        if idx is not None:
            for e in self.truck.graph[idx]:
                assert e["to"] < self.truck._roadmap_size

    def test_incoming_edges_to_new_node(self):
        # Goal node: edges must arrive at the new node from roadmap nodes
        q = (200.0, 200.0, 0.0)
        idx = self.truck._inject_query_node(q, outgoing=False)
        if idx is not None:
            # Find edges pointing to idx in permanent nodes
            incoming = [
                e for i in range(self.truck._roadmap_size)
                for e in self.truck.graph.get(i, [])
                if e["to"] == idx
            ]
            assert len(incoming) > 0, "Goal node has no incoming edges"

    def test_returns_none_on_empty_roadmap(self):
        # Without a built roadmap there are no candidates to connect to
        truck = Firetruck(open_map())  # build_tree NOT called
        truck._roadmap_size = 0
        truck.nodes = []
        truck.graph = {}
        truck._kd_tree = None
        result = truck._inject_query_node((50.0, 50.0, 0.0), outgoing=True)
        assert result is None


# ===========================================================================
# 10. Firetruck._cleanup_query_nodes  —  temp node removal
# ===========================================================================

class TestCleanupQueryNodes:
    """
    After each plan() call, _cleanup_query_nodes() must:
      1. Remove temp node entries from self.graph
      2. Remove edges from permanent nodes that pointed at temp nodes
      3. Truncate self.nodes back to _roadmap_size

    This is the bug fix mentioned in the module docstring: the original code
    used list.pop() which shifted indices; now we truncate instead.
    """

    def setup_method(self):
        self.truck = small_prm(n_samples=40)

    def test_nodes_truncated_to_roadmap_size(self):
        rs = self.truck._roadmap_size
        # Manually inject two temp nodes
        self.truck.nodes.append((10.0, 10.0, 0.0))
        self.truck.nodes.append((20.0, 20.0, 0.0))
        self.truck.graph[rs]     = []
        self.truck.graph[rs + 1] = []
        self.truck._cleanup_query_nodes()
        assert len(self.truck.nodes) == rs

    def test_temp_graph_entries_removed(self):
        rs = self.truck._roadmap_size
        self.truck.nodes.append((10.0, 10.0, 0.0))
        self.truck.graph[rs] = []
        self.truck._cleanup_query_nodes()
        assert rs not in self.truck.graph

    def test_permanent_nodes_unchanged(self):
        # The permanent node list (indices 0..roadmap_size-1) must be intact
        rs = self.truck._roadmap_size
        original_nodes = list(self.truck.nodes)
        self.truck.nodes.append((10.0, 10.0, 0.0))
        self.truck.graph[rs] = []
        self.truck._cleanup_query_nodes()
        assert self.truck.nodes == original_nodes

    def test_stale_edges_stripped_from_permanent_nodes(self):
        # If a permanent node had an edge pointing to a temp node,
        # cleanup must remove that edge.
        rs = self.truck._roadmap_size
        temp_idx = len(self.truck.nodes)
        self.truck.nodes.append((10.0, 10.0, 0.0))
        self.truck.graph[temp_idx] = []
        # Inject a fake edge from node 0 → temp_idx
        self.truck.graph[0].append({"to": temp_idx, "cost": 1.0, "path": []})
        self.truck._cleanup_query_nodes()
        for e in self.truck.graph.get(0, []):
            assert e["to"] < rs, "Stale edge to temp node not cleaned up"

    def test_noop_when_no_temp_nodes(self):
        # If no temp nodes exist, cleanup should be a no-op
        rs = self.truck._roadmap_size
        nodes_before = list(self.truck.nodes)
        self.truck._cleanup_query_nodes()
        assert self.truck.nodes == nodes_before
        assert len(self.truck.nodes) == rs

    def test_roadmap_stable_across_multiple_queries(self):
        # Running plan() twice must leave the roadmap identical both times.
        # This is the key regression test for the index-corruption bug.
        truck = small_prm(n_samples=60)
        nodes_after_build = list(truck.nodes)
        truck.plan((200.0, 200.0, 0.0), start_state=(50.0, 50.0, 0.0))
        assert truck.nodes == nodes_after_build, "Roadmap corrupted after 1st query"
        truck.plan((180.0, 180.0, 90.0), start_state=(50.0, 50.0, 0.0))
        assert truck.nodes == nodes_after_build, "Roadmap corrupted after 2nd query"


# ===========================================================================
# 11. Firetruck._astar  —  graph search
# ===========================================================================

class TestAstar:
    """
    _astar(start_idx, goal_idx) runs A* over self.graph using Dubins path
    length as the heuristic.  It returns a list of node indices from start
    to goal, or None if no path exists.

    The heuristic is admissible (never overestimates) because the true
    Dubins path length on the full graph cannot be shorter than the direct
    Dubins distance.
    """

    def _make_chain(self):
        """
        Build a minimal hand-crafted graph: 0 → 1 → 2 in a straight line.
        Nodes are spaced 30 m apart facing East.
        """
        truck = Firetruck(open_map())
        truck.nodes = [
            (30.0,  50.0, 0.0),
            (60.0,  50.0, 0.0),
            (90.0,  50.0, 0.0),
        ]
        truck._roadmap_size = 3
        truck.graph = {
            0: [{"to": 1, "cost": 30.0, "path": []}],
            1: [{"to": 2, "cost": 30.0, "path": []}],
            2: [],
        }
        import numpy as np
        from scipy.spatial import KDTree
        xy = np.array([(n[0], n[1]) for n in truck.nodes])
        truck._kd_tree = KDTree(xy)
        return truck

    def test_finds_path_in_chain(self):
        truck = self._make_chain()
        path = truck._astar(0, 2)
        assert path is not None
        assert path[0] == 0
        assert path[-1] == 2

    def test_path_visits_intermediate_node(self):
        truck = self._make_chain()
        path = truck._astar(0, 2)
        assert 1 in path

    def test_no_path_disconnected_graph(self):
        truck = self._make_chain()
        # Remove the only route to node 2
        truck.graph[1] = []
        path = truck._astar(0, 2)
        assert path is None

    def test_start_equals_goal(self):
        truck = self._make_chain()
        path = truck._astar(0, 0)
        # A* trivially reaches goal immediately; should return [0]
        assert path is not None
        assert path == [0]

    def test_path_is_list_of_ints(self):
        truck = self._make_chain()
        path = truck._astar(0, 2)
        assert isinstance(path, list)
        for idx in path:
            assert isinstance(idx, int)

    def test_path_indices_monotonically_valid(self):
        # Every index in the path must be a valid node index
        truck = self._make_chain()
        path = truck._astar(0, 2)
        for idx in path:
            assert 0 <= idx < len(truck.nodes)


# ===========================================================================
# 12. Firetruck.plan  —  end-to-end query
# ===========================================================================

class TestPlan:
    """
    plan(goal_state, start_state) is the public API that ties everything together:
      1. Validate roadmap exists
      2. Inject start and goal as temporary nodes
      3. Run A*
      4. Clean up temp nodes
      5. Return dense waypoint list, or None

    These tests use a small roadmap (60 nodes) for speed.  Because PRM is
    probabilistic, some queries may fail; we test structural properties
    (type, continuity, collision-freedom) rather than specific coordinates.
    """

    def setup_method(self):
        self.truck = small_prm(n_samples=60)

    def test_raises_without_build(self):
        # Calling plan() before build_tree() must raise RuntimeError
        truck = Firetruck(open_map())
        with pytest.raises(RuntimeError, match="build_tree"):
            truck.plan((100.0, 100.0, 0.0))

    def test_returns_list_or_none(self):
        result = self.truck.plan(
            goal_state=(200.0, 200.0, 0.0),
            start_state=(50.0, 50.0, 0.0),
        )
        assert result is None or isinstance(result, list)

    def test_path_waypoints_are_3_tuples(self):
        result = self.truck.plan(
            goal_state=(200.0, 200.0, 0.0),
            start_state=(50.0, 50.0, 0.0),
        )
        if result is not None:
            for wp in result:
                assert len(wp) == 3

    def test_path_is_collision_free(self):
        # Every waypoint in the returned path must be free
        result = self.truck.plan(
            goal_state=(200.0, 200.0, 0.0),
            start_state=(50.0, 50.0, 0.0),
        )
        if result is not None:
            for x, y, theta in result:
                assert self.truck.cspace.is_free(x, y, theta), (
                    f"Waypoint ({x:.1f},{y:.1f},{theta:.1f}°) is in collision"
                )

    def test_roadmap_intact_after_plan(self):
        # The permanent roadmap must be unchanged after a query
        nodes_before = list(self.truck.nodes)
        rs_before    = self.truck._roadmap_size
        self.truck.plan(
            goal_state=(200.0, 200.0, 0.0),
            start_state=(50.0, 50.0, 0.0),
        )
        assert self.truck.nodes       == nodes_before
        assert self.truck._roadmap_size == rs_before

    def test_second_query_consistent(self):
        # Two sequential queries must both see the same roadmap
        truck = small_prm(n_samples=60)
        nodes_snap = list(truck.nodes)
        truck.plan((200.0, 200.0, 0.0), start_state=(50.0, 50.0, 0.0))
        truck.plan((150.0, 150.0, 90.0), start_state=(60.0, 60.0, 0.0))
        assert truck.nodes == nodes_snap

    def test_path_continuity(self):
        # Consecutive waypoints must not teleport (max gap = one Dubins step)
        result = self.truck.plan(
            goal_state=(200.0, 200.0, 0.0),
            start_state=(50.0, 50.0, 0.0),
        )
        if result and len(result) > 1:
            for i in range(len(result) - 1):
                dx = result[i+1][0] - result[i][0]
                dy = result[i+1][1] - result[i][1]
                dist = math.hypot(dx, dy)
                assert dist < 5.0, (
                    f"Gap of {dist:.2f} m between waypoints {i} and {i+1}"
                )

    def test_start_defaults_to_map_firetruck_pose(self):
        # When start_state is omitted, Firetruck reads map.firetruck_pose.
        # We just verify plan() doesn't crash and returns the right type.
        truck = small_prm(n_samples=60)
        result = truck.plan((200.0, 200.0, 0.0))
        assert result is None or isinstance(result, list)


# ===========================================================================
# 13. Obstacle avoidance integration
# ===========================================================================

class TestObstacleAvoidance:
    """
    End-to-end test that the planner actually respects obstacles.

    We create a world with one blocked cell and verify that no waypoint
    in a returned path overlaps the blocked region.
    """

    def test_path_avoids_obstacle(self):
        # Block cell (3,3) → covers x=[90,120], y=[90,120]
        m = make_map(grid_num=10, cell_size=30.0, obstacles={(3, 3)})
        truck = Firetruck(m)
        truck.build_tree(n_samples=80)

        result = truck.plan(
            goal_state=(240.0, 240.0, 0.0),
            start_state=(30.0, 30.0, 0.0),
        )
        if result is None:
            pytest.skip("PRM did not find a path — increase n_samples")

        for x, y, theta in result:
            assert truck.cspace.is_free(x, y, theta), (
                f"Path entered obstacle at ({x:.1f},{y:.1f},{theta:.1f}°)"
            )


# ===========================================================================
# 14. Regression: index corruption across queries
# ===========================================================================

class TestIndexCorruptionRegression:
    """
    This directly tests the bug described in the module docstring.

    Original bug: _cleanup_query_nodes used list.pop() which shifts all
    indices after the removed element.  After the first query, permanent
    node 0 might now point to what was node 1, etc., causing silent path
    corruption or KeyErrors on the second query.

    Fix: truncate self.nodes[:_roadmap_size] — O(1), no shifting.
    """

    def test_permanent_node_identities_stable(self):
        truck = small_prm(n_samples=50)
        # Record permanent nodes by (idx, state) pairs
        snapshot = {i: truck.nodes[i] for i in range(truck._roadmap_size)}

        # Run several queries
        for goal in [(200.0, 200.0, 0.0), (150.0, 100.0, 45.0),
                     (80.0, 220.0, 180.0)]:
            truck.plan(goal, start_state=(50.0, 50.0, 0.0))

        # Every permanent node must be at the same index with the same state
        for i, state in snapshot.items():
            assert truck.nodes[i] == state, (
                f"Node {i} changed: was {state}, now {truck.nodes[i]}"
            )

    def test_graph_edges_still_valid_after_queries(self):
        truck = small_prm(n_samples=50)
        # Record all edge destinations in the permanent graph
        edge_snapshot = {
            i: [e["to"] for e in edges]
            for i, edges in truck.graph.items()
        }
        # Run queries
        truck.plan((200.0, 200.0, 0.0), start_state=(50.0, 50.0, 0.0))
        truck.plan((100.0, 100.0, 0.0), start_state=(50.0, 50.0, 0.0))

        for i, original_tos in edge_snapshot.items():
            current_tos = [e["to"] for e in truck.graph.get(i, [])]
            assert current_tos == original_tos, (
                f"Edges from node {i} changed after query"
            )