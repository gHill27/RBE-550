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

# ── Step 1: stdlib only — nothing from the project yet ──────────────────────
import math
import os
import sys
import types
from typing import List, Optional, Tuple
from unittest.mock import MagicMock

# ── Step 2: register ALL stub modules before touching sys.path ───────────────
#
# Any name that firetruck.py (or its transitive imports) tries to import
# must appear here.  Add extras freely — a stub that is never used is harmless.

def _stub(name: str) -> types.ModuleType:
    """Create an empty module stub and register it in sys.modules."""
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


# External project modules that pull in pygame, ROS, heavy GUI libs, etc.
_mg  = _stub("Map_Generator")
_ps  = _stub("pathSimulator")
_pv  = _stub("pathVisualizer")
_ps2 = _stub("pathSimulator2")
_pv2 = _stub("pathVisualizer2")

# Populate the symbols that firetruck.py actually imports by name
_mg.Status = MagicMock()


class _FakeMap:
    """Minimal stand-in for Map — only the attributes Firetruck touches."""
    def __init__(
        self,
        grid_num: int = 50,
        cell_size: float = 5.0,
        obstacle_set=None,
        firetruck_pose=(10.0, 10.0),
    ):
        self.grid_num       = grid_num
        self.cell_size      = cell_size
        self.obstacle_set   = obstacle_set or set()
        self.firetruck_pose = firetruck_pose


_mg.Map = _FakeMap

# PlannerVisualizer is imported by name in firetruck.py
_pv.PlannerVisualizer  = MagicMock()
_pv2.PlannerVisualizer = MagicMock()
_ps.PathSimulator      = MagicMock()
_ps2.PathSimulator     = MagicMock()

# ── Step 3: NOW we can safely add the project root to sys.path ───────────────
_tests_dir  = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_tests_dir)
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

# ── Step 4: import project code — stubs are already in sys.modules ───────────
import pytest  # noqa: E402  (pytest must come after sys.path is set)
from shapely.geometry import Point, Polygon, box  # noqa: E402

from firetruck import (  # noqa: E402
    CarModel,
    ConfigurationSpace,
    DubinsEdge,
    DubinsPlanner,
    Firetruck,
)

# ---------------------------------------------------------------------------
# Shared constants matching the firetruck spec
# ---------------------------------------------------------------------------

CAR_LENGTH    = 4.9
CAR_WIDTH     = 2.2
CAR_WHEELBASE = 3.0
CAR_RMIN      = 13.0
CAR_VMAX      = 10.0

WORLD_SIZE = 250.0
CELL_SIZE  = 5.0


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def car() -> CarModel:
    return CarModel(
        length=CAR_LENGTH, width=CAR_WIDTH,
        wheelbase=CAR_WHEELBASE, r_min=CAR_RMIN, v_max=CAR_VMAX,
    )


@pytest.fixture
def empty_cspace(car) -> ConfigurationSpace:
    """250 m × 250 m world with no obstacles."""
    return ConfigurationSpace(
        car=car, world_size=WORLD_SIZE,
        obstacle_set=set(), cell_size=CELL_SIZE,
    )


@pytest.fixture
def cspace_with_obstacle(car) -> ConfigurationSpace:
    """World with a single 5 × 5 m obstacle at grid cell (10, 10)
       → world coords x=[50,55], y=[50,55]."""
    return ConfigurationSpace(
        car=car, world_size=WORLD_SIZE,
        obstacle_set={(10, 10)}, cell_size=CELL_SIZE,
    )


@pytest.fixture
def dubins(car, empty_cspace) -> DubinsPlanner:
    return DubinsPlanner(car, empty_cspace)


@pytest.fixture
def fake_map() -> _FakeMap:
    return _FakeMap(
        grid_num=50, cell_size=5.0,
        obstacle_set=set(),
        firetruck_pose=(10.0, 10.0),
    )


@pytest.fixture
def truck(fake_map) -> Firetruck:
    """Firetruck with no visualiser — safe for headless testing."""
    return Firetruck(fake_map, plot=False)


@pytest.fixture
def built_truck(truck) -> Firetruck:
    """Firetruck with a small roadmap pre-built (50 nodes, fast)."""
    truck.build_tree(n_samples=50)
    return truck


# ---------------------------------------------------------------------------
# Helper: permanent roadmap size
# ---------------------------------------------------------------------------

def _roadmap_size(truck: Firetruck) -> int:
    """
    Return the number of *permanent* roadmap nodes.

    The reference implementation exposes this as truck._roadmap_size.
    The current firetruck.py does not store it explicitly, so we derive
    it as len(nodes) right after build_tree() — the fixture built_truck
    calls build_tree() before returning, so at that point no temp nodes
    have been injected yet.
    """
    return getattr(truck, "_roadmap_size", len(truck.nodes))


# ===========================================================================
# CarModel tests
# ===========================================================================

class TestCarModel:

    def test_default_dimensions(self, car):
        assert car.length    == CAR_LENGTH
        assert car.width     == CAR_WIDTH
        assert car.wheelbase == CAR_WHEELBASE
        assert car.r_min     == CAR_RMIN
        assert car.v_max     == CAR_VMAX

    def test_symmetric_overhang(self, car):
        expected = (CAR_LENGTH - CAR_WHEELBASE) / 2.0
        assert car.front_overhang == pytest.approx(expected)
        assert car.rear_overhang  == pytest.approx(expected)

    def test_asymmetric_overhang_override(self):
        c = CarModel(
            length=4.9, width=2.2, wheelbase=3.0,
            r_min=13.0, v_max=10.0,
            front_overhang=1.2, rear_overhang=0.7,
        )
        assert c.front_overhang == 1.2
        assert c.rear_overhang  == 0.7

    def test_local_footprint_is_polygon(self, car):
        assert isinstance(car._local_footprint, Polygon)

    def test_local_footprint_area(self, car):
        assert car._local_footprint.area == pytest.approx(
            car.length * car.width, rel=1e-3
        )

    def test_footprint_at_returns_polygon(self, car):
        fp = car.footprint_at(0.0, 0.0, 0.0)
        assert isinstance(fp, Polygon)

    def test_footprint_area_preserved(self, car):
        fp = car.footprint_at(100.0, 50.0, 0.0)
        assert fp.area == pytest.approx(car.length * car.width, rel=1e-3)

    def test_footprint_area_preserved_after_rotation(self, car):
        for theta in [0, 45, 90, 135, 180, 270]:
            fp = car.footprint_at(125.0, 125.0, float(theta))
            assert fp.area == pytest.approx(
                car.length * car.width, rel=1e-3
            ), f"Area wrong at theta={theta}"

    def test_footprint_moves_with_position(self, car):
        fp1 = car.footprint_at(50.0, 50.0, 0.0)
        fp2 = car.footprint_at(100.0, 50.0, 0.0)
        assert not fp1.equals(fp2)
        assert fp1.centroid.x == pytest.approx(fp2.centroid.x - 50.0, abs=0.5)

    def test_footprint_90deg_swaps_bounding_box(self, car):
        fp0  = car.footprint_at(125.0, 125.0, 0.0)
        fp90 = car.footprint_at(125.0, 125.0, 90.0)
        b0   = fp0.bounds   # (minx, miny, maxx, maxy)
        b90  = fp90.bounds
        w0,  h0  = b0[2] - b0[0],   b0[3] - b0[1]
        w90, h90 = b90[2] - b90[0], b90[3] - b90[1]
        assert w90 == pytest.approx(h0,  abs=0.3)
        assert h90 == pytest.approx(w0,  abs=0.3)

    def test_max_steering_angle_positive(self, car):
        assert car.max_steering_angle_deg > 0

    def test_max_steering_angle_value(self, car):
        expected = math.degrees(math.atan(car.wheelbase / car.r_min))
        assert car.max_steering_angle_deg == pytest.approx(expected)

    def test_repr_contains_dimensions(self, car):
        r = repr(car)
        assert "4.9" in r
        assert "2.2" in r
        assert "13.0" in r


# ===========================================================================
# ConfigurationSpace tests
# ===========================================================================

class TestConfigurationSpace:

    # ── construction ──────────────────────────────────────────────────────

    def test_empty_obstacle_set_gives_none_geometry(self, empty_cspace):
        assert empty_cspace.full_obstacle_geometry is None

    def test_obstacle_geometry_built(self, cspace_with_obstacle):
        assert cspace_with_obstacle.full_obstacle_geometry is not None

    def test_obstacle_geometry_covers_cell_centre(self, cspace_with_obstacle):
        centre = Point(52.5, 52.5)   # cell (10,10) → world [50,55]×[50,55]
        assert cspace_with_obstacle.full_obstacle_geometry.contains(centre)

    def test_multiple_obstacles_area(self, car):
        cs = ConfigurationSpace(
            car=car, world_size=WORLD_SIZE,
            obstacle_set={(0, 0), (1, 0), (2, 0)},
            cell_size=CELL_SIZE,
        )
        assert cs.full_obstacle_geometry.area == pytest.approx(
            3 * CELL_SIZE * CELL_SIZE, rel=1e-3
        )

    # ── is_free: bounds ───────────────────────────────────────────────────

    def test_centre_is_free(self, empty_cspace):
        assert empty_cspace.is_free(125.0, 125.0, 0.0) is True

    def test_negative_x_not_free(self, empty_cspace):
        assert empty_cspace.is_free(-10.0, 125.0, 0.0) is False

    def test_negative_y_not_free(self, empty_cspace):
        assert empty_cspace.is_free(125.0, -10.0, 0.0) is False

    def test_beyond_world_x_not_free(self, empty_cspace):
        assert empty_cspace.is_free(260.0, 125.0, 0.0) is False

    def test_beyond_world_y_not_free(self, empty_cspace):
        assert empty_cspace.is_free(125.0, 260.0, 0.0) is False

    def test_too_close_to_left_wall(self, empty_cspace):
        assert empty_cspace.is_free(0.5, 125.0, 0.0) is False

    def test_comfortable_margin_is_free(self, empty_cspace):
        assert empty_cspace.is_free(20.0, 125.0, 0.0) is True

    # ── is_free: obstacles ────────────────────────────────────────────────

    def test_inside_obstacle_not_free(self, cspace_with_obstacle):
        assert cspace_with_obstacle.is_free(52.5, 52.5, 0.0) is False

    def test_far_from_obstacle_is_free(self, cspace_with_obstacle):
        assert cspace_with_obstacle.is_free(200.0, 200.0, 0.0) is True

    def test_footprint_clipping_obstacle_not_free(self, cspace_with_obstacle):
        # Obstacle x=[50,55]; car heading 0, close enough to clip
        assert cspace_with_obstacle.is_free(48.0, 52.5, 0.0) is False

    def test_car_clear_of_obstacle_is_free(self, cspace_with_obstacle):
        # Far enough left that the footprint doesn't reach x=50
        assert cspace_with_obstacle.is_free(30.0, 52.5, 0.0) is True

    # ── is_path_free ──────────────────────────────────────────────────────

    def test_empty_pose_list_is_free(self, empty_cspace):
        assert empty_cspace.is_path_free([]) is True

    def test_all_free_poses_pass(self, empty_cspace):
        poses = [(125.0 + i, 125.0, 0.0) for i in range(10)]
        assert empty_cspace.is_path_free(poses) is True

    def test_one_bad_pose_fails_entire_path(self, cspace_with_obstacle):
        poses = [
            (125.0, 125.0, 0.0),
            (52.5,  52.5,  0.0),   # inside obstacle
            (200.0, 200.0, 0.0),
        ]
        assert cspace_with_obstacle.is_path_free(poses) is False

    def test_fails_fast_does_not_check_beyond_collision(self, cspace_with_obstacle):
        """Verify early-exit: poses after the first collision are never checked."""
        checked = []

        class TrackingCSpace(ConfigurationSpace):
            def is_free(self, x, y, theta):
                checked.append((x, y))
                return super().is_free(x, y, theta)

        ts = TrackingCSpace(
            car=cspace_with_obstacle.car,
            world_size=WORLD_SIZE,
            obstacle_set={(10, 10)},
            cell_size=CELL_SIZE,
        )
        ts.is_path_free([
            (52.5,  52.5,  0.0),   # collision here
            (200.0, 200.0, 0.0),   # should never be reached
        ])
        assert (200.0, 200.0) not in checked


# ===========================================================================
# DubinsEdge tests
# ===========================================================================

class TestDubinsEdge:

    def test_fields_stored_correctly(self):
        e = DubinsEdge(
            node_from=0, node_to=1,
            cost=42.5, path_type='LSL',
            seg_lengths=(10.0, 20.0, 12.5),
        )
        assert e.node_from   == 0
        assert e.node_to     == 1
        assert e.cost        == pytest.approx(42.5)
        assert e.path_type   == 'LSL'
        assert e.seg_lengths == (10.0, 20.0, 12.5)

    def test_default_seg_lengths(self):
        e = DubinsEdge(node_from=0, node_to=1, cost=1.0, path_type='RSR')
        assert e.seg_lengths == (0.0, 0.0, 0.0)

    def test_all_path_types_accepted(self):
        for pt in ['LSL', 'RSR', 'LSR', 'RSL', 'RLR', 'LRL']:
            e = DubinsEdge(node_from=0, node_to=1, cost=1.0, path_type=pt)
            assert e.path_type == pt


# ===========================================================================
# DubinsPlanner tests
# ===========================================================================

class TestDubinsPlanner:

    # ── _mod2pi ──────────────────────────────────────────────────────────

    def test_mod2pi_zero(self, dubins):
        assert dubins._mod2pi(0.0) == pytest.approx(0.0)

    def test_mod2pi_full_circle(self, dubins):
        assert dubins._mod2pi(2 * math.pi) == pytest.approx(0.0, abs=1e-9)

    def test_mod2pi_negative_quarter(self, dubins):
        result = dubins._mod2pi(-math.pi / 2)
        assert result == pytest.approx(3 * math.pi / 2, rel=1e-6)

    def test_mod2pi_result_in_range(self, dubins):
        for angle in [-10.0, -1.0, 0.0, 1.0, 7.0, 100.0]:
            r = dubins._mod2pi(angle)
            assert 0.0 <= r < 2 * math.pi, f"mod2pi({angle}) = {r} out of [0, 2π)"

    # ── _shortest_path ────────────────────────────────────────────────────

    def test_coincident_poses_returns_none(self, dubins):
        q = (100.0, 100.0, 0.0)
        assert dubins._shortest_path(q, q) is None

    def test_returns_three_element_tuple(self, dubins):
        result = dubins._shortest_path((100.0, 100.0, 0.0), (150.0, 100.0, 0.0))
        assert result is not None
        assert len(result) == 3

    def test_path_type_is_valid_word(self, dubins):
        result = dubins._shortest_path((100.0, 100.0, 0.0), (150.0, 120.0, 45.0))
        assert result is not None
        assert result[0] in DubinsPlanner._PATH_TYPES

    def test_total_length_is_positive(self, dubins):
        result = dubins._shortest_path((100.0, 100.0, 0.0), (160.0, 130.0, 90.0))
        assert result is not None
        assert result[2] > 0.0

    def test_seg_lengths_all_non_negative(self, dubins):
        for q1 in [
            (150.0, 100.0,   0.0),
            (150.0, 120.0,  90.0),
            (80.0,  100.0, 180.0),
            (100.0, 150.0, 270.0),
        ]:
            result = dubins._shortest_path((100.0, 100.0, 0.0), q1)
            if result is not None:
                t, p, q = result[1]
                assert t >= 0, f"t<0 for {q1}"
                assert p >= 0, f"p<0 for {q1}"
                assert q >= 0, f"q<0 for {q1}"

    def test_seg_lengths_sum_to_total(self, dubins):
        q0 = (100.0, 100.0, 30.0)
        q1 = (160.0, 130.0, 75.0)
        result = dubins._shortest_path(q0, q1)
        assert result is not None
        t, p, q_seg = result[1]
        assert t + p + q_seg == pytest.approx(result[2], rel=1e-4)

    def test_path_length_geq_euclidean(self, dubins):
        q0 = (100.0, 100.0, 0.0)
        q1 = (150.0, 130.0, 0.0)
        euc = math.hypot(50.0, 30.0)
        length = dubins.path_length(q0, q1)
        assert length >= euc - 1e-6   # Dubins ≥ straight-line

    def test_straight_aligned_path_length_geq_distance(self, dubins):
        """Same heading, points aligned — Dubins ≥ Euclidean distance."""
        q0 = (100.0, 100.0, 0.0)
        q1 = (150.0, 100.0, 0.0)
        length = dubins.path_length(q0, q1)
        assert length >= 50.0 - 1e-6

    def test_opposite_heading_longer_than_straight(self, dubins):
        q0 = (100.0, 100.0,   0.0)
        q1 = (150.0, 100.0, 180.0)
        assert dubins.path_length(q0, q1) > 50.0

    def test_path_length_asymmetric(self, dubins):
        """Dubins(A→B) ≠ Dubins(B→A) in general."""
        q0 = (100.0, 100.0,  0.0)
        q1 = (150.0, 100.0, 90.0)
        assert dubins.path_length(q0, q1) != pytest.approx(
            dubins.path_length(q1, q0), rel=0.01
        )

    def test_path_length_coincident_is_inf(self, dubins):
        q = (100.0, 100.0, 0.0)
        assert dubins.path_length(q, q) == float('inf')

    # ── _interpolate ──────────────────────────────────────────────────────

    def test_interpolate_starts_at_q_start(self, dubins):
        q0 = (100.0, 100.0, 0.0)
        q1 = (150.0, 100.0, 0.0)
        result = dubins._shortest_path(q0, q1)
        assert result is not None
        poses = dubins._interpolate(q0, result[1], result[0], step_size=1.0)
        assert poses[0][0] == pytest.approx(100.0, abs=0.1)
        assert poses[0][1] == pytest.approx(100.0, abs=0.1)

    def test_interpolate_ends_near_q_end(self, dubins):
        q0 = (100.0, 100.0, 0.0)
        q1 = (150.0, 100.0, 0.0)
        result = dubins._shortest_path(q0, q1)
        assert result is not None
        poses = dubins._interpolate(q0, result[1], result[0], step_size=0.5)
        assert poses[-1][0] == pytest.approx(150.0, abs=1.5)
        assert poses[-1][1] == pytest.approx(100.0, abs=1.5)

    def test_interpolate_finer_step_more_poses(self, dubins):
        q0 = (100.0, 100.0, 0.0)
        q1 = (150.0, 100.0, 0.0)
        result = dubins._shortest_path(q0, q1)
        assert result is not None
        coarse = dubins._interpolate(q0, result[1], result[0], step_size=5.0)
        fine   = dubins._interpolate(q0, result[1], result[0], step_size=0.5)
        assert len(fine) > len(coarse)

    def test_interpolate_all_poses_are_3tuples(self, dubins):
        q0 = (100.0, 100.0, 45.0)
        q1 = (140.0, 120.0, 90.0)
        result = dubins._shortest_path(q0, q1)
        assert result is not None
        poses = dubins._interpolate(q0, result[1], result[0], step_size=1.0)
        for pose in poses:
            assert len(pose) == 3

    def test_interpolate_no_nans(self, dubins):
        q0 = (100.0, 100.0,  0.0)
        q1 = (160.0, 130.0, 45.0)
        result = dubins._shortest_path(q0, q1)
        assert result is not None
        poses = dubins._interpolate(q0, result[1], result[0], step_size=1.0)
        for x, y, theta in poses:
            assert math.isfinite(x),     f"x={x} not finite"
            assert math.isfinite(y),     f"y={y} not finite"
            assert math.isfinite(theta), f"theta={theta} not finite"

    # ── compute_edge ──────────────────────────────────────────────────────

    def test_compute_edge_clear_path_returns_edge(self, dubins):
        edge = dubins.compute_edge(0, (100.0, 100.0, 0.0), 1, (150.0, 100.0, 0.0))
        assert edge is not None
        assert isinstance(edge, DubinsEdge)

    def test_compute_edge_stores_correct_indices(self, dubins):
        edge = dubins.compute_edge(7, (100.0, 100.0, 0.0), 42, (150.0, 100.0, 0.0))
        assert edge is not None
        assert edge.node_from == 7
        assert edge.node_to   == 42

    def test_compute_edge_cost_positive(self, dubins):
        edge = dubins.compute_edge(0, (100.0, 100.0, 0.0), 1, (150.0, 100.0, 0.0))
        assert edge is not None
        assert edge.cost > 0.0

    def test_compute_edge_path_type_valid(self, dubins):
        edge = dubins.compute_edge(0, (100.0, 100.0, 0.0), 1, (150.0, 110.0, 30.0))
        assert edge is not None
        assert edge.path_type in DubinsPlanner._PATH_TYPES

    def test_compute_edge_coincident_returns_none(self, dubins):
        q = (100.0, 100.0, 0.0)
        assert dubins.compute_edge(0, q, 1, q) is None

    def test_compute_edge_through_obstacle_returns_none(self, car, cspace_with_obstacle):
        dp = DubinsPlanner(car, cspace_with_obstacle)
        # Straight shot through obstacle at x=[50,55], y=[50,55]
        q0 = (40.0, 52.5, 0.0)
        q1 = (70.0, 52.5, 0.0)
        assert dp.compute_edge(0, q0, 1, q1) is None

    # ── interpolate_edge ─────────────────────────────────────────────────

    def test_interpolate_edge_returns_list(self, dubins):
        q0   = (100.0, 100.0, 0.0)
        q1   = (150.0, 100.0, 0.0)
        edge = dubins.compute_edge(0, q0, 1, q1)
        assert edge is not None
        poses = dubins.interpolate_edge(q0, edge, step_size=1.0)
        assert isinstance(poses, list)
        assert len(poses) > 0

    def test_interpolate_edge_first_pose_at_source(self, dubins):
        q0   = (100.0, 100.0, 0.0)
        q1   = (150.0, 100.0, 0.0)
        edge = dubins.compute_edge(0, q0, 1, q1)
        assert edge is not None
        poses = dubins.interpolate_edge(q0, edge, step_size=1.0)
        assert poses[0][0] == pytest.approx(100.0, abs=0.1)
        assert poses[0][1] == pytest.approx(100.0, abs=0.1)


# ===========================================================================
# Firetruck._sample_points tests
# ===========================================================================

class TestSamplePoints:

    def test_nodes_are_3tuples(self, truck):
        truck._sample_points(n_samples=20)
        for node in truck.nodes:
            assert len(node) == 3, f"Expected 3-tuple, got {node}"

    def test_theta_stored_in_nodes(self, truck):
        truck._sample_points(n_samples=20)
        for x, y, theta in truck.nodes:
            assert 0.0 <= theta < 360.0

    def test_all_nodes_within_world_bounds(self, truck):
        truck._sample_points(n_samples=50)
        limit = truck.map.grid_num * truck.map.cell_size
        for x, y, theta in truck.nodes:
            assert 0 < x < limit
            assert 0 < y < limit

    def test_all_nodes_collision_free(self, truck):
        truck._sample_points(n_samples=30)
        for x, y, theta in truck.nodes:
            assert truck.cspace.is_free(x, y, theta), \
                f"Node ({x:.1f},{y:.1f},{theta}) is in collision"

    def test_graph_keys_match_indices(self, truck):
        truck._sample_points(n_samples=25)
        for i in range(len(truck.nodes)):
            assert i in truck.graph

    def test_graph_values_empty_after_sampling(self, truck):
        truck._sample_points(n_samples=25)
        for edges in truck.graph.values():
            assert edges == []

    def test_kd_tree_built(self, truck):
        truck._sample_points(n_samples=20)
        assert truck._kd_tree is not None

    def test_kd_tree_size_matches_nodes(self, truck):
        truck._sample_points(n_samples=30)
        assert truck._kd_tree.n == len(truck.nodes)

    def test_does_not_exceed_requested_samples(self, truck):
        truck._sample_points(n_samples=40)
        assert len(truck.nodes) <= 40

    def test_resets_on_second_call(self, truck):
        truck._sample_points(n_samples=30)
        truck._sample_points(n_samples=15)
        assert len(truck.nodes) <= 15


# ===========================================================================
# Firetruck._connect_nodes tests
# ===========================================================================

class TestConnectNodes:

    def test_edges_are_dubins_edge_objects(self, truck):
        truck._sample_points(n_samples=30)
        truck._connect_nodes(k_neighbors=5, r_connect=50.0)
        for edges in truck.graph.values():
            for e in edges:
                assert isinstance(e, DubinsEdge)

    def test_edge_costs_positive(self, truck):
        truck._sample_points(n_samples=30)
        truck._connect_nodes(k_neighbors=5, r_connect=50.0)
        for edges in truck.graph.values():
            for e in edges:
                assert e.cost > 0.0

    def test_no_self_loops(self, truck):
        truck._sample_points(n_samples=30)
        truck._connect_nodes(k_neighbors=5, r_connect=50.0)
        for i, edges in truck.graph.items():
            for e in edges:
                assert e.node_to != i

    def test_edge_node_to_in_valid_range(self, truck):
        truck._sample_points(n_samples=30)
        truck._connect_nodes(k_neighbors=5, r_connect=50.0)
        n = len(truck.nodes)
        for edges in truck.graph.values():
            for e in edges:
                assert 0 <= e.node_to < n

    def test_some_edges_are_built(self, truck):
        truck._sample_points(n_samples=40)
        truck._connect_nodes(k_neighbors=10, r_connect=60.0)
        total = sum(len(v) for v in truck.graph.values())
        assert total > 0


# ===========================================================================
# Firetruck.build_tree tests
# ===========================================================================

class TestBuildTree:

    def test_nodes_not_empty_after_build(self, truck):
        truck.build_tree(n_samples=30)
        assert len(truck.nodes) > 0

    def test_graph_populated_after_build(self, truck):
        truck.build_tree(n_samples=30)
        assert len(truck.graph) > 0

    def test_kd_tree_exists_after_build(self, truck):
        truck.build_tree(n_samples=20)
        assert truck._kd_tree is not None

    def test_all_nodes_are_3tuples_after_build(self, truck):
        truck.build_tree(n_samples=20)
        for node in truck.nodes:
            assert len(node) == 3


# ===========================================================================
# Firetruck._inject_query_node tests
# ===========================================================================

class TestInjectQueryNode:

    def test_start_node_appended(self, built_truck):
        before = len(built_truck.nodes)
        idx = built_truck._inject_query_node((125.0, 125.0, 0.0), outgoing=True)
        assert idx is not None
        assert len(built_truck.nodes) == before + 1

    def test_goal_node_appended_after_start(self, built_truck):
        built_truck._inject_query_node((125.0, 125.0, 0.0), outgoing=True)
        before = len(built_truck.nodes)
        idx = built_truck._inject_query_node((200.0, 200.0, 0.0), outgoing=False)
        assert idx is not None
        assert len(built_truck.nodes) == before + 1

    def test_start_has_outgoing_edges_in_graph(self, built_truck):
        idx = built_truck._inject_query_node((125.0, 125.0, 0.0), outgoing=True)
        if idx is not None:
            assert len(built_truck.graph[idx]) > 0

    def test_goal_has_incoming_edges_from_roadmap(self, built_truck):
        idx = built_truck._inject_query_node((125.0, 125.0, 0.0), outgoing=False)
        if idx is not None:
            incoming = [
                e for edges in built_truck.graph.values()
                for e in edges if e.node_to == idx
            ]
            assert len(incoming) > 0

    def test_returns_none_for_out_of_world_pose(self, built_truck):
        idx = built_truck._inject_query_node((9999.0, 9999.0, 0.0), outgoing=True)
        # Clean up whatever was appended before asserting
        built_truck._cleanup_query_nodes(idx, None)
        assert idx is None


# ===========================================================================
# Firetruck._cleanup_query_nodes tests
# ===========================================================================

class TestCleanupQueryNodes:
    """
    firetruck.py signature:
        _cleanup_query_nodes(self, start_idx, goal_idx)

    Both arguments can be None (e.g., injection failed).
    """

    def test_nodes_restored_after_cleanup(self, built_truck):
        before = len(built_truck.nodes)
        s = built_truck._inject_query_node((125.0, 125.0, 0.0), outgoing=True)
        g = built_truck._inject_query_node((200.0, 200.0, 0.0), outgoing=False)
        built_truck._cleanup_query_nodes(s, g)
        assert len(built_truck.nodes) == before

    def test_temp_graph_entries_removed(self, built_truck):
        s = built_truck._inject_query_node((125.0, 125.0, 0.0), outgoing=True)
        g = built_truck._inject_query_node((200.0, 200.0, 0.0), outgoing=False)
        built_truck._cleanup_query_nodes(s, g)
        if s is not None:
            assert s not in built_truck.graph
        if g is not None:
            assert g not in built_truck.graph

    def test_permanent_node_values_unchanged(self, built_truck):
        original = list(built_truck.nodes)
        s = built_truck._inject_query_node((125.0, 125.0, 0.0), outgoing=True)
        g = built_truck._inject_query_node((200.0, 200.0, 0.0), outgoing=False)
        built_truck._cleanup_query_nodes(s, g)
        assert built_truck.nodes == original

    def test_no_edges_pointing_to_removed_nodes(self, built_truck):
        s = built_truck._inject_query_node((125.0, 125.0, 0.0), outgoing=True)
        g = built_truck._inject_query_node((200.0, 200.0, 0.0), outgoing=False)
        removed = {i for i in [s, g] if i is not None}
        built_truck._cleanup_query_nodes(s, g)
        for edges in built_truck.graph.values():
            for e in edges:
                assert e.node_to not in removed

    def test_none_args_do_not_crash(self, built_truck):
        """Cleanup with both None is a no-op and must not raise."""
        before = len(built_truck.nodes)
        built_truck._cleanup_query_nodes(None, None)
        assert len(built_truck.nodes) == before

    def test_repeated_inject_cleanup_stable(self, built_truck):
        """Core regression: repeated plan()-style cycles must not grow nodes."""
        before = len(built_truck.nodes)
        for _ in range(5):
            s = built_truck._inject_query_node((125.0, 125.0, 0.0), outgoing=True)
            g = built_truck._inject_query_node((200.0, 200.0, 0.0), outgoing=False)
            built_truck._cleanup_query_nodes(s, g)
            assert len(built_truck.nodes) == before, \
                "Node list grew — cleanup is not fully reversing injection"


# ===========================================================================
# Firetruck._astar tests
# ===========================================================================

class TestAstar:
    """
    _astar uses came_from: Dict[int, Tuple[int, DubinsEdge]]
    so test fixtures must match that format.
    """

    def _wire_linear_graph(self, truck: Firetruck):
        """
        Four nodes in a straight line, wired 0→1→2→3 with real DubinsEdges
        so _astar can look up self.nodes[idx] for the heuristic.
        """
        truck.nodes = [
            (50.0,  125.0, 0.0),
            (80.0,  125.0, 0.0),
            (110.0, 125.0, 0.0),
            (140.0, 125.0, 0.0),
        ]
        truck.graph = {i: [] for i in range(4)}

        for i in range(3):
            edge = DubinsEdge(
                node_from=i, node_to=i + 1,
                cost=30.0, path_type='LSL',
                seg_lengths=(10.0, 10.0, 10.0),
            )
            truck.graph[i].append(edge)

    def test_finds_path_on_linear_chain(self, truck):
        self._wire_linear_graph(truck)
        path = truck._astar(0, 3)
        assert path is not None
        assert path[0] == 0
        assert path[-1] == 3

    def test_path_visits_nodes_in_order(self, truck):
        self._wire_linear_graph(truck)
        path = truck._astar(0, 3)
        assert path == [0, 1, 2, 3]

    def test_disconnected_graph_returns_none(self, truck):
        self._wire_linear_graph(truck)
        truck.graph[1] = []          # cut 1→2 link
        assert truck._astar(0, 3) is None

    def test_start_equals_goal(self, truck):
        self._wire_linear_graph(truck)
        path = truck._astar(2, 2)
        assert path is not None
        assert path == [2]

    def test_prefers_shortcut(self, truck):
        self._wire_linear_graph(truck)
        # Add a cheap direct 0→3 edge
        truck.graph[0].append(DubinsEdge(
            node_from=0, node_to=3,
            cost=5.0, path_type='RSR',
            seg_lengths=(1.0, 3.0, 1.0),
        ))
        path = truck._astar(0, 3)
        assert path == [0, 3]

    def test_all_indices_in_valid_range(self, truck):
        self._wire_linear_graph(truck)
        path = truck._astar(0, 3)
        assert path is not None
        n = len(truck.nodes)
        for idx in path:
            assert 0 <= idx < n


# ===========================================================================
# Firetruck._reconstruct_index_path tests
# ===========================================================================

class TestReconstructIndexPath:
    """
    came_from format in firetruck.py:
        came_from[child] = (parent_int, DubinsEdge)

    Tests must use that format — not plain ints.
    """

    def _cf(self, parent: int) -> tuple:
        """Make a fake came_from value with a dummy DubinsEdge."""
        dummy = DubinsEdge(
            node_from=parent, node_to=parent + 1,
            cost=1.0, path_type='LSL',
        )
        return (parent, dummy)

    def test_single_node_no_came_from(self, truck):
        path = truck._reconstruct_index_path({}, current=5)
        assert path == [5]

    def test_two_node_path(self, truck):
        cf = {1: self._cf(0)}
        path = truck._reconstruct_index_path(cf, current=1)
        assert path == [0, 1]

    def test_multi_node_path_order(self, truck):
        cf = {
            1: self._cf(0),
            2: self._cf(1),
            3: self._cf(2),
        }
        path = truck._reconstruct_index_path(cf, current=3)
        assert path == [0, 1, 2, 3]

    def test_path_starts_with_start_node(self, truck):
        cf = {1: self._cf(0), 2: self._cf(1)}
        path = truck._reconstruct_index_path(cf, current=2)
        assert path[0] == 0

    def test_path_ends_with_goal_node(self, truck):
        cf = {1: self._cf(0), 2: self._cf(1)}
        path = truck._reconstruct_index_path(cf, current=2)
        assert path[-1] == 2


# ===========================================================================
# Firetruck._reconstruct_path tests
# ===========================================================================

class TestReconstructPath:

    def _simple_roadmap(self, truck: Firetruck):
        """Two-node roadmap with a real computed edge."""
        truck.nodes = [
            (100.0, 125.0, 0.0),
            (150.0, 125.0, 0.0),
        ]
        truck.graph = {0: [], 1: []}
        edge = truck.dubins.compute_edge(0, truck.nodes[0], 1, truck.nodes[1])
        if edge:
            truck.graph[0].append(edge)

    def test_empty_index_path_returns_empty(self, truck):
        truck.nodes = [(100.0, 125.0, 0.0)]
        truck.graph = {0: []}
        assert truck._reconstruct_path([]) == []

    def test_single_node_path(self, truck):
        truck.nodes = [(100.0, 125.0, 0.0)]
        truck.graph = {0: []}
        result = truck._reconstruct_path([0])
        assert len(result) == 1
        assert result[0] == (100.0, 125.0, 0.0)

    def test_two_node_path_has_waypoints(self, truck):
        self._simple_roadmap(truck)
        result = truck._reconstruct_path([0, 1])
        assert len(result) >= 2

    def test_waypoints_are_3tuples(self, truck):
        self._simple_roadmap(truck)
        result = truck._reconstruct_path([0, 1])
        for wp in result:
            assert len(wp) == 3

    def test_first_waypoint_matches_start_node(self, truck):
        self._simple_roadmap(truck)
        result = truck._reconstruct_path([0, 1])
        assert result[0][0] == pytest.approx(100.0, abs=0.1)
        assert result[0][1] == pytest.approx(125.0, abs=0.1)

    def test_last_waypoint_near_end_node(self, truck):
        self._simple_roadmap(truck)
        result = truck._reconstruct_path([0, 1])
        assert result[-1][0] == pytest.approx(150.0, abs=2.0)
        assert result[-1][1] == pytest.approx(125.0, abs=2.0)


# ===========================================================================
# Firetruck.plan integration tests
# ===========================================================================

class TestPlan:

    def test_plan_returns_list_or_none(self, built_truck):
        result = built_truck.plan(
            goal_state=(200.0, 200.0, 0.0),
            start_state=(50.0, 50.0, 0.0),
        )
        assert result is None or isinstance(result, list)

    def test_plan_nodes_restored_after_success_or_failure(self, built_truck):
        before = len(built_truck.nodes)
        built_truck.plan(
            goal_state=(200.0, 200.0, 0.0),
            start_state=(50.0, 50.0, 0.0),
        )
        assert len(built_truck.nodes) == before

    def test_plan_nodes_restored_when_goal_unreachable(self, built_truck):
        before = len(built_truck.nodes)
        built_truck.plan(
            goal_state=(1000.0, 1000.0, 0.0),  # outside world
            start_state=(50.0, 50.0, 0.0),
        )
        assert len(built_truck.nodes) == before

    def test_plan_waypoints_are_3tuples(self, built_truck):
        result = built_truck.plan(
            goal_state=(200.0, 200.0, 0.0),
            start_state=(50.0, 50.0, 0.0),
        )
        if result is not None:
            for wp in result:
                assert len(wp) == 3

    def test_plan_uses_map_pose_as_default_start(self, built_truck):
        built_truck.map.firetruck_pose = (50.0, 50.0)
        before = len(built_truck.nodes)
        built_truck.plan(goal_state=(200.0, 200.0, 0.0))
        assert len(built_truck.nodes) == before

    def test_plan_roadmap_stable_across_multiple_calls(self, built_truck):
        """Repeated queries must not corrupt the permanent roadmap."""
        original_nodes = list(built_truck.nodes)
        original_keys  = set(built_truck.graph.keys())

        for _ in range(3):
            built_truck.plan(
                goal_state=(200.0, 200.0, 0.0),
                start_state=(50.0, 50.0, 0.0),
            )

        assert built_truck.nodes == original_nodes
        assert set(built_truck.graph.keys()) == original_keys

    def test_plan_different_goals_do_not_interfere(self, built_truck):
        before = len(built_truck.nodes)

        built_truck.plan(
            goal_state=(150.0, 150.0, 0.0),
            start_state=(50.0, 50.0, 0.0),
        )
        assert len(built_truck.nodes) == before

        built_truck.plan(
            goal_state=(200.0, 200.0, 0.0),
            start_state=(50.0, 50.0, 0.0),
        )
        assert len(built_truck.nodes) == before


# ===========================================================================
# Edge-case / regression tests
# ===========================================================================

class TestEdgeCases:

    def test_empty_obstacle_set_world_is_navigable(self, car):
        cs = ConfigurationSpace(
            car=car, world_size=WORLD_SIZE,
            obstacle_set=set(), cell_size=CELL_SIZE,
        )
        assert cs.is_free(125.0, 125.0, 0.0) is True

    def test_pose_exactly_at_world_corner_not_free(self, empty_cspace):
        assert empty_cspace.is_free(WORLD_SIZE, WORLD_SIZE, 0.0) is False

    def test_dubins_all_six_types_evaluated(self, dubins):
        """A geometry that forces CCC paths should still return a result."""
        q0 = (100.0, 100.0,   0.0)
        q1 = (100.0, 126.0, 180.0)
        assert dubins._shortest_path(q0, q1) is not None

    def test_node_list_never_grows_across_ten_plans(self, built_truck):
        before = len(built_truck.nodes)
        for i in range(10):
            built_truck.plan(
                goal_state=(200.0, 200.0, 0.0),
                start_state=(50.0, 50.0, 0.0),
            )
            assert len(built_truck.nodes) == before, \
                f"Node list grew on iteration {i}"

    def test_cleanup_with_only_start_injected(self, built_truck):
        before = len(built_truck.nodes)
        s = built_truck._inject_query_node((125.0, 125.0, 0.0), outgoing=True)
        built_truck._cleanup_query_nodes(s, None)
        assert len(built_truck.nodes) == before

    def test_cleanup_with_only_goal_injected(self, built_truck):
        before = len(built_truck.nodes)
        g = built_truck._inject_query_node((125.0, 125.0, 0.0), outgoing=False)
        built_truck._cleanup_query_nodes(None, g)
        assert len(built_truck.nodes) == before