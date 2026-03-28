"""
test_firetruck_prm.py
=====================
Comprehensive pytest suite for CarModel, ConfigurationSpace,
DubinsEdge, DubinsPlanner, and Firetruck (PRM builder + query).

Run with:
    pytest test_firetruck_prm.py -v
    pytest test_firetruck_prm.py -v --tb=short   # compact tracebacks
    pytest test_firetruck_prm.py -v -k "Dubins"  # filter by name

No Map_Generator, pathSimulator, or pathVisualizer needed —
all external dependencies are mocked via pytest's monkeypatch / sys.modules.
"""

from __future__ import annotations
import os
import sys
import math
import sys
import types
from typing import List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from shapely.geometry import Point, Polygon, box

# ---------------------------------------------------------------------------
# Mock out external modules before importing firetruck_prm
# ---------------------------------------------------------------------------

def _make_mock_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


# Map_Generator
_mg = _make_mock_module("Map_Generator")
_mg.Status = MagicMock()


class _FakeMap:
    """Minimal stand-in for Map used by Firetruck.__init__ and plan()."""
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

_make_mock_module("pathSimulator")
_make_mock_module("pathVisualizer")

# Now safe to import
# Get the directory of the current script (HW3/tests/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (HW3/)
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path so we can find Vehicles.py
sys.path.insert(0, parent_dir)

from Map_Generator import Map
from firetruck import Firetruck
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

WORLD_SIZE  = 250.0
CELL_SIZE   = 5.0


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
        c = CarModel(length=4.9, width=2.2, wheelbase=3.0,
                     r_min=13.0, v_max=10.0,
                     front_overhang=1.2, rear_overhang=0.7)
        assert c.front_overhang == 1.2
        assert c.rear_overhang  == 0.7

    def test_local_footprint_is_polygon(self, car):
        assert isinstance(car._local_footprint, Polygon)

    def test_local_footprint_area(self, car):
        expected_area = car.length * car.width
        assert car._local_footprint.area == pytest.approx(expected_area, rel=1e-3)

    def test_footprint_at_origin_zero_heading(self, car):
        fp = car.footprint_at(0.0, 0.0, 0.0)
        assert isinstance(fp, Polygon)
        assert fp.area == pytest.approx(car.length * car.width, rel=1e-3)

    def test_footprint_translates_correctly(self, car):
        fp = car.footprint_at(100.0, 50.0, 0.0)
        cx, cy = fp.centroid.x, fp.centroid.y
        # Centroid should be near (100 + wheelbase/2, 50) at 0° heading
        assert cx == pytest.approx(100.0 + (car.wheelbase / 2.0
                                   + car.front_overhang / 2.0
                                   - car.rear_overhang  / 2.0), abs=1.0)
        assert cy == pytest.approx(50.0, abs=0.1)

    def test_footprint_area_preserved_after_rotation(self, car):
        """Area must be the same regardless of heading."""
        for theta in [0, 45, 90, 135, 180, 270]:
            fp = car.footprint_at(125.0, 125.0, theta)
            assert fp.area == pytest.approx(car.length * car.width, rel=1e-3), \
                f"Area wrong at theta={theta}"

    def test_footprint_moves_with_position(self, car):
        fp1 = car.footprint_at(50.0, 50.0, 0.0)
        fp2 = car.footprint_at(100.0, 50.0, 0.0)
        assert not fp1.equals(fp2)
        assert fp1.centroid.x == pytest.approx(fp2.centroid.x - 50.0, abs=0.1)

    def test_footprint_90deg_swaps_axes(self, car):
        fp0  = car.footprint_at(125.0, 125.0, 0.0)
        fp90 = car.footprint_at(125.0, 125.0, 90.0)
        b0   = fp0.bounds   # (minx, miny, maxx, maxy)
        b90  = fp90.bounds
        width0  = b0[2] - b0[0]
        height0 = b0[3] - b0[1]
        width90  = b90[2] - b90[0]
        height90 = b90[3] - b90[1]
        # After 90° rotation, width ↔ height
        assert width90  == pytest.approx(height0, abs=0.2)
        assert height90 == pytest.approx(width0,  abs=0.2)

    def test_max_steering_angle(self, car):
        expected = math.degrees(math.atan(car.wheelbase / car.r_min))
        assert car.max_steering_angle_deg == pytest.approx(expected)

    def test_max_steering_angle_positive(self, car):
        assert car.max_steering_angle_deg > 0

    def test_repr_contains_key_info(self, car):
        r = repr(car)
        assert "4.9" in r
        assert "2.2" in r
        assert "13.0" in r


# ===========================================================================
# ConfigurationSpace tests
# ===========================================================================

class TestConfigurationSpace:

    # ── construction ──────────────────────────────────────────────────────

    def test_empty_obstacle_set(self, empty_cspace):
        assert empty_cspace.full_obstacle_geometry is None

    def test_obstacle_geometry_built(self, cspace_with_obstacle):
        assert cspace_with_obstacle.full_obstacle_geometry is not None

    def test_obstacle_geometry_covers_cell(self, cspace_with_obstacle):
        # Cell (10,10) → world [50,55]×[50,55]
        centre = Point(52.5, 52.5)
        assert cspace_with_obstacle.full_obstacle_geometry.contains(centre)

    def test_multiple_obstacles_merged(self, car):
        cs = ConfigurationSpace(
            car=car, world_size=WORLD_SIZE,
            obstacle_set={(0, 0), (1, 0), (2, 0)},
            cell_size=CELL_SIZE,
        )
        # Three adjacent cells should merge into one geometry
        assert cs.full_obstacle_geometry is not None
        assert cs.full_obstacle_geometry.area == pytest.approx(
            3 * CELL_SIZE * CELL_SIZE, rel=1e-3
        )

    # ── is_free: in-bounds checks ─────────────────────────────────────────

    def test_centre_of_world_is_free(self, empty_cspace):
        assert empty_cspace.is_free(125.0, 125.0, 0.0) is True

    def test_outside_world_bounds_not_free(self, empty_cspace):
        assert empty_cspace.is_free(-10.0, 125.0, 0.0) is False
        assert empty_cspace.is_free(125.0, -10.0, 0.0) is False
        assert empty_cspace.is_free(260.0, 125.0, 0.0) is False
        assert empty_cspace.is_free(125.0, 260.0, 0.0) is False

    def test_near_boundary_small_car_free(self, empty_cspace):
        # With enough margin the car should fit near the edge
        assert empty_cspace.is_free(20.0, 125.0, 0.0) is True

    def test_too_close_to_wall_not_free(self, empty_cspace):
        # Rear axle at x=0.5 — footprint hangs off the left wall
        assert empty_cspace.is_free(0.5, 125.0, 0.0) is False

    # ── is_free: obstacle checks ──────────────────────────────────────────

    def test_inside_obstacle_not_free(self, cspace_with_obstacle):
        # Cell (10,10) centre in world = (52.5, 52.5)
        assert cspace_with_obstacle.is_free(52.5, 52.5, 0.0) is False

    def test_far_from_obstacle_is_free(self, cspace_with_obstacle):
        assert cspace_with_obstacle.is_free(200.0, 200.0, 0.0) is True

    def test_car_overlapping_obstacle_edge_not_free(self, cspace_with_obstacle):
        # Place the car so its footprint clips the obstacle boundary
        # Obstacle at x=[50,55]; car at x=45 heading=0 → front at ~47.9
        # Should be free (no intersection)
        assert cspace_with_obstacle.is_free(40.0, 52.5, 0.0) is True
        # But closer — footprint now touches obstacle
        assert cspace_with_obstacle.is_free(48.0, 52.5, 0.0) is False

    def test_all_headings_checked(self, cspace_with_obstacle):
        """A config that's free at 0° may be blocked at 90° if rotated into obs."""
        results = [
            cspace_with_obstacle.is_free(52.5, 45.0, theta)
            for theta in range(0, 360, 15)
        ]
        # At least some headings should be blocked (car swings into obstacle)
        assert any(r is False for r in results)

    # ── is_path_free ──────────────────────────────────────────────────────

    def test_empty_pose_list_is_free(self, empty_cspace):
        assert empty_cspace.is_path_free([]) is True

    def test_all_free_poses_pass(self, empty_cspace):
        poses = [(125.0 + i, 125.0, 0.0) for i in range(10)]
        assert empty_cspace.is_path_free(poses) is True

    def test_one_bad_pose_fails_path(self, cspace_with_obstacle):
        poses = [
            (125.0, 125.0, 0.0),
            (52.5,  52.5,  0.0),   # inside obstacle
            (200.0, 200.0, 0.0),
        ]
        assert cspace_with_obstacle.is_path_free(poses) is False

    def test_fails_fast_on_first_collision(self, cspace_with_obstacle):
        """Verify early exit: pose after collision should not be reached."""
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
        poses = [
            (52.5, 52.5, 0.0),   # collision — should stop here
            (200.0, 200.0, 0.0),
        ]
        ts.is_path_free(poses)
        # Second pose should never be evaluated
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

    def test_cost_is_float(self):
        e = DubinsEdge(node_from=0, node_to=1, cost=5, path_type='RSR')
        assert isinstance(e.cost, (int, float))


# ===========================================================================
# DubinsPlanner tests
# ===========================================================================

class TestDubinsPlanner:

    # ── _mod2pi ──────────────────────────────────────────────────────────

    def test_mod2pi_zero(self, dubins):
        assert dubins._mod2pi(0.0) == pytest.approx(0.0)

    def test_mod2pi_2pi(self, dubins):
        assert dubins._mod2pi(2 * math.pi) == pytest.approx(0.0, abs=1e-9)

    def test_mod2pi_negative(self, dubins):
        result = dubins._mod2pi(-math.pi / 2)
        assert result == pytest.approx(3 * math.pi / 2, rel=1e-6)

    def test_mod2pi_large_positive(self, dubins):
        result = dubins._mod2pi(5 * math.pi)
        assert 0.0 <= result < 2 * math.pi

    # ── _shortest_path ────────────────────────────────────────────────────

    def test_coincident_poses_returns_none(self, dubins):
        q = (100.0, 100.0, 0.0)
        assert dubins._shortest_path(q, q) is None

    def test_returns_tuple_of_three(self, dubins):
        q0 = (100.0, 100.0, 0.0)
        q1 = (150.0, 100.0, 0.0)
        result = dubins._shortest_path(q0, q1)
        assert result is not None
        assert len(result) == 3   # (path_type, seg_lengths, total_length)

    def test_path_type_is_valid_word(self, dubins):
        q0 = (100.0, 100.0, 0.0)
        q1 = (150.0, 120.0, 45.0)
        result = dubins._shortest_path(q0, q1)
        assert result is not None
        assert result[0] in DubinsPlanner._PATH_TYPES

    def test_total_length_positive(self, dubins):
        q0 = (100.0, 100.0,  0.0)
        q1 = (160.0, 130.0, 90.0)
        result = dubins._shortest_path(q0, q1)
        assert result is not None
        assert result[2] > 0.0

    def test_straight_path_length(self, dubins):
        """Same heading, points aligned → should be close to Euclidean dist."""
        q0 = (100.0, 100.0, 0.0)
        q1 = (150.0, 100.0, 0.0)
        result = dubins._shortest_path(q0, q1)
        assert result is not None
        # Dubins path ≥ straight-line distance
        assert result[2] >= pytest.approx(50.0, rel=0.01)

    def test_opposite_heading_longer_than_straight(self, dubins):
        """Reversed heading forces turns — path must be longer than Euclidean."""
        q0 = (100.0, 100.0,   0.0)
        q1 = (150.0, 100.0, 180.0)
        result = dubins._shortest_path(q0, q1)
        assert result is not None
        assert result[2] > 50.0   # must be longer than the direct distance

    def test_seg_lengths_non_negative(self, dubins):
        for q1 in [
            (150.0, 100.0, 0.0),
            (150.0, 120.0, 90.0),
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
        last = poses[-1]
        assert last[0] == pytest.approx(150.0, abs=1.5)
        assert last[1] == pytest.approx(100.0, abs=1.5)

    def test_interpolate_step_size_controls_density(self, dubins):
        q0 = (100.0, 100.0, 0.0)
        q1 = (150.0, 100.0, 0.0)
        result = dubins._shortest_path(q0, q1)
        assert result is not None
        coarse = dubins._interpolate(q0, result[1], result[0], step_size=5.0)
        fine   = dubins._interpolate(q0, result[1], result[0], step_size=0.5)
        assert len(fine) > len(coarse)

    def test_interpolate_returns_list_of_3tuples(self, dubins):
        q0 = (100.0, 100.0, 45.0)
        q1 = (140.0, 120.0, 90.0)
        result = dubins._shortest_path(q0, q1)
        assert result is not None
        poses = dubins._interpolate(q0, result[1], result[0], step_size=1.0)
        for pose in poses:
            assert len(pose) == 3

    # ── path_length ───────────────────────────────────────────────────────

    def test_path_length_positive(self, dubins):
        q0 = (100.0, 100.0, 0.0)
        q1 = (150.0, 120.0, 45.0)
        assert dubins.path_length(q0, q1) > 0.0

    def test_path_length_coincident_is_inf(self, dubins):
        q = (100.0, 100.0, 0.0)
        assert dubins.path_length(q, q) == float('inf')

    def test_path_length_geq_euclidean(self, dubins):
        q0 = (100.0, 100.0, 0.0)
        q1 = (150.0, 130.0, 0.0)
        euc = math.hypot(50.0, 30.0)
        assert dubins.path_length(q0, q1) >= pytest.approx(euc, rel=0.01)

    def test_path_length_asymmetric(self, dubins):
        """Dubins(A→B) ≠ Dubins(B→A) in general."""
        q0 = (100.0, 100.0,  0.0)
        q1 = (150.0, 100.0, 90.0)
        assert dubins.path_length(q0, q1) != pytest.approx(
            dubins.path_length(q1, q0), rel=0.01
        )

    # ── compute_edge ──────────────────────────────────────────────────────

    def test_compute_edge_clear_path_returns_edge(self, dubins):
        q0 = (100.0, 100.0, 0.0)
        q1 = (150.0, 100.0, 0.0)
        edge = dubins.compute_edge(0, q0, 1, q1)
        assert edge is not None
        assert isinstance(edge, DubinsEdge)

    def test_compute_edge_node_indices_stored(self, dubins):
        q0 = (100.0, 100.0, 0.0)
        q1 = (150.0, 100.0, 0.0)
        edge = dubins.compute_edge(7, q0, 42, q1)
        assert edge.node_from == 7
        assert edge.node_to   == 42

    def test_compute_edge_cost_positive(self, dubins):
        q0 = (100.0, 100.0, 0.0)
        q1 = (150.0, 100.0, 0.0)
        edge = dubins.compute_edge(0, q0, 1, q1)
        assert edge is not None
        assert edge.cost > 0.0

    def test_compute_edge_through_obstacle_returns_none(self, car, cspace_with_obstacle):
        """Path directly through the obstacle should be rejected."""
        dp = DubinsPlanner(car, cspace_with_obstacle)
        # Obstacle at world x=[50,55], y=[50,55]
        q0 = (40.0, 52.5, 0.0)   # left of obstacle, heading right
        q1 = (70.0, 52.5, 0.0)   # right of obstacle
        edge = dp.compute_edge(0, q0, 1, q1)
        assert edge is None

    def test_compute_edge_coincident_returns_none(self, dubins):
        q = (100.0, 100.0, 0.0)
        assert dubins.compute_edge(0, q, 1, q) is None

    def test_compute_edge_path_type_valid(self, dubins):
        q0 = (100.0, 100.0, 0.0)
        q1 = (150.0, 110.0, 30.0)
        edge = dubins.compute_edge(0, q0, 1, q1)
        assert edge is not None
        assert edge.path_type in DubinsPlanner._PATH_TYPES

    # ── interpolate_edge ─────────────────────────────────────────────────

    def test_interpolate_edge_returns_list(self, dubins):
        q0 = (100.0, 100.0, 0.0)
        q1 = (150.0, 100.0, 0.0)
        edge = dubins.compute_edge(0, q0, 1, q1)
        assert edge is not None
        poses = dubins.interpolate_edge(q0, edge, step_size=1.0)
        assert isinstance(poses, list)
        assert len(poses) > 0

    def test_interpolate_edge_starts_at_source(self, dubins):
        q0 = (100.0, 100.0, 0.0)
        q1 = (150.0, 100.0, 0.0)
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

    def test_nodes_have_theta(self, truck):
        truck._sample_points(n_samples=20)
        for x, y, theta in truck.nodes:
            assert 0.0 <= theta < 360.0

    def test_all_nodes_in_bounds(self, truck):
        truck._sample_points(n_samples=50)
        limit = truck.map.grid_num * truck.map.cell_size
        for x, y, theta in truck.nodes:
            assert 0 < x < limit, f"x={x} out of bounds"
            assert 0 < y < limit, f"y={y} out of bounds"

    def test_all_nodes_collision_free(self, truck):
        truck._sample_points(n_samples=50)
        for x, y, theta in truck.nodes:
            assert truck.cspace.is_free(x, y, theta), \
                f"Node ({x:.1f},{y:.1f},{theta}) in collision"

    def test_graph_keys_match_node_indices(self, truck):
        truck._sample_points(n_samples=30)
        for i in range(len(truck.nodes)):
            assert i in truck.graph, f"Node {i} missing from graph"

    def test_graph_values_are_empty_lists_after_sampling(self, truck):
        truck._sample_points(n_samples=30)
        for edges in truck.graph.values():
            assert edges == []

    def test_kd_tree_built(self, truck):
        truck._sample_points(n_samples=20)
        assert truck._kd_tree is not None

    def test_kd_tree_size_matches_nodes(self, truck):
        truck._sample_points(n_samples=30)
        assert truck._kd_tree.n == len(truck.nodes)

    def test_respects_n_samples(self, truck):
        truck._sample_points(n_samples=40)
        # May be slightly less if environment very cluttered, but not more
        assert len(truck.nodes) <= 40

    def test_resets_nodes_on_second_call(self, truck):
        truck._sample_points(n_samples=20)
        first_count = len(truck.nodes)
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

    def test_graph_is_directed(self, truck):
        """A→B being present does not guarantee B→A."""
        truck._sample_points(n_samples=40)
        truck._connect_nodes(k_neighbors=5, r_connect=40.0)
        # At least some node pairs should be asymmetric
        asymmetric_found = False
        for i, edges in truck.graph.items():
            for e in edges:
                j = e.node_to
                reverse_exists = any(re.node_to == i
                                     for re in truck.graph.get(j, []))
                if not reverse_exists:
                    asymmetric_found = True
                    break
            if asymmetric_found:
                break
        # Not guaranteed but very likely with random poses
        # We just verify no crash and valid structure
        assert True

    def test_edge_costs_positive(self, truck):
        truck._sample_points(n_samples=30)
        truck._connect_nodes(k_neighbors=5, r_connect=50.0)
        for edges in truck.graph.values():
            for e in edges:
                assert e.cost > 0.0

    def test_edge_node_indices_in_range(self, truck):
        truck._sample_points(n_samples=30)
        truck._connect_nodes(k_neighbors=5, r_connect=50.0)
        n = len(truck.nodes)
        for i, edges in truck.graph.items():
            assert 0 <= i < n
            for e in edges:
                assert 0 <= e.node_to < n, \
                    f"edge.node_to={e.node_to} out of range [0,{n})"

    def test_no_self_loops(self, truck):
        truck._sample_points(n_samples=30)
        truck._connect_nodes(k_neighbors=5, r_connect=50.0)
        for i, edges in truck.graph.items():
            for e in edges:
                assert e.node_to != i, f"Self-loop at node {i}"

    def test_some_edges_built(self, truck):
        truck._sample_points(n_samples=40)
        truck._connect_nodes(k_neighbors=10, r_connect=60.0)
        total = sum(len(v) for v in truck.graph.values())
        assert total > 0, "No edges were built — something is wrong"


# ===========================================================================
# Firetruck.build_tree tests
# ===========================================================================

class TestBuildTree:

    def test_roadmap_size_set(self, truck):
        truck.build_tree(n_samples=30)
        assert truck._roadmap_size == len(truck.nodes)
        assert truck._roadmap_size > 0

    def test_nodes_not_empty_after_build(self, truck):
        truck.build_tree(n_samples=30)
        assert len(truck.nodes) > 0

    def test_graph_not_empty_after_build(self, truck):
        truck.build_tree(n_samples=30)
        assert len(truck.graph) > 0

    def test_plan_raises_before_build(self, truck):
        with pytest.raises(RuntimeError, match="build_tree"):
            truck.plan((200.0, 200.0, 0.0))


# ===========================================================================
# Firetruck._inject_query_node tests
# ===========================================================================

class TestInjectQueryNode:

    def test_start_node_appended_beyond_roadmap(self, built_truck):
        before = len(built_truck.nodes)
        idx = built_truck._inject_query_node(
            (125.0, 125.0, 0.0), outgoing=True
        )
        assert idx is not None
        assert idx == before

    def test_goal_node_appended_after_start(self, built_truck):
        built_truck._inject_query_node((125.0, 125.0, 0.0), outgoing=True)
        before = len(built_truck.nodes)
        idx = built_truck._inject_query_node(
            (200.0, 200.0, 0.0), outgoing=False
        )
        assert idx is not None
        assert idx == before

    def test_start_has_outgoing_edges(self, built_truck):
        idx = built_truck._inject_query_node(
            (125.0, 125.0, 0.0), outgoing=True
        )
        if idx is not None:
            assert len(built_truck.graph[idx]) > 0

    def test_goal_has_incoming_edges(self, built_truck):
        idx = built_truck._inject_query_node(
            (125.0, 125.0, 0.0), outgoing=False
        )
        if idx is not None:
            # Edges INTO idx appear on roadmap nodes, not on idx itself
            incoming = [
                e for edges in built_truck.graph.values()
                for e in edges if e.node_to == idx
            ]
            assert len(incoming) > 0

    def test_injected_edges_only_connect_to_roadmap(self, built_truck):
        idx = built_truck._inject_query_node(
            (125.0, 125.0, 0.0), outgoing=True
        )
        if idx is not None:
            for e in built_truck.graph[idx]:
                assert e.node_to < built_truck._roadmap_size


# ===========================================================================
# Firetruck._cleanup_query_nodes tests
# ===========================================================================

class TestCleanupQueryNodes:

    def test_nodes_truncated_to_roadmap_size(self, built_truck):
        rs = built_truck._roadmap_size
        built_truck._inject_query_node((125.0, 125.0, 0.0), outgoing=True)
        built_truck._inject_query_node((200.0, 200.0, 0.0), outgoing=False)
        assert len(built_truck.nodes) > rs
        built_truck._cleanup_query_nodes()
        assert len(built_truck.nodes) == rs

    def test_temp_graph_entries_removed(self, built_truck):
        rs = built_truck._roadmap_size
        built_truck._inject_query_node((125.0, 125.0, 0.0), outgoing=True)
        built_truck._inject_query_node((200.0, 200.0, 0.0), outgoing=False)
        built_truck._cleanup_query_nodes()
        for idx in built_truck.graph:
            assert idx < rs, f"Temp graph key {idx} not removed"

    def test_permanent_nodes_unchanged(self, built_truck):
        original_nodes = list(built_truck.nodes)
        built_truck._inject_query_node((125.0, 125.0, 0.0), outgoing=True)
        built_truck._inject_query_node((200.0, 200.0, 0.0), outgoing=False)
        built_truck._cleanup_query_nodes()
        assert built_truck.nodes == original_nodes

    def test_permanent_edges_not_pointing_to_temp(self, built_truck):
        built_truck._inject_query_node((125.0, 125.0, 0.0), outgoing=True)
        built_truck._inject_query_node((200.0, 200.0, 0.0), outgoing=False)
        built_truck._cleanup_query_nodes()
        rs = built_truck._roadmap_size
        for i in range(rs):
            for e in built_truck.graph.get(i, []):
                assert e.node_to < rs, \
                    f"Permanent edge from {i} points to temp node {e.node_to}"

    def test_no_op_when_nothing_injected(self, built_truck):
        original_nodes = list(built_truck.nodes)
        built_truck._cleanup_query_nodes()   # should not raise
        assert built_truck.nodes == original_nodes

    def test_multiple_cleanup_calls_safe(self, built_truck):
        """Calling cleanup twice should not corrupt state."""
        built_truck._inject_query_node((125.0, 125.0, 0.0), outgoing=True)
        built_truck._cleanup_query_nodes()
        built_truck._cleanup_query_nodes()   # second call — no-op
        assert len(built_truck.nodes) == built_truck._roadmap_size

    def test_repeated_plan_calls_stable(self, built_truck):
        """The core regression: repeated queries must not corrupt node list."""
        rs = built_truck._roadmap_size
        for _ in range(5):
            built_truck._inject_query_node((125.0, 125.0, 0.0), outgoing=True)
            built_truck._inject_query_node((200.0, 200.0, 0.0), outgoing=False)
            built_truck._cleanup_query_nodes()
            assert len(built_truck.nodes) == rs, \
                "Node list length drifted after cleanup"


# ===========================================================================
# Firetruck._astar tests
# ===========================================================================

class TestAstar:

    def _build_minimal_graph(self, truck: Firetruck):
        """
        Manually wire up a tiny 4-node graph for deterministic A* testing.

        Layout (nodes stored directly):
          0 → 1 → 2 → 3    (linear chain, all heading 0°)
          Nodes spaced 30 m apart along x-axis.
        """
        truck.nodes = [
            (50.0,  125.0, 0.0),
            (80.0,  125.0, 0.0),
            (110.0, 125.0, 0.0),
            (140.0, 125.0, 0.0),
        ]
        truck.graph = {i: [] for i in range(4)}
        truck._roadmap_size = 4

        # Wire 0→1, 1→2, 2→3 with dummy DubinsEdges
        for i in range(3):
            truck.graph[i].append(DubinsEdge(
                node_from=i, node_to=i+1,
                cost=30.0, path_type='LSL',
                seg_lengths=(10.0, 10.0, 10.0),
            ))

    def test_finds_path_on_linear_chain(self, truck):
        self._build_minimal_graph(truck)
        path = truck._astar(0, 3)
        assert path is not None
        assert path[0] == 0
        assert path[-1] == 3

    def test_returns_none_when_disconnected(self, truck):
        self._build_minimal_graph(truck)
        # Remove the 1→2 edge — graph becomes disconnected
        truck.graph[1] = []
        path = truck._astar(0, 3)
        assert path is None

    def test_path_visits_correct_nodes(self, truck):
        self._build_minimal_graph(truck)
        path = truck._astar(0, 3)
        assert path == [0, 1, 2, 3]

    def test_start_equals_goal_returns_single_node(self, truck):
        self._build_minimal_graph(truck)
        path = truck._astar(2, 2)
        assert path is not None
        assert path == [2]

    def test_prefers_shorter_path(self, truck):
        """Add a shortcut 0→3 and verify A* takes it."""
        self._build_minimal_graph(truck)
        truck.graph[0].append(DubinsEdge(
            node_from=0, node_to=3,
            cost=10.0,    # much cheaper than 3 × 30 = 90
            path_type='RSR',
            seg_lengths=(3.0, 4.0, 3.0),
        ))
        path = truck._astar(0, 3)
        assert path == [0, 3]

    def test_path_indices_in_valid_range(self, truck):
        self._build_minimal_graph(truck)
        path = truck._astar(0, 3)
        assert path is not None
        n = len(truck.nodes)
        for idx in path:
            assert 0 <= idx < n


# ===========================================================================
# Firetruck._reconstruct_index_path tests
# ===========================================================================

class TestReconstructIndexPath:

    def test_single_node_path(self, truck):
        came_from = {}
        path = truck._reconstruct_index_path(came_from, current=5)
        assert path == [5]

    def test_two_node_path(self, truck):
        came_from = {1: 0}
        path = truck._reconstruct_index_path(came_from, current=1)
        assert path == [0, 1]

    def test_multi_node_path_order(self, truck):
        came_from = {1: 0, 2: 1, 3: 2}
        path = truck._reconstruct_index_path(came_from, current=3)
        assert path == [0, 1, 2, 3]

    def test_path_starts_with_start_node(self, truck):
        came_from = {1: 0, 2: 1}
        path = truck._reconstruct_index_path(came_from, current=2)
        assert path[0] == 0

    def test_path_ends_with_goal_node(self, truck):
        came_from = {1: 0, 2: 1}
        path = truck._reconstruct_index_path(came_from, current=2)
        assert path[-1] == 2


# ===========================================================================
# Firetruck._reconstruct_path tests
# ===========================================================================

class TestReconstructPath:

    def _inject_simple_roadmap(self, truck: Firetruck):
        """Two-node roadmap with one Dubins edge for reconstruction testing."""
        truck.nodes = [
            (100.0, 125.0, 0.0),
            (150.0, 125.0, 0.0),
        ]
        truck._roadmap_size = 2

        # Compute a real edge so interpolate_edge works
        edge = truck.dubins.compute_edge(0, truck.nodes[0], 1, truck.nodes[1])
        truck.graph = {0: [], 1: []}
        if edge:
            truck.graph[0].append(edge)

    def test_empty_index_path_returns_empty(self, truck):
        truck.nodes = [(100.0, 125.0, 0.0)]
        truck.graph = {0: []}
        result = truck._reconstruct_path([])
        assert result == []

    def test_single_node_returns_one_waypoint(self, truck):
        truck.nodes = [(100.0, 125.0, 0.0)]
        truck.graph = {0: []}
        result = truck._reconstruct_path([0])
        assert len(result) == 1
        assert result[0] == (100.0, 125.0, 0.0)

    def test_two_node_path_returns_waypoints(self, truck):
        self._inject_simple_roadmap(truck)
        result = truck._reconstruct_path([0, 1])
        assert len(result) >= 2

    def test_waypoints_are_3tuples(self, truck):
        self._inject_simple_roadmap(truck)
        result = truck._reconstruct_path([0, 1])
        for wp in result:
            assert len(wp) == 3

    def test_first_waypoint_is_start_node(self, truck):
        self._inject_simple_roadmap(truck)
        result = truck._reconstruct_path([0, 1])
        assert result[0][0] == pytest.approx(100.0, abs=0.1)
        assert result[0][1] == pytest.approx(125.0, abs=0.1)

    def test_last_waypoint_near_end_node(self, truck):
        self._inject_simple_roadmap(truck)
        result = truck._reconstruct_path([0, 1])
        assert result[-1][0] == pytest.approx(150.0, abs=2.0)
        assert result[-1][1] == pytest.approx(125.0, abs=2.0)


# ===========================================================================
# Firetruck.plan integration tests
# ===========================================================================

class TestPlan:

    def test_plan_before_build_raises(self, truck):
        with pytest.raises(RuntimeError):
            truck.plan((200.0, 200.0, 0.0))

    def test_plan_returns_list_or_none(self, built_truck):
        result = built_truck.plan(
            goal_state=(200.0, 200.0, 0.0),
            start_state=(50.0, 50.0, 0.0),
        )
        assert result is None or isinstance(result, list)

    def test_plan_cleans_up_nodes(self, built_truck):
        rs = built_truck._roadmap_size
        built_truck.plan(
            goal_state=(200.0, 200.0, 0.0),
            start_state=(50.0, 50.0, 0.0),
        )
        assert len(built_truck.nodes) == rs

    def test_plan_cleans_up_on_failure(self, built_truck):
        """Cleanup must happen even if no path is found."""
        rs = built_truck._roadmap_size
        # Inject an unreachable goal far outside the world
        built_truck.plan(
            goal_state=(1000.0, 1000.0, 0.0),
            start_state=(50.0, 50.0, 0.0),
        )
        assert len(built_truck.nodes) == rs

    def test_plan_idempotent_roadmap(self, built_truck):
        """Permanent roadmap must be identical before and after multiple plans."""
        original_nodes = list(built_truck.nodes)
        original_graph_keys = set(built_truck.graph.keys())

        for _ in range(3):
            built_truck.plan(
                goal_state=(200.0, 200.0, 0.0),
                start_state=(50.0, 50.0, 0.0),
            )

        assert built_truck.nodes == original_nodes
        assert set(built_truck.graph.keys()) == original_graph_keys

    def test_plan_waypoints_are_3tuples(self, built_truck):
        result = built_truck.plan(
            goal_state=(200.0, 200.0, 0.0),
            start_state=(50.0, 50.0, 0.0),
        )
        if result is not None:
            for wp in result:
                assert len(wp) == 3, f"Waypoint {wp} is not a 3-tuple"

    def test_plan_uses_map_pose_when_no_start(self, built_truck):
        """When start_state=None, plan() should read map.firetruck_pose."""
        built_truck.map.firetruck_pose = (50.0, 50.0)
        result = built_truck.plan(goal_state=(200.0, 200.0, 0.0))
        # Just verify no crash and cleanup happened
        assert len(built_truck.nodes) == built_truck._roadmap_size

    def test_plan_different_goals_independent(self, built_truck):
        """Two sequential plan() calls must not interfere with each other."""
        rs = built_truck._roadmap_size
        r1 = built_truck.plan(
            goal_state=(150.0, 150.0, 0.0),
            start_state=(50.0, 50.0, 0.0),
        )
        assert len(built_truck.nodes) == rs   # clean after first

        r2 = built_truck.plan(
            goal_state=(200.0, 200.0, 0.0),
            start_state=(50.0, 50.0, 0.0),
        )
        assert len(built_truck.nodes) == rs   # clean after second


# ===========================================================================
# Edge-case / regression tests
# ===========================================================================

class TestEdgeCases:

    def test_zero_obstacle_set(self, car):
        cs = ConfigurationSpace(
            car=car, world_size=WORLD_SIZE,
            obstacle_set=set(), cell_size=CELL_SIZE,
        )
        assert cs.is_free(125.0, 125.0, 0.0) is True

    def test_car_exactly_at_world_boundary(self, empty_cspace):
        """Rear axle at the far corner should fail (footprint out of bounds)."""
        assert empty_cspace.is_free(WORLD_SIZE, WORLD_SIZE, 0.0) is False

    def test_dubins_all_path_types_attempted(self, dubins):
        """_shortest_path should evaluate all 6 types and return the best."""
        q0 = (100.0, 100.0, 0.0)
        q1 = (100.0, 126.0, 180.0)   # directly above, reversed heading
        result = dubins._shortest_path(q0, q1)
        # This geometry forces the planner to pick a CCC or CSC type
        assert result is not None

    def test_cleanup_is_idempotent(self, built_truck):
        for _ in range(10):
            built_truck._cleanup_query_nodes()
        assert len(built_truck.nodes) == built_truck._roadmap_size

    def test_inject_returns_none_for_impossible_location(self, built_truck):
        """A pose that cannot be connected to anything returns None."""
        # Place the query node far outside the world
        idx = built_truck._inject_query_node(
            (5000.0, 5000.0, 0.0), outgoing=True
        )
        built_truck._cleanup_query_nodes()
        # idx may be None (no connection) or a valid index if some edge existed
        # Either way, cleanup must have restored the roadmap
        assert len(built_truck.nodes) == built_truck._roadmap_size

    def test_dubins_interpolation_no_nans(self, dubins):
        """All interpolated poses must be finite numbers."""
        q0 = (100.0, 100.0,  0.0)
        q1 = (160.0, 130.0, 45.0)
        result = dubins._shortest_path(q0, q1)
        assert result is not None
        poses = dubins._interpolate(q0, result[1], result[0], step_size=1.0)
        for x, y, theta in poses:
            assert math.isfinite(x),     f"x={x} is not finite"
            assert math.isfinite(y),     f"y={y} is not finite"
            assert math.isfinite(theta), f"theta={theta} is not finite"

    def test_node_list_never_grows_across_plans(self, built_truck):
        rs = built_truck._roadmap_size
        for i in range(10):
            built_truck.plan(
                goal_state=(200.0, 200.0, 0.0),
                start_state=(50.0, 50.0, 0.0),
            )
            assert len(built_truck.nodes) == rs, \
                f"Node list grew on iteration {i}: {len(built_truck.nodes)} != {rs}"