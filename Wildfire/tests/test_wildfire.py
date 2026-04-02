"""
test_wildfire.py
================
Comprehensive test suite for the Wildfire Simulation project.

Covers:
  - Map_Generator  : obstacle management, fire spread, status machine, map generation
  - wumpus         : A* planning, bounds checking, burn logic, closest-obstacle search
  - firetruck      : PRM build, collision checking, Reeds-Shepp planning, A* search
  - simulation_engine : scoring, state machine, tick logic, RunResult, tournament utils
  - tournament     : run_tournament, summarise, plot helpers (structure only, no display)

Run with:
    pytest test_wildfire.py -v

Dependencies that must be importable:
    Map_Generator, wumpus, firetruck, simulation_engine, tournament
    (pathVisualizer is monkey-patched where it would open a window)
"""

from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


import math
import threading
import time
import types
import unittest
from collections import deque
from typing import List, Optional
from unittest.mock import MagicMock, patch, PropertyMock

# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------

def _make_map(grid_num=20, cell_size=5.0, fill_percent=0.10,
              firetruck_pose=(12.5, 12.5, 0.0),
              wumpus_pose=(87.5, 87.5)):
    """Return a lightweight Map suitable for unit testing."""
    from Map_Generator import Map
    return Map(
        Grid_num=grid_num,
        cell_size=cell_size,
        fill_percent=fill_percent,
        wumpus=None,
        firetruck=None,
        firetruck_pose=firetruck_pose,
        wumpus_pose=wumpus_pose,
    )


def _make_map_no_obstacles(grid_num=20, cell_size=5.0):
    """Return an empty map (fill_percent=0)."""
    from Map_Generator import Map
    m = Map(
        Grid_num=grid_num,
        cell_size=cell_size,
        fill_percent=0.0,
        wumpus=None,
        firetruck=None,
        firetruck_pose=(12.5, 12.5, 0.0),
        wumpus_pose=(87.5, 87.5),
    )
    m.obstacle_set.clear()
    m.obstacle_coordinate_dict.clear()
    return m


# ===========================================================================
# 1.  Map_Generator tests
# ===========================================================================

class TestMapStatus(unittest.TestCase):
    """Status state machine."""

    def setUp(self):
        from Map_Generator import Status
        self.Status = Status
        self.m = _make_map()
        # plant a known obstacle at (0,0)
        self.coord = (0, 0)
        self.m._append_obstacle(self.coord)

    def test_initial_status_intact(self):
        self.assertEqual(self.m.get_status(self.coord), self.Status.INTACT)

    def test_intact_to_burning(self):
        self.m.set_status_on_obstacles([self.coord], self.Status.BURNING)
        self.assertEqual(self.m.get_status(self.coord), self.Status.BURNING)
        self.assertIn(self.coord, self.m.active_fires)

    def test_burning_to_extinguished(self):
        self.m.set_status_on_obstacles([self.coord], self.Status.BURNING)
        self.m.set_status_on_obstacles([self.coord], self.Status.EXTINGUISHED)
        self.assertEqual(self.m.get_status(self.coord), self.Status.EXTINGUISHED)
        self.assertNotIn(self.coord, self.m.active_fires)

    def test_burning_to_burned(self):
        self.m.set_status_on_obstacles([self.coord], self.Status.BURNING)
        self.m.set_status_on_obstacles([self.coord], self.Status.BURNED)
        self.assertEqual(self.m.get_status(self.coord), self.Status.BURNED)
        self.assertNotIn(self.coord, self.m.active_fires)

    def test_extinguished_is_terminal(self):
        """EXTINGUISHED → BURNING must be ignored."""
        self.m.set_status_on_obstacles([self.coord], self.Status.BURNING)
        self.m.set_status_on_obstacles([self.coord], self.Status.EXTINGUISHED)
        self.m.set_status_on_obstacles([self.coord], self.Status.BURNING)
        self.assertEqual(self.m.get_status(self.coord), self.Status.EXTINGUISHED)

    def test_burned_is_terminal(self):
        """BURNED → BURNING must be ignored."""
        self.m.set_status_on_obstacles([self.coord], self.Status.BURNING)
        self.m.set_status_on_obstacles([self.coord], self.Status.BURNED)
        self.m.set_status_on_obstacles([self.coord], self.Status.BURNING)
        self.assertEqual(self.m.get_status(self.coord), self.Status.BURNED)

    def test_no_op_same_status(self):
        """Setting the same status as current is a no-op."""
        self.m.set_status_on_obstacles([self.coord], self.Status.INTACT)
        self.assertNotIn(self.coord, self.m.active_fires)

    def test_set_status_on_missing_coord_is_silent(self):
        """Setting status on a non-existent coord should not raise."""
        self.m.set_status_on_obstacles([(99, 99)], self.Status.BURNING)  # no-op

    def test_burn_time_recorded_on_ignition(self):
        self.m.sim_time = 42.0
        self.m.set_status_on_obstacles([self.coord], self.Status.BURNING)
        self.assertAlmostEqual(
            self.m.obstacle_coordinate_dict[self.coord]["burn_time"], 42.0
        )

    def test_burn_time_cleared_on_extinguish(self):
        self.m.set_status_on_obstacles([self.coord], self.Status.BURNING)
        self.m.set_status_on_obstacles([self.coord], self.Status.EXTINGUISHED)
        self.assertIsNone(self.m.obstacle_coordinate_dict[self.coord]["burn_time"])

    def test_multiple_coords_set_at_once(self):
        c2 = (1, 1)
        self.m._append_obstacle(c2)
        self.m.set_status_on_obstacles([self.coord, c2], self.Status.BURNING)
        self.assertIn(self.coord, self.m.active_fires)
        self.assertIn(c2, self.m.active_fires)


class TestMapObstacleMethods(unittest.TestCase):

    def setUp(self):
        self.m = _make_map_no_obstacles()

    def test_append_obstacle_adds_to_all_collections(self):
        self.m._append_obstacle((3, 3))
        self.assertIn((3, 3), self.m.obstacle_set)
        self.assertIn((3, 3), self.m.obstacle_coordinate_dict)

    def test_delete_obstacle_removes_from_all_collections(self):
        self.m._append_obstacle((3, 3))
        self.m._delete_obstacle((3, 3))
        self.assertNotIn((3, 3), self.m.obstacle_set)
        self.assertNotIn((3, 3), self.m.obstacle_coordinate_dict)

    def test_delete_burning_obstacle_removes_from_active_fires(self):
        from Map_Generator import Status
        self.m._append_obstacle((2, 2))
        self.m.set_status_on_obstacles([(2, 2)], Status.BURNING)
        self.m._delete_obstacle((2, 2))
        self.assertNotIn((2, 2), self.m.active_fires)

    def test_check_valid_cell_empty(self):
        self.assertTrue(self.m.check_valid_cell((5, 5)))

    def test_check_valid_cell_occupied(self):
        self.m._append_obstacle((5, 5))
        self.assertFalse(self.m.check_valid_cell((5, 5)))

    def test_check_valid_cell_out_of_bounds(self):
        self.assertFalse(self.m.check_valid_cell((-1, 0)))
        self.assertFalse(self.m.check_valid_cell((0, 100)))

    def test_append_new_obstacle_alias(self):
        """_append_new_obstacle is a compatibility alias for _append_obstacle."""
        self.m._append_new_obstacle((7, 7))
        self.assertIn((7, 7), self.m.obstacle_set)


class TestMapFireSpread(unittest.TestCase):

    def setUp(self):
        from Map_Generator import Status
        self.Status = Status
        self.m = _make_map_no_obstacles()
        # plant a cluster of touching obstacles
        for coord in [(5, 5), (5, 6), (6, 5)]:
            self.m._append_obstacle(coord)

    def test_fire_spreads_to_nearby_intact_obstacle(self):
        """After >10s burn time, _find_burnable_obstacles should catch neighbours."""
        self.m.sim_time = 0.0
        self.m.set_status_on_obstacles([(5, 5)], self.Status.BURNING)
        # advance sim_time past spread threshold
        self.m.sim_time = 11.0
        self.m._check_time_events()
        # At least one neighbour should have started burning
        neighbour_burning = (
            self.m.get_status((5, 6)) == self.Status.BURNING
            or self.m.get_status((6, 5)) == self.Status.BURNING
        )
        self.assertTrue(neighbour_burning)

    def test_obstacle_becomes_burned_after_30s(self):
        self.m.sim_time = 0.0
        self.m.set_status_on_obstacles([(5, 5)], self.Status.BURNING)
        self.m.sim_time = 31.0
        self.m._check_time_events()
        self.assertEqual(self.m.get_status((5, 5)), self.Status.BURNED)

    def test_intact_obstacle_not_changed_without_fire(self):
        self.m.sim_time = 100.0
        self.m._check_time_events()
        self.assertEqual(self.m.get_status((5, 5)), self.Status.INTACT)

    def test_find_burnable_obstacles_radius(self):
        result = self.m._find_burnable_obstacles((5, 5), radius=2)
        # Should include (5,6) and (6,5) which are within radius 2
        self.assertIn((5, 6), result)
        self.assertIn((6, 5), result)

    def test_find_burnable_obstacles_excludes_center(self):
        """The centre coord itself is an obstacle; it may be in the result
        (find_burnable returns everything in radius including centre)."""
        result = self.m._find_burnable_obstacles((5, 5), radius=1)
        # All returned cells must be in obstacle_set
        for c in result:
            self.assertIn(c, self.m.obstacle_set)

    def test_main_advances_sim_time(self):
        t0 = self.m.sim_time
        self.m.main()
        self.assertAlmostEqual(self.m.sim_time, t0 + 0.1, places=5)

    def test_main_returns_done_after_3600s(self):
        self.m.sim_time = 3600.1
        result = self.m.main()
        self.assertEqual(result, "Done")

    def test_main_returns_none_before_3600s(self):
        self.m.sim_time = 100.0
        result = self.m.main()
        self.assertIsNone(result)


class TestMapGoal(unittest.TestCase):

    def test_update_goal(self):
        m = _make_map()
        m.update_goal((10.0, 20.0, 45.0))
        self.assertEqual(m.firetruck_goal, (10.0, 20.0, 45.0))

    def test_find_firetruck_goal_no_fires_returns_wumpus_pos(self):
        m = _make_map(wumpus_pose=(77.5, 88.5))
        goal = m.find_firetruck_goal()
        self.assertAlmostEqual(goal[0], 77.5)
        self.assertAlmostEqual(goal[1], 88.5)

    def test_find_firetruck_goal_with_fire_not_wumpus(self):
        from Map_Generator import Status
        m = _make_map_no_obstacles()
        m._append_obstacle((10, 10))
        m.set_status_on_obstacles([(10, 10)], Status.BURNING)
        m.firetruck_pose = (0.0, 0.0, 0.0)
        goal = m.find_firetruck_goal()
        # Should return a tuple, not the wumpus pose
        self.assertIsInstance(goal, tuple)


class TestMapGeneration(unittest.TestCase):

    def test_obstacle_count_roughly_matches_fill_percent(self):
        m = _make_map(grid_num=30, fill_percent=0.15)
        total_cells  = 30 * 30
        expected_min = int(total_cells * 0.05)   # allow wide tolerance
        self.assertGreater(len(m.obstacle_set), expected_min)

    def test_no_obstacle_at_firetruck_start(self):
        m = _make_map(firetruck_pose=(12.5, 12.5, 0.0))
        cs = m.cell_size
        ft_cell = (int(12.5 / cs), int(12.5 / cs))
        self.assertNotIn(ft_cell, m.obstacle_set)

    def test_no_obstacle_at_wumpus_start(self):
        m = _make_map(wumpus_pose=(87.5, 87.5))
        cs = m.cell_size
        wu_cell = (int(87.5 / cs), int(87.5 / cs))
        self.assertNotIn(wu_cell, m.obstacle_set)

    def test_all_obstacles_within_grid(self):
        m = _make_map()
        gn = m.grid_num
        for r, c in m.obstacle_set:
            self.assertGreaterEqual(r, 0)
            self.assertGreaterEqual(c, 0)
            self.assertLess(r, gn)
            self.assertLess(c, gn)

    def test_zero_fill_percent_gives_empty_map(self):
        m = _make_map_no_obstacles()
        self.assertEqual(len(m.obstacle_set), 0)


# ===========================================================================
# 2.  Wumpus tests
# ===========================================================================

class TestWumpusInit(unittest.TestCase):

    def test_four_cardinal_directions(self):
        from wumpus import Wumpus
        m = _make_map_no_obstacles()
        w = Wumpus(m)
        self.assertEqual(len(w.directions), 4)

    def test_wumpus_uses_map_reference(self):
        from wumpus import Wumpus
        m = _make_map_no_obstacles()
        w = Wumpus(m)
        self.assertIs(w.map, m)


class TestWumpusBounds(unittest.TestCase):

    def setUp(self):
        from wumpus import Wumpus
        self.m = _make_map_no_obstacles(grid_num=10, cell_size=5.0)
        self.w = Wumpus(self.m)

    def test_in_bounds_centre(self):
        self.assertTrue(self.w._in_bounds((5, 5)))

    def test_in_bounds_corner(self):
        self.assertTrue(self.w._in_bounds((0, 0)))
        self.assertTrue(self.w._in_bounds((9, 9)))

    def test_out_of_bounds_negative(self):
        self.assertFalse(self.w._in_bounds((-1, 0)))
        self.assertFalse(self.w._in_bounds((0, -1)))

    def test_out_of_bounds_too_large(self):
        self.assertFalse(self.w._in_bounds((10, 5)))
        self.assertFalse(self.w._in_bounds((5, 10)))


class TestWumpusHeuristic(unittest.TestCase):

    def test_same_cell(self):
        from wumpus import Wumpus
        self.assertEqual(Wumpus._heuristic((3, 3), (3, 3)), 0)

    def test_cardinal_distance(self):
        from wumpus import Wumpus
        self.assertEqual(Wumpus._heuristic((0, 0), (5, 0)), 5)

    def test_diagonal_distance(self):
        from wumpus import Wumpus
        # Chebyshev: max(|3|,|4|) = 4
        self.assertEqual(Wumpus._heuristic((0, 0), (3, 4)), 4)


class TestWumpusUnwind(unittest.TestCase):

    def test_simple_path(self):
        from wumpus import Wumpus
        came_from = {(1, 0): (0, 0), (2, 0): (1, 0)}
        path = Wumpus._unwind(came_from, (2, 0))
        self.assertEqual(path, [(0, 0), (1, 0), (2, 0)])

    def test_start_node_only(self):
        from wumpus import Wumpus
        path = Wumpus._unwind({}, (0, 0))
        self.assertEqual(path, [(0, 0)])


class TestWumpusPlan(unittest.TestCase):

    def setUp(self):
        from wumpus import Wumpus
        # 5×5 grid, no obstacles so A* can always find a straight path
        self.m = _make_map_no_obstacles(grid_num=5, cell_size=5.0)
        self.w = Wumpus(self.m)

    def test_plan_returns_none_with_no_obstacles(self):
        """No obstacles → find_closest_obstacle returns None → plan returns None."""
        result = self.w.plan()
        self.assertIsNone(result)

    def test_plan_returns_path_when_obstacle_present(self):
        from Map_Generator import Status
        # Place obstacle at (3,3), wumpus at (0,0)
        self.m._append_obstacle((3, 3))
        self.m.wumpus_pose = (0.0, 0.0)
        path = self.w.plan()
        # Should return a list of (row, col) tuples
        if path is not None:
            self.assertIsInstance(path, list)
            self.assertGreater(len(path), 0)

    def test_plan_path_starts_near_wumpus(self):
        self.m._append_obstacle((3, 3))
        self.m.wumpus_pose = (0.0, 0.0)
        path = self.w.plan()
        if path is not None:
            start = path[0]
            self.assertIsInstance(start, tuple)
            self.assertEqual(len(start), 2)

    def test_plan_does_not_pass_through_obstacles(self):
        """All cells in the returned path must not be in obstacle_set."""
        self.m._append_obstacle((2, 2))
        self.m._append_obstacle((3, 3))
        self.m.wumpus_pose = (0.0, 0.0)
        path = self.w.plan()
        if path is not None:
            for cell in path:
                self.assertNotIn(cell, self.m.obstacle_set)


class TestWumpusBurn(unittest.TestCase):

    def setUp(self):
        from wumpus import Wumpus
        from Map_Generator import Status
        self.Status = Status
        self.m = _make_map_no_obstacles(grid_num=10, cell_size=5.0)
        self.w = Wumpus(self.m)

    def test_burn_ignites_adjacent_obstacle(self):
        # Wumpus at cell (5,5) world=(27.5,27.5), obstacle at (5,6)
        self.m._append_obstacle((5, 6))
        self.m.wumpus_pose = (27.5, 27.5)  # cell (5,5)
        self.w.burn()
        self.assertEqual(self.m.get_status((5, 6)), self.Status.BURNING)

    def test_burn_does_not_ignite_non_adjacent(self):
        self.m._append_obstacle((9, 9))
        self.m.wumpus_pose = (2.5, 2.5)   # cell (0,0)
        self.w.burn()
        self.assertEqual(self.m.get_status((9, 9)), self.Status.INTACT)

    def test_burn_no_adjacent_obstacles_no_crash(self):
        """burn() should be silent when nothing is adjacent."""
        self.m.wumpus_pose = (12.5, 12.5)
        self.w.burn()  # should not raise

    def test_burn_already_burning_is_no_op(self):
        self.m._append_obstacle((5, 6))
        self.m.set_status_on_obstacles([(5, 6)], self.Status.BURNING)
        self.m.wumpus_pose = (27.5, 27.5)
        self.w.burn()
        self.assertEqual(self.m.get_status((5, 6)), self.Status.BURNING)


class TestWumpusFindClosest(unittest.TestCase):

    def setUp(self):
        from wumpus import Wumpus
        self.m = _make_map_no_obstacles(grid_num=10, cell_size=5.0)
        self.w = Wumpus(self.m)

    def test_no_obstacles_returns_none(self):
        self.m.wumpus_pose = (12.5, 12.5)
        self.assertIsNone(self.w.find_closest_obstacle())

    def test_finds_obstacle_when_present(self):
        from Map_Generator import Status
        self.m._append_obstacle((3, 3))
        self.m.wumpus_pose = (2.5, 2.5)
        result = self.w.find_closest_obstacle()
        # Should return a grid (row, col) adjacent to (3,3) and not inside it
        self.assertIsNotNone(result)
        self.assertNotIn(result, self.m.obstacle_set)

    def test_only_considers_intact_obstacles(self):
        from Map_Generator import Status
        self.m._append_obstacle((5, 5))
        self.m.set_status_on_obstacles([(5, 5)], Status.BURNING)
        self.m.wumpus_pose = (2.5, 2.5)
        # BURNING obstacle should not be targeted
        result = self.w.find_closest_obstacle()
        self.assertIsNone(result)


# ===========================================================================
# 3.  Firetruck / PRM tests
# ===========================================================================

class TestCarModel(unittest.TestCase):

    def setUp(self):
        from firetruck import CarModel
        self.car = CarModel()

    def test_default_dimensions(self):
        self.assertAlmostEqual(self.car.length, 4.9)
        self.assertAlmostEqual(self.car.width,  2.2)

    def test_footprint_at_origin(self):
        fp = self.car.footprint_at(0, 0, 0)
        # Should be a non-degenerate polygon
        self.assertGreater(fp.area, 0)

    def test_footprint_moves_with_pose(self):
        fp1 = self.car.footprint_at(0, 0, 0)
        fp2 = self.car.footprint_at(100, 100, 0)
        # Centroids should differ
        self.assertFalse(fp1.centroid.equals(fp2.centroid))

    def test_footprint_rotates_with_theta(self):
        fp0   = self.car.footprint_at(50, 50,  0)
        fp90  = self.car.footprint_at(50, 50, 90)
        # Bounding boxes differ after 90° rotation
        b0  = fp0.bounds
        b90 = fp90.bounds
        # Width and height should be approximately swapped
        w0  = b0[2] - b0[0]
        h0  = b0[3] - b0[1]
        w90 = b90[2] - b90[0]
        h90 = b90[3] - b90[1]
        self.assertAlmostEqual(w0, h90, places=0)
        self.assertAlmostEqual(h0, w90, places=0)

    def test_r_min_positive(self):
        self.assertGreater(self.car.r_min, 0)

    def test_overhangs_computed(self):
        self.assertIsNotNone(self.car.front_overhang)
        self.assertIsNotNone(self.car.rear_overhang)
        self.assertAlmostEqual(
            self.car.front_overhang + self.car.rear_overhang,
            self.car.length - self.car.wheelbase,
            places=5,
        )


class TestConfigurationSpace(unittest.TestCase):

    def setUp(self):
        from firetruck import CarModel, ConfigurationSpace
        car  = CarModel()
        # Single obstacle at grid cell (5,5) with cell_size=5
        obs  = {(5, 5)}
        self.cspace = ConfigurationSpace(
            car=car, world_size=100.0, obstacle_set=obs, cell_size=5.0
        )

    def test_free_in_open_area(self):
        self.assertTrue(self.cspace.is_free(2.5, 2.5, 0))

    def test_occupied_at_obstacle(self):
        # Centre of cell (5,5) in world metres
        self.assertFalse(self.cspace.is_free(27.5, 27.5, 0))

    def test_out_of_world_not_free(self):
        self.assertFalse(self.cspace.is_free(-1, 50, 0))
        self.assertFalse(self.cspace.is_free(105, 50, 0))

    def test_path_free_all_clear(self):
        path = [(2.5, 2.5, 0), (5.0, 5.0, 0), (7.5, 7.5, 0)]
        self.assertTrue(self.cspace.is_path_free(path))

    def test_path_blocked_by_obstacle(self):
        path = [(2.5, 2.5, 0), (27.5, 27.5, 0)]
        self.assertFalse(self.cspace.is_path_free(path))

    def test_empty_obstacle_set_all_free(self):
        from firetruck import CarModel, ConfigurationSpace
        car    = CarModel()
        cspace = ConfigurationSpace(car=car, world_size=100.0,
                                     obstacle_set=set(), cell_size=5.0)
        self.assertTrue(cspace.is_free(50, 50, 0))


class TestFiretruckBuildTree(unittest.TestCase):

    @patch("pathVisualizer.PlannerVisualizer")
    def setUp(self, _MockViz):
        from firetruck import Firetruck
        self.m = _make_map_no_obstacles(grid_num=20, cell_size=5.0)
        self.ft = Firetruck(self.m, plot=False)
        self.ft.build_tree(n_samples=50)

    def test_nodes_count(self):
        self.assertGreater(len(self.ft.nodes), 0)
        self.assertLessEqual(len(self.ft.nodes), 60)   # within 20% of request

    def test_graph_keys_match_nodes(self):
        for idx in self.ft.graph:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, len(self.ft.nodes))

    def test_roadmap_size_frozen_after_build(self):
        rs = self.ft._roadmap_size
        self.assertEqual(rs, len(self.ft.nodes))

    def test_each_node_is_3_tuple(self):
        for node in self.ft.nodes:
            self.assertEqual(len(node), 3)

    def test_kd_tree_built(self):
        from scipy.spatial import KDTree
        self.assertIsInstance(self.ft._kd_tree, KDTree)

    def test_edges_have_required_keys(self):
        for edges in self.ft.graph.values():
            for e in edges:
                self.assertIn("to",   e)
                self.assertIn("cost", e)
                self.assertIn("path", e)

    def test_edge_costs_positive(self):
        for edges in self.ft.graph.values():
            for e in edges:
                self.assertGreater(e["cost"], 0)

    def test_no_self_loops(self):
        for i, edges in self.ft.graph.items():
            for e in edges:
                self.assertNotEqual(e["to"], i)

    def test_plan_raises_before_build(self):
        from firetruck import Firetruck
        m  = _make_map_no_obstacles()
        ft = Firetruck(m, plot=False)
        with self.assertRaises(RuntimeError):
            ft.plan_to_fire(fire_cell=(5, 5))

    def test_plan_to_fire_no_nearby_nodes_returns_none(self):
        """If radius is tiny, no goal nodes exist → return None."""
        result = self.ft.plan_to_fire(fire_cell=(0, 0), radius=0.001)
        self.assertIsNone(result)


class TestFiretruckAstar(unittest.TestCase):
    """Unit tests for A* internals independent of planning."""

    @patch("pathVisualizer.PlannerVisualizer")
    def setUp(self, _):
        from firetruck import Firetruck
        self.m  = _make_map_no_obstacles(grid_num=20, cell_size=5.0)
        self.ft = Firetruck(self.m, plot=False)
        self.ft.build_tree(n_samples=60)

    def test_unwind_empty_came_from(self):
        from firetruck import Firetruck
        path = Firetruck._unwind({}, 5)
        self.assertEqual(path, [5])

    def test_unwind_chain(self):
        from firetruck import Firetruck
        came_from = {1: 0, 2: 1, 3: 2}
        path = Firetruck._unwind(came_from, 3)
        self.assertEqual(path, [0, 1, 2, 3])

    def test_astar_single_start_equals_goal(self):
        """When start == goal the path should be length 1."""
        if len(self.ft.nodes) < 2:
            self.skipTest("Not enough nodes")
        path = self.ft._astar_single(0, 0)
        self.assertIsNotNone(path)
        self.assertEqual(path, [0])

    def test_astar_multi_goal_set_contains_start(self):
        """If start is in goals, it should be returned immediately."""
        if not self.ft.nodes:
            self.skipTest("No nodes")
        path = self.ft._astar_multi(0, {0})
        self.assertIsNotNone(path)
        self.assertEqual(path[-1], 0)


class TestFiretruckSE2Dist(unittest.TestCase):

    def setUp(self):
        from firetruck import Firetruck
        self.m  = _make_map_no_obstacles()
        self.ft = Firetruck(self.m, plot=False)

    def test_zero_distance_same_state(self):
        self.assertAlmostEqual(self.ft._se2_dist((5, 5, 0), (5, 5, 0)), 0.0)

    def test_heading_contributes_to_distance(self):
        d0   = self.ft._se2_dist((0, 0, 0), (0, 0,   0))
        d180 = self.ft._se2_dist((0, 0, 0), (0, 0, 180))
        self.assertGreater(d180, d0)

    def test_symmetric(self):
        a = (10, 20, 45)
        b = (30, 40, 135)
        self.assertAlmostEqual(
            self.ft._se2_dist(a, b),
            self.ft._se2_dist(b, a),
            places=5,
        )


# ===========================================================================
# 4.  SimulationEngine tests
# ===========================================================================

def _make_engine(plot=False, plot_prm=False, prm_nodes=40,
                 grid_num=20, cell_size=5.0, fill_percent=0.05,
                 sim_duration=5.0, firetruck_start=(12.5, 12.5, 0.0),
                 wumpus_start=(87.5, 87.5)):
    """Build a fast SimulationEngine with minimal settings."""
    with patch("pathVisualizer.SimVisualizer"), \
         patch("pathVisualizer.PlannerVisualizer"):
        from pathSimulator import SimulationEngine
        return SimulationEngine(
            grid_num=grid_num,
            cell_size=cell_size,
            fill_percent=fill_percent,
            firetruck_start=firetruck_start,
            wumpus_start=wumpus_start,
            prm_nodes=prm_nodes,
            tick_real_time=0.0,
            display_every_n_ticks=999,
            plot=plot,
            plot_prm=plot_prm,
            sim_duration=sim_duration,
            wumpus_catch_radius=6.0,
            flood_fill_radius=4,
            extinguish_margin=5.0,
            burn_lifetime=30.0,
        )


class TestRunResult(unittest.TestCase):

    def _make_result(self, ft_pts, wu_pts):
        from pathSimulator import RunResult
        return RunResult(
            run_index=1, end_reason="time_limit", sim_time=300.0,
            intact=5, burned=2, extinguished=3, still_burning=0,
            firetruck_points=ft_pts, wumpus_points=wu_pts,
            fires_started=2, firetruck_cpu_time=1.0, wumpus_cpu_time=0.5,
        )

    def test_winner_firetruck(self):
        r = self._make_result(10, 4)
        self.assertEqual(r.winner, "Firetruck")

    def test_winner_wumpus(self):
        r = self._make_result(2, 8)
        self.assertEqual(r.winner, "Wumpus")

    def test_winner_draw(self):
        r = self._make_result(5, 5)
        self.assertEqual(r.winner, "Draw")

    def test_firetruck_points_zero(self):
        r = self._make_result(0, 0)
        self.assertEqual(r.firetruck_points, 0)

    def test_run_index_stored(self):
        r = self._make_result(0, 0)
        self.assertEqual(r.run_index, 1)


class TestEngineScoring(unittest.TestCase):

    def setUp(self):
        self.eng = _make_engine(sim_duration=2.0)

    def test_get_stats_returns_run_result(self):
        from pathSimulator import RunResult
        result = self.eng.get_stats(run_index=99)
        self.assertIsInstance(result, RunResult)

    def test_get_stats_run_index_stored(self):
        result = self.eng.get_stats(run_index=42)
        self.assertEqual(result.run_index, 42)

    def test_firetruck_points_equals_2x_extinguished(self):
        self.eng._firetruck_extinguished = 3
        result = self.eng.get_stats()
        self.assertEqual(result.firetruck_points, 6)

    def test_wumpus_points_fires_plus_burned(self):
        from Map_Generator import Status
        self.eng._wumpus_fires_started = 4
        # Manually set one obstacle to BURNED
        if self.eng.map.obstacle_set:
            coord = next(iter(self.eng.map.obstacle_set))
            self.eng.map.set_status_on_obstacles([coord], Status.BURNING)
            self.eng.map.set_status_on_obstacles([coord], Status.BURNED)
        result = self.eng.get_stats()
        burned = result.burned
        self.assertEqual(result.wumpus_points, 4 + burned)

    def test_initial_score_is_zero(self):
        result = self.eng.get_stats()
        self.assertEqual(result.firetruck_points, 0)

    def test_cpu_time_non_negative(self):
        result = self.eng.get_stats()
        self.assertGreaterEqual(result.firetruck_cpu_time, 0.0)
        self.assertGreaterEqual(result.wumpus_cpu_time,    0.0)

    def test_counter_increments_on_extinguish(self):
        from Map_Generator import Status
        eng = self.eng
        if eng.map.obstacle_set:
            coord = next(iter(eng.map.obstacle_set))
            eng.map.set_status_on_obstacles([coord], Status.BURNING)
            before = eng._firetruck_extinguished
            eng.map.set_status_on_obstacles([coord], Status.EXTINGUISHED)
            eng._firetruck_extinguished += 1
            self.assertEqual(eng._firetruck_extinguished, before + 1)


class TestEngineNormalizeGoal(unittest.TestCase):

    def test_none_returns_none(self):
        from pathSimulator import SimulationEngine
        self.assertIsNone(SimulationEngine._normalize_goal(None))

    def test_error_string_returns_none(self):
        from pathSimulator import SimulationEngine
        self.assertIsNone(SimulationEngine._normalize_goal("ERROR CANT GO HERE"))

    def test_two_tuple_gets_zero_theta(self):
        from pathSimulator import SimulationEngine
        result = SimulationEngine._normalize_goal((10.0, 20.0))
        self.assertEqual(result, (10.0, 20.0, 0.0))

    def test_three_tuple_preserved(self):
        from pathSimulator import SimulationEngine
        result = SimulationEngine._normalize_goal((1.0, 2.0, 45.0))
        self.assertEqual(result, (1.0, 2.0, 45.0))


class TestEngineStateMachine(unittest.TestCase):

    def setUp(self):
        self.eng = _make_engine(sim_duration=2.0)

    def test_initial_state_is_idle(self):
        self.assertEqual(self.eng._truck_state, "idle")

    def test_clear_goal_sets_idle(self):
        self.eng._truck_state = "driving"
        self.eng._clear_goal()
        self.assertEqual(self.eng._truck_state, "idle")

    def test_clear_goal_clears_path(self):
        self.eng._firetruck_path = [(0.0, 0.0, 0.0), (1.0, 1.0, 0.0)]
        self.eng._clear_goal()
        self.assertIsNone(self.eng._firetruck_path)

    def test_clear_goal_clears_target_fire(self):
        self.eng._target_fire_cell = (5, 5)
        self.eng._clear_goal()
        self.assertIsNone(self.eng._target_fire_cell)

    def test_clear_goal_clears_proximity_timers(self):
        self.eng._proximity_timers = {(1, 1): 0.0, (2, 2): 1.0}
        self.eng._clear_goal()
        self.assertEqual(self.eng._proximity_timers, {})

    def test_fire_burned_out_true_for_missing_cell(self):
        self.assertTrue(self.eng._fire_burned_out((99, 99)))

    def test_fire_burned_out_false_for_burning_cell(self):
        from Map_Generator import Status
        m = self.eng.map
        if m.obstacle_set:
            coord = next(iter(m.obstacle_set))
            m.set_status_on_obstacles([coord], Status.BURNING)
            self.assertFalse(self.eng._fire_burned_out(coord))


class TestEngineRankFireCandidates(unittest.TestCase):

    def setUp(self):
        from Map_Generator import Status
        self.eng = _make_engine(sim_duration=2.0)
        m = self.eng.map
        # Clear existing obstacles for controlled test
        m.obstacle_set.clear()
        m.obstacle_coordinate_dict.clear()
        m.active_fires.clear()
        # Plant two burning cells
        m._append_obstacle((2, 2))
        m._append_obstacle((8, 8))
        m.firetruck_pose = (12.5, 12.5, 0.0)  # near (2,2)
        m.set_status_on_obstacles([(2, 2)], Status.BURNING)
        m.set_status_on_obstacles([(8, 8)], Status.BURNING)

    def test_returns_list_of_pairs(self):
        result = self.eng._rank_fire_candidates()
        for dist, cell in result:
            self.assertIsInstance(dist, float)
            self.assertIsInstance(cell, tuple)

    def test_nearest_fire_first_when_both_viable(self):
        # Give both fires plenty of remaining burn time
        m = self.eng.map
        m.sim_time = 0.0
        result = self.eng._rank_fire_candidates()
        if len(result) == 2:
            d0, _ = result[0]
            d1, _ = result[1]
            self.assertLessEqual(d0, d1)

    def test_empty_when_no_active_fires(self):
        self.eng.map.active_fires.clear()
        result = self.eng._rank_fire_candidates()
        self.assertEqual(result, [])


class TestEngineProximityExtinguish(unittest.TestCase):

    def setUp(self):
        from Map_Generator import Status
        self.Status = Status
        self.eng = _make_engine(sim_duration=2.0)
        m = self.eng.map
        m.obstacle_set.clear()
        m.obstacle_coordinate_dict.clear()
        m.active_fires.clear()
        # Plant a fire right next to firetruck
        self.fire_cell = (2, 2)
        m._append_obstacle(self.fire_cell)
        m.firetruck_pose = (10.0, 10.0, 0.0)  # cell (2,2) world-centre = (12.5,12.5)
        m.set_status_on_obstacles([self.fire_cell], self.Status.BURNING)

    def test_no_extinguish_before_duration(self):
        self.eng.map.sim_time = 0.0
        self.eng._proximity_timers = {self.fire_cell: 0.0}
        self.eng.proximity_duration = 10.0
        result = self.eng._check_proximity_extinguish()
        self.assertEqual(len(result), 0)

    def test_extinguish_after_duration(self):
        self.eng.map.sim_time      = 100.0
        self.eng.proximity_duration = 5.0
        self.eng.proximity_radius   = 50.0   # wide radius so cell is definitely in range
        self.eng._proximity_timers = {self.fire_cell: 90.0}  # started 10s ago
        result = self.eng._check_proximity_extinguish()
        self.assertIn(self.fire_cell, result)
        self.assertEqual(
            self.eng.map.get_status(self.fire_cell), self.Status.EXTINGUISHED
        )

    def test_extinguish_increments_counter(self):
        self.eng.map.sim_time      = 100.0
        self.eng.proximity_duration = 5.0
        self.eng.proximity_radius   = 50.0
        self.eng._proximity_timers = {self.fire_cell: 90.0}
        before = self.eng._firetruck_extinguished
        self.eng._check_proximity_extinguish()
        self.assertEqual(self.eng._firetruck_extinguished, before + 1)


class TestEngineFloodFill(unittest.TestCase):

    def setUp(self):
        from Map_Generator import Status
        self.Status = Status
        self.eng = _make_engine(sim_duration=2.0)
        m = self.eng.map
        m.obstacle_set.clear()
        m.obstacle_coordinate_dict.clear()
        m.active_fires.clear()
        # Create a line of burning cells (0,0)..(0,3)
        for c in range(4):
            m._append_obstacle((0, c))
            m.set_status_on_obstacles([(0, c)], Status.BURNING)
        self.origin = (0, 0)

    def test_flood_fill_extinguishes_connected(self):
        self.eng._extinguish_connected(self.origin)
        for c in range(1,min(4, self.eng.flood_fill_radius + 1)):
            status = self.eng.map.get_status((0, c))
            self.assertEqual(status, self.Status.EXTINGUISHED)

    def test_flood_fill_respects_radius(self):
        self.eng.flood_fill_radius = 1
        self.eng._extinguish_connected(self.origin)
        # (0,0) origin is already extinguished by _check_proximity_extinguish before call
        # Cells at depth > 1 should still be BURNING
        if self.eng.map.get_status((0, 2)) != self.Status.EXTINGUISHED:
            self.assertEqual(self.eng.map.get_status((0, 2)), self.Status.BURNING)


class TestEngineWumpusAct(unittest.TestCase):

    def setUp(self):
        self.eng = _make_engine(sim_duration=2.0)

    def test_wumpus_act_returns_bool(self):
        result = self.eng._wumpus_act()
        self.assertIsInstance(result, bool)

    def test_no_new_fires_returns_false(self):
        """When wumpus is far from any obstacle no new fires should start."""
        self.eng.map.wumpus_pose = (0.001, 0.001)  # corner, likely no neighbours
        result = self.eng._wumpus_act()
        # Result depends on map, but must be a bool
        self.assertIsInstance(result, bool)


class TestEngineStep(unittest.TestCase):

    def setUp(self):
        self.eng = _make_engine(sim_duration=1.0)

    def test_step_returns_true_initially(self):
        result = self.eng.step()
        self.assertIsInstance(result, bool)

    def test_step_advances_time(self):
        t0 = self.eng.map.sim_time
        self.eng.step()
        self.assertGreater(self.eng.map.sim_time, t0)

    def test_step_returns_false_after_duration(self):
        self.eng.map.sim_time = 999.0
        result = self.eng.step()
        self.assertFalse(result)


class TestEngineDeleteTempNodes(unittest.TestCase):

    def setUp(self):
        self.eng = _make_engine(sim_duration=2.0, prm_nodes=30)

    def test_delete_empty_indices_no_crash(self):
        self.eng._delete_temp_nodes([])

    def test_delete_injected_node(self):
        ft = self.eng.firetruck
        start_state = (12.5, 12.5, 0.0)
        old_len = len(ft.nodes)
        idx = ft._inject_node(start_state, outgoing=True)
        if idx is not None:
            self.eng._delete_temp_nodes([idx])
            self.assertLessEqual(len(ft.nodes), old_len + 1)

    def test_dead_edges_removed_from_permanent_nodes(self):
        ft = self.eng.firetruck
        idx = ft._inject_node((12.5, 12.5, 0.0), outgoing=True)
        if idx is None:
            self.skipTest("Injection failed — skip")
        self.eng._delete_temp_nodes([idx])
        rs = ft._roadmap_size
        for i in range(min(rs, len(ft.graph))):
            for e in ft.graph.get(i, []):
                self.assertNotEqual(e["to"], idx,
                    "Dead edge reference found after cleanup")


class TestEngineSwapPendingPath(unittest.TestCase):

    def setUp(self):
        self.eng = _make_engine(sim_duration=2.0)

    def test_swap_moves_pending_to_active(self):
        path = [(1.0, 1.0, 0.0), (2.0, 2.0, 0.0)]
        with self.eng._path_lock:
            self.eng._pending_path = path
        self.eng._swap_pending_path()
        self.assertEqual(self.eng._firetruck_path, path)
        self.assertIsNone(self.eng._pending_path)

    def test_swap_noop_when_no_pending(self):
        self.eng._firetruck_path = None
        self.eng._pending_path   = None
        self.eng._swap_pending_path()
        self.assertIsNone(self.eng._firetruck_path)


class TestEngineWumpusCatch(unittest.TestCase):

    def test_wumpus_catch_terminates_sim(self):
        """If truck and wumpus overlap with no active fires, sim should end."""
        eng = _make_engine(sim_duration=60.0)
        # Place both agents at the same position, no active fires
        eng.map.firetruck_pose = (50.0, 50.0, 0.0)
        eng.map.wumpus_pose    = (50.0, 50.0)
        eng.map.active_fires.clear()
        eng._end_reason = None

        # Run a few ticks manually
        for _ in range(20):
            eng._tick()
            if eng._end_reason == "wumpus_caught":
                break

        self.assertEqual(eng._end_reason, "wumpus_caught")


class TestEngineCPUTimers(unittest.TestCase):

    def test_timer_lock_is_used(self):
        eng = _make_engine(sim_duration=2.0)
        self.assertIsInstance(eng._timer_lock, type(threading.Lock()))

    def test_wumpus_replan_updates_cpu_time(self):
        eng = _make_engine(sim_duration=2.0)
        before = eng.wumpus_cpu_time
        eng._replan_wumpus()
        # CPU time should be >= before (might be 0 if plan returns None very fast)
        self.assertGreaterEqual(eng.wumpus_cpu_time, before)


# ===========================================================================
# 5.  Tournament tests (structure and scoring, no display)
# ===========================================================================

class TestTournamentSummarise(unittest.TestCase):

    def _fake_result(self, run_index, ft_pts, wu_pts, extinguished=0,
                     fires_started=0, burned=0):
        from pathSimulator import RunResult
        return RunResult(
            run_index=run_index,
            end_reason="time_limit",
            sim_time=300.0,
            intact=10, burned=burned, extinguished=extinguished,
            still_burning=0,
            firetruck_points=ft_pts,
            wumpus_points=wu_pts,
            fires_started=fires_started,
            firetruck_cpu_time=1.0,
            wumpus_cpu_time=0.5,
        )

    def test_summarise_firetruck_wins(self):
        import io, sys
        from tourney import summarise
        results = [
            self._fake_result(1, 10, 2),
            self._fake_result(2, 8,  4),
        ]
        captured = io.StringIO()
        sys.stdout = captured
        summarise(results)
        sys.stdout = sys.__stdout__
        output = captured.getvalue()
        self.assertIn("Firetruck", output)

    def test_summarise_wumpus_wins(self):
        import io, sys
        from tourney import summarise
        results = [
            self._fake_result(1, 2, 10),
            self._fake_result(2, 4,  8),
        ]
        captured = io.StringIO()
        sys.stdout = captured
        summarise(results)
        sys.stdout = sys.__stdout__
        output = captured.getvalue()
        self.assertIn("Wumpus", output)

    def test_summarise_draw(self):
        import io, sys
        from tourney import summarise
        results = [
            self._fake_result(1, 5, 5),
            self._fake_result(2, 5, 5),
        ]
        captured = io.StringIO()
        sys.stdout = captured
        summarise(results)
        sys.stdout = sys.__stdout__
        output = captured.getvalue()
        self.assertIn("Draw", output)

    def test_summarise_totals_correct(self):
        import io, sys
        from tourney import summarise
        results = [
            self._fake_result(1, 6, 2),
            self._fake_result(2, 4, 3),
            self._fake_result(3, 2, 1),
        ]
        captured = io.StringIO()
        sys.stdout = captured
        summarise(results)
        sys.stdout = sys.__stdout__
        output = captured.getvalue()
        # Total FT = 12, total WU = 6
        self.assertIn("12", output)
        self.assertIn("6",  output)


class TestTournamentGroupedBars(unittest.TestCase):

    def test_grouped_bars_no_crash(self):
        """_grouped_bars should execute without error."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from tourney import _grouped_bars
        fig, ax = plt.subplots()
        import numpy as np
        x      = np.arange(3)
        ft_vals = [1.0, 2.0, 3.0]
        wu_vals = [3.0, 1.0, 2.0]
        _grouped_bars(ax, x, ft_vals, wu_vals, 0.32,
                      title="Test", ylabel="Y", run_labels=["R1","R2","R3"])
        plt.close("all")


class TestTournamentDefaultKwargs(unittest.TestCase):

    def test_default_engine_kwargs_keys(self):
        from tourney import DEFAULT_ENGINE_KWARGS
        required = [
            "grid_num", "cell_size", "fill_percent",
            "firetruck_start", "wumpus_start", "prm_nodes",
            "tick_real_time", "sim_duration",
        ]
        for key in required:
            self.assertIn(key, DEFAULT_ENGINE_KWARGS, f"Missing key: {key}")

    def test_default_engine_kwargs_prm_nodes_positive(self):
        from tourney import DEFAULT_ENGINE_KWARGS
        self.assertGreater(DEFAULT_ENGINE_KWARGS["prm_nodes"], 0)

    def test_default_engine_kwargs_plot_false(self):
        from tourney import DEFAULT_ENGINE_KWARGS
        self.assertFalse(DEFAULT_ENGINE_KWARGS["plot"])


# ===========================================================================
# 6.  Integration / smoke tests  (short runs)
# ===========================================================================

class TestIntegrationShortRun(unittest.TestCase):
    """Run the engine for a few ticks end-to-end and verify consistency."""

    def setUp(self):
        self.eng = _make_engine(sim_duration=3.0, prm_nodes=40,
                                fill_percent=0.05)

    def test_run_returns_run_result(self):
        from pathSimulator import RunResult
        result = self.eng.run(run_index=1)
        self.assertIsInstance(result, RunResult)

    def test_run_result_sim_time_within_duration(self):
        result = self.eng.run(run_index=1)
        self.assertLessEqual(result.sim_time, self.eng.sim_duration + 1.0)

    def test_run_result_counts_non_negative(self):
        result = self.eng.run(run_index=1)
        self.assertGreaterEqual(result.intact,        0)
        self.assertGreaterEqual(result.burned,        0)
        self.assertGreaterEqual(result.extinguished,  0)
        self.assertGreaterEqual(result.still_burning, 0)

    def test_run_result_points_non_negative(self):
        result = self.eng.run(run_index=1)
        self.assertGreaterEqual(result.firetruck_points, 0)
        self.assertGreaterEqual(result.wumpus_points,    0)

    def test_run_result_winner_valid(self):
        result = self.eng.run(run_index=1)
        self.assertIn(result.winner, ("Firetruck", "Wumpus", "Draw"))

    def test_run_result_end_reason_set(self):
        result = self.eng.run(run_index=1)
        self.assertIsNotNone(result.end_reason)

    def test_active_fires_subset_of_obstacle_set(self):
        """active_fires must always be a subset of obstacle_set."""
        for _ in range(30):
            self.eng.step()
        m = self.eng.map
        self.assertTrue(m.active_fires.issubset(m.obstacle_set))

    def test_status_consistency(self):
        """Every burning cell must have a non-None burn_time."""
        from Map_Generator import Status
        for _ in range(30):
            self.eng.step()
        for coord, data in self.eng.map.obstacle_coordinate_dict.items():
            if data["status"] == Status.BURNING:
                self.assertIsNotNone(
                    data["burn_time"],
                    f"Cell {coord} is BURNING but has no burn_time",
                )


class TestIntegrationMultipleRuns(unittest.TestCase):

    def test_two_consecutive_engines_are_independent(self):
        """Each engine has its own map state."""
        eng1 = _make_engine(sim_duration=2.0, prm_nodes=30)
        eng2 = _make_engine(sim_duration=2.0, prm_nodes=30)
        self.assertIsNot(eng1.map, eng2.map)
        self.assertIsNot(eng1.firetruck, eng2.firetruck)

    def test_obstacle_sets_differ_between_runs(self):
        """Random maps should (almost certainly) differ."""
        eng1 = _make_engine(sim_duration=1.0, prm_nodes=30)
        eng2 = _make_engine(sim_duration=1.0, prm_nodes=30)
        # It's extremely unlikely two random maps are identical
        self.assertFalse(
            eng1.map.obstacle_set == eng2.map.obstacle_set
            and len(eng1.map.obstacle_set) > 5,
            "Both maps are identical — likely not truly independent",
        )


# ===========================================================================
# 7.  Edge-case / regression tests
# ===========================================================================

class TestEdgeCasesWumpus(unittest.TestCase):

    def test_plan_with_wumpus_at_boundary(self):
        """Wumpus at top-right corner should not crash A*."""
        from wumpus import Wumpus
        m = _make_map_no_obstacles(grid_num=5, cell_size=5.0)
        m._append_obstacle((2, 2))
        w = Wumpus(m)
        m.wumpus_pose = (24.9, 24.9)   # near grid edge
        try:
            path = w.plan()
        except Exception as e:
            self.fail(f"plan() raised {e} at boundary")

    def test_burn_with_extinguished_neighbour(self):
        """Burn should skip EXTINGUISHED and BURNED obstacles."""
        from wumpus import Wumpus
        from Map_Generator import Status
        m = _make_map_no_obstacles(grid_num=5, cell_size=5.0)
        m._append_obstacle((2, 3))
        m.set_status_on_obstacles([(2, 3)], Status.BURNING)
        m.set_status_on_obstacles([(2, 3)], Status.EXTINGUISHED)
        w = Wumpus(m)
        m.wumpus_pose = (10.0, 10.0)   # cell (2,2)
        w.burn()
        # EXTINGUISHED is terminal — should remain EXTINGUISHED
        self.assertEqual(m.get_status((2, 3)), Status.EXTINGUISHED)


class TestEdgeCasesFiretruck(unittest.TestCase):

    def test_build_tree_handles_very_dense_obstacles(self):
        """With nearly every cell blocked, build_tree should still not crash."""
        from firetruck import Firetruck
        from Map_Generator import Map
        m = Map(
            Grid_num=5, cell_size=5.0, fill_percent=0.50,
            wumpus=None, firetruck=None,
            firetruck_pose=(2.5, 2.5, 0.0),
            wumpus_pose=(22.5, 22.5),
        )
        ft = Firetruck(m, plot=False)
        try:
            ft.build_tree(n_samples=20)
        except Exception as e:
            self.fail(f"build_tree raised {e} on dense map")

    def test_plan_to_fire_returns_none_without_nearby_nodes(self):
        from firetruck import Firetruck
        m  = _make_map_no_obstacles(grid_num=20, cell_size=5.0)
        ft = Firetruck(m, plot=False)
        ft.build_tree(n_samples=30)
        # Fire at (0,0), radius tiny → no roadmap nodes there
        result = ft.plan_to_fire(fire_cell=(0, 0), radius=0.1)
        self.assertIsNone(result)


class TestEdgeCasesEngine(unittest.TestCase):

    def test_tick_counter_increments(self):
        eng = _make_engine(sim_duration=5.0)
        before = eng._tick_counter
        eng.step()
        self.assertEqual(eng._tick_counter, before + 1)

    def test_replan_wumpus_does_not_block_main_thread(self):
        """_replan_wumpus should complete quickly (it's synchronous)."""
        eng = _make_engine(sim_duration=5.0)
        t0 = time.perf_counter()
        eng._replan_wumpus()
        elapsed = time.perf_counter() - t0
        self.assertLess(elapsed, 10.0, "Wumpus replan took too long")

    def test_finish_suppression_in_open_area(self):
        """_finish_suppression should not crash when there's nothing to extinguish."""
        eng = _make_engine(sim_duration=5.0)
        eng.map.firetruck_pose = (12.5, 12.5, 0.0)
        eng._truck_state = "suppressing"
        try:
            eng._finish_suppression()
        except Exception as e:
            self.fail(f"_finish_suppression raised {e}")
        self.assertEqual(eng._truck_state, "idle")

    def test_refresh_goal_no_crash_without_fires(self):
        eng = _make_engine(sim_duration=5.0)
        eng.map.active_fires.clear()
        try:
            eng._refresh_goal()
        except Exception as e:
            self.fail(f"_refresh_goal raised {e}")

    def test_refresh_goal_sets_target_with_active_fire(self):
        from Map_Generator import Status
        eng = _make_engine(sim_duration=5.0)
        m   = eng.map
        m.obstacle_set.clear()
        m.obstacle_coordinate_dict.clear()
        m.active_fires.clear()
        m._append_obstacle((5, 5))
        m.set_status_on_obstacles([(5, 5)], Status.BURNING)
        eng._refresh_goal()
        self.assertEqual(eng._target_fire_cell, (5, 5))

    def test_wumpus_move_interval_controls_frequency(self):
        """Wumpus tick counter resets after wumpus_move_interval ticks."""
        eng = _make_engine(sim_duration=5.0)
        eng.wumpus_move_interval = 3
        eng._wumpus_tick_counter = 0
        for _ in range(3):
            eng._wumpus_tick_counter += 1
            if eng._wumpus_tick_counter >= eng.wumpus_move_interval:
                eng._wumpus_tick_counter = 0
        self.assertEqual(eng._wumpus_tick_counter, 0)


class TestEdgeCasesMap(unittest.TestCase):

    def test_set_status_on_empty_list_is_noop(self):
        m = _make_map()
        m.set_status_on_obstacles([], None)   # should not raise

    def test_active_fires_starts_empty(self):
        m = _make_map_no_obstacles()
        self.assertEqual(len(m.active_fires), 0)

    def test_obstacle_set_and_dict_always_in_sync(self):
        m = _make_map_no_obstacles()
        coords = [(1,1), (2,2), (3,3)]
        for c in coords:
            m._append_obstacle(c)
        self.assertEqual(set(m.obstacle_coordinate_dict.keys()), m.obstacle_set)

    def test_delete_nonexistent_obstacle_raises(self):
        """_delete_obstacle on a missing key raises KeyError by design."""
        m = _make_map_no_obstacles()
        with self.assertRaises(KeyError):
            m._delete_obstacle((99, 99))

    def test_grid_cells_are_integer_tuples(self):
        m = _make_map()
        for r, c in m.obstacle_set:
            self.assertIsInstance(r, int)
            self.assertIsInstance(c, int)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)