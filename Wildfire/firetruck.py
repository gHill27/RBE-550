"""
firetruck.py
============
Nonholonomic PRM planner for the firetruck using Reeds-Shepp curves (rsplan).

Public API
----------
  build_tree(n_samples)          — sample + connect the roadmap (one-time)
  plan_to_fire(fire_cell, ...)   — multi-goal A* to roadmap nodes near a fire
  plan(goal_state, ...)          — single-goal A* (wumpus chase / point goals)

State tuples are (x_m, y_m, theta_degrees) throughout.
Reeds-Shepp computations work internally in radians.
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TypeAlias

import numpy as np
from scipy.spatial import KDTree
from shapely.affinity import rotate, translate, affine_transform
from shapely.geometry import Polygon, box
from shapely.strtree import STRtree

from rsplan import planner
from Map_Generator import Map
from pathVisualizer import PlannerVisualizer
import time

State: TypeAlias = Tuple[float, float, float]   # (x_m, y_m, theta_deg)


# ---------------------------------------------------------------------------
# Reeds-Shepp wrapper (rsplan)
# ---------------------------------------------------------------------------

class _RSPath:
    """Thin wrapper around rsplan.path for length and waypoint sampling."""
    def __init__(self, q0, q1, r: float):
        self._path = planner.path(q0, q1, r, 0.0, 2.0)

    def length(self) -> float:
        return sum(abs(s.length) for s in self._path.segments)

    def poses_deg(self) -> List[State]:
        """Interpolated (x, y, theta_deg) waypoints."""
        return [(w.x, w.y, math.degrees(w.yaw)) for w in self._path.waypoints()]


def _rs_path(q_start: State, q_end: State, r: float) -> Optional[_RSPath]:
    """
    Return shortest RS path between two (x, y, theta_deg) states, or None
    when start == end (no movement needed).
    """
    q0 = (q_start[0], q_start[1], math.radians(q_start[2]))
    q1 = (q_end[0],   q_end[1],   math.radians(q_end[2]))
    if math.hypot(q1[0]-q0[0], q1[1]-q0[1]) < 1e-6 and abs(q0[2]-q1[2]) < 1e-6:
        return None
    return _RSPath(q0, q1, r)


# ---------------------------------------------------------------------------
# Car geometry
# ---------------------------------------------------------------------------

@dataclass
class CarModel:
    """
    Firetruck footprint + kinematic limits.
    Coordinate origin: rear-axle centre (+x forward, +y left).
    """
    length:         float = 4.9
    width:          float = 2.2
    wheelbase:      float = 3.0
    r_min:          float = 13.0    # minimum turning radius (m)
    v_max:          float = 10.0    # maximum speed (m/s) — used for triage
    front_overhang: Optional[float] = None
    rear_overhang:  Optional[float] = None

    def __post_init__(self):
        oh = self.length - self.wheelbase
        if self.front_overhang is None: self.front_overhang = oh / 2.0
        if self.rear_overhang  is None: self.rear_overhang  = oh / 2.0
        hw = self.width / 2.0
        self._footprint = Polygon([
            (-self.rear_overhang,                   -hw),
            (-self.rear_overhang,                    hw),
            (self.wheelbase + self.front_overhang,   hw),
            (self.wheelbase + self.front_overhang,  -hw),
        ])

    def footprint_at(self, x: float, y: float, theta_deg: float) -> Polygon:
        angle = math.radians(theta_deg)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        # Define the 2D Affine Matrix: [a, b, d, e, xoff, yoff]
        # Representing:
        # x' = a*x + b*y + xoff
        # y' = d*x + e*y + yoff
        matrix = [cos_a, -sin_a, sin_a, cos_a, float(x), float(y)]
        
        return affine_transform(self._footprint, matrix)


# ---------------------------------------------------------------------------
# Configuration space (collision checking)
# ---------------------------------------------------------------------------

class ConfigurationSpace:
    def __init__(self, car: CarModel, world_size: float,
                 obstacle_set, cell_size: float):
        self.car        = car
        self._world_box = box(0.01, 0.01, world_size, world_size)
        self.polys      = [
            box(r*cell_size, c*cell_size, (r+1)*cell_size, (c+1)*cell_size)
            for r, c in obstacle_set
        ]
        # STRtree is None when the map has no obstacles (open world)
        self._tree = STRtree(self.polys) if self.polys else None

    def is_free(self, x: float, y: float, theta_deg: float) -> bool:
        fp = self.car.footprint_at(x, y, theta_deg)
        if not self._world_box.contains(fp):
            return False
        if self._tree is None:
            return True
        for idx in self._tree.query(fp):
            if fp.intersects(self.polys[idx]):
                return False
        return True

    def is_path_free(self, poses: List[State]) -> bool:
        return all(self.is_free(x, y, th) for x, y, th in poses)


# ---------------------------------------------------------------------------
# PRM Planner
# ---------------------------------------------------------------------------

class Firetruck:
    """
    Probabilistic Roadmap planner with Reeds-Shepp curve edges.

    Build once with build_tree(), then query repeatedly with plan_to_fire()
    or plan().  The engine is responsible for deferred cleanup of injected
    temporary nodes (see simulation_engine._delete_temp_nodes).
    """

    def __init__(self, map: Map, plot: bool = False):
        self.map    = map
        self.car    = CarModel()
        self.cspace = ConfigurationSpace(
            car=self.car,
            world_size=map.grid_num * map.cell_size,
            obstacle_set=map.obstacle_set,
            cell_size=map.cell_size,
        )
        self.nodes: List[State]           = []
        self.graph: Dict[int, List[dict]] = {}   # edge = {"to", "cost", "path"}
        self._kd_tree: Optional[KDTree]   = None
        self._roadmap_size: int           = 0
        self.viz = PlannerVisualizer((self.car.width, self.car.length)) if plot else None

    # ------------------------------------------------------------------
    # Reeds-Shepp helpers
    # ------------------------------------------------------------------

    def _rs_length(self, q_start: State, q_end: State) -> float:
        p = _rs_path(q_start, q_end, self.car.r_min)
        return p.length() if p else float("inf")

    def _rs_poses(self, q_start: State, q_end: State) -> List[State]:
        """Interpolated (x, y, theta_deg) poses along the RS path."""
        p = _rs_path(q_start, q_end, self.car.r_min)
        return p.poses_deg() if p else []

    def _se2_dist(self, q1: State, q2: State) -> float:
        """
        SE(2) distance combining position and heading, weighted by r_min.
        Used to prioritise heading-compatible neighbours during injection,
        reducing the number of RS paths that fail collision checking.
        """
        dth = abs(q1[2] - q2[2]) % 360
        if dth > 180: dth = 360 - dth
        return math.sqrt(
            (q1[0]-q2[0])**2 + (q1[1]-q2[1])**2
            + (self.car.r_min * math.radians(dth))**2
        )

    # ------------------------------------------------------------------
    # Build phase
    # ------------------------------------------------------------------

    def build_tree(self, n_samples: int = 200) -> None:
        print("Sampling free configurations...")
        t0 = time.perf_counter()
        self._sample_points(n_samples)
        t1 = time.perf_counter()
        sample_time = t1 - t0
        print(f"Connecting {len(self.nodes)} nodes...")
        self._connect_nodes()
        t2 = time.perf_counter()
        connect_time = t2 - t1
        self._roadmap_size = len(self.nodes)
        n_edges = sum(len(v) for v in self.graph.values())
        print(f"PRM built: {self._roadmap_size} nodes, {n_edges} directed edges")
        print(f"  [PRM Build] Sampling: {sample_time:.4f}s | Connections: {connect_time:.4f}s")

    def _sample_points(self, n_samples: int) -> None:
        limit = self.map.grid_num * self.map.cell_size
        self.nodes, self.graph = [], {}
        attempts, max_att = 0, n_samples * 20

        while len(self.nodes) < n_samples and attempts < max_att:
            attempts += 1
            tx     = random.uniform(5.0, limit - 5.0)
            ty     = random.uniform(5.0, limit - 5.0)
            ttheta = random.randrange(0, 360, 45)
            if self.cspace.is_free(tx, ty, ttheta):
                idx = len(self.nodes)
                self.nodes.append((tx, ty, ttheta))
                self.graph[idx] = []

        if len(self.nodes) < n_samples:
            print(f"  Warning: only {len(self.nodes)}/{n_samples} "
                  f"configs found after {max_att} attempts.")

        xy = np.array([(n[0], n[1]) for n in self.nodes])
        self._kd_tree = KDTree(xy)

    def _connect_nodes(self, k: int = 30, r: float = 50.0) -> None:
        """
        Connect each node to its k nearest neighbours and all nodes within
        radius r.  Candidates are sorted by SE(2) distance so heading-
        compatible edges are attempted first, improving early-exit efficiency.
        """
        if not self.nodes or self._kd_tree is None:
            return
        xy = np.array([(n[0], n[1]) for n in self.nodes])
        for i, q_i in enumerate(self.nodes):
            _, k_idx = self._kd_tree.query(xy[i], k=min(k+1, len(self.nodes)))
            r_idx    = self._kd_tree.query_ball_point(xy[i], r=r)
            cands    = sorted(
                (set(k_idx.tolist()) | set(r_idx)) - {i},
                key=lambda j: self._se2_dist(q_i, self.nodes[j]),
            )
            for j in cands:
                poses = self._rs_poses(q_i, self.nodes[j])
                if poses and self.cspace.is_path_free(poses):
                    self.graph[i].append({
                        "to":   j,
                        "cost": self._rs_length(q_i, self.nodes[j]),
                        "path": poses,
                    })

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def _fire_goal_nodes(self, fire_cell: Tuple[int, int],
                         radius: float = 10.0) -> List[int]:
        """
        Return indices of permanent roadmap nodes within `radius` metres of
        the centre of `fire_cell`.  These nodes require no injection — they
        already have full graph connectivity from build time.
        """
        cs = self.map.cell_size
        cx = fire_cell[0] * cs + cs / 2.0
        cy = fire_cell[1] * cs + cs / 2.0
        return [i for i in self._kd_tree.query_ball_point([cx, cy], r=radius)
                if i < self._roadmap_size]

    def _inject_node(self, q: State, outgoing: bool,
                     k: int = 10, r: float = 40.0) -> Optional[int]:
        """
        Temporarily add q to the graph.
          outgoing=True  → edges q → neighbours  (start node)
          outgoing=False → edges neighbours → q   (goal node)

        Returns the new node index on success, None if no connection found.
        The engine is responsible for removing injected nodes via
        _delete_temp_nodes when they are no longer needed.
        """
        idx = len(self.nodes)
        self.nodes.append(q)
        self.graph[idx] = []

        pos   = np.array([q[0], q[1]])
        n_q   = min(k*3, self._roadmap_size)
        if n_q == 0:
            return None

        _, k_idx = self._kd_tree.query(pos, k=n_q)
        r_idx    = self._kd_tree.query_ball_point(pos, r=r)
        cands    = sorted(
            {c for c in (set(k_idx.tolist()) | set(r_idx)) if c < self._roadmap_size},
            key=lambda j: self._se2_dist(q, self.nodes[j]),
        )

        connected = False
        for j in cands:
            q_src, q_dst = (q, self.nodes[j]) if outgoing else (self.nodes[j], q)
            src, dst     = (idx, j)            if outgoing else (j, idx)
            poses = self._rs_poses(q_src, q_dst)
            if poses and self.cspace.is_path_free(poses):
                self.graph[src].append({
                    "to":   dst,
                    "cost": self._rs_length(q_src, q_dst),
                    "path": poses,
                })
                connected = True

        return idx if connected else None

    # ------------------------------------------------------------------
    # Public planners
    # ------------------------------------------------------------------

    def plan_to_fire(self, fire_cell: Tuple[int, int],
                     start_state: Optional[State] = None,
                     radius: float = 10.0) -> Optional[List[State]]:
        """
        Multi-goal A* from start_state to the cheapest permanent roadmap
        node within `radius` metres of fire_cell.

        Only the start node is injected (no goal injection needed — goal
        candidates are permanent nodes with full graph connectivity).
        The caller (engine) owns cleanup of the injected start node.
        """
        if self._roadmap_size == 0:
            raise RuntimeError("Call build_tree() first.")

        if start_state is None:
            fp = self.map.firetruck_pose
            start_state = (float(fp[0]), float(fp[1]), float(fp[2]))

        goal_nodes = self._fire_goal_nodes(fire_cell, radius)
        if not goal_nodes:
            return None

        start_idx = self._inject_node(start_state, outgoing=True)
        if start_idx is None:
            # Could not connect — remove the orphaned node
            self.nodes.pop()
            self.graph.pop(len(self.nodes), None)
            return None

        idx_path = self._astar_multi(start_idx, set(goal_nodes))
        return self._build_waypoints(idx_path) if idx_path else None

    def plan(self, goal_state: State,
             start_state: Optional[State] = None) -> Optional[List[State]]:
        """
        Single-goal A* for precise targets (wumpus roadmap node).
        Injects both start and goal nodes.  Caller owns cleanup.
        """
        if self._roadmap_size == 0:
            raise RuntimeError("Call build_tree() first.")
        if start_state is None:
            fp = self.map.firetruck_pose
            start_state = (float(fp[0]), float(fp[1]), 0.0)

        start_idx = self._inject_node(start_state, outgoing=True)
        goal_idx  = self._inject_node(goal_state,  outgoing=False)

        if start_idx is None or goal_idx is None:
            print("plan(): could not connect start or goal to PRM.")
            return None

        idx_path = self._astar_single(start_idx, goal_idx)
        if idx_path is None:
            print("plan(): A* found no path.")
        return self._build_waypoints(idx_path) if idx_path else None

    # ------------------------------------------------------------------
    # A* search
    # ------------------------------------------------------------------

    def _astar_multi(self, start: int, goals: set) -> Optional[List[int]]:
        """
        A* to the cheapest reachable goal in `goals`.
        Heuristic: min RS-length from current node to any goal.
        Admissible because RS-length ≤ true graph path cost.
        Results are cached per node to avoid recomputing across expansions.
        """
        h_cache: Dict[int, float] = {}

        def h(idx: int) -> float:
            if idx not in h_cache:
                q = self.nodes[idx]
                h_cache[idx] = min(self._rs_length(q, self.nodes[g]) for g in goals)
            return h_cache[idx]

        open_set  = [(h(start), start)]
        g_score   = {start: 0.0}
        came_from: Dict[int, int] = {}
        visited:   set            = set()

        while open_set:
            _, cur = heapq.heappop(open_set)
            if cur in visited:
                continue
            visited.add(cur)
            if cur in goals:
                return self._unwind(came_from, cur)
            for e in self.graph.get(cur, []):
                nb, ng = e["to"], g_score[cur] + e["cost"]
                if ng < g_score.get(nb, float("inf")):
                    g_score[nb]   = ng
                    came_from[nb] = cur
                    heapq.heappush(open_set, (ng + h(nb), nb))
        return None

    def _astar_single(self, start: int, goal: int) -> Optional[List[int]]:
        """Standard A* to a single goal node."""
        q_goal    = self.nodes[goal]
        h_cache: Dict[int, float] = {}

        def h(idx: int) -> float:
            if idx not in h_cache:
                h_cache[idx] = self._rs_length(self.nodes[idx], q_goal)
            return h_cache[idx]

        open_set  = [(h(start), start)]
        g_score   = {start: 0.0}
        came_from: Dict[int, int] = {}
        visited:   set            = set()

        while open_set:
            _, cur = heapq.heappop(open_set)
            if cur in visited:
                continue
            visited.add(cur)
            if cur == goal:
                return self._unwind(came_from, cur)
            for e in self.graph.get(cur, []):
                nb, ng = e["to"], g_score[cur] + e["cost"]
                if ng < g_score.get(nb, float("inf")):
                    g_score[nb]   = ng
                    came_from[nb] = cur
                    heapq.heappush(open_set, (ng + h(nb), nb))
        return None

    @staticmethod
    def _unwind(came_from: Dict[int, int], cur: int) -> List[int]:
        path = []
        while cur in came_from:
            path.append(cur)
            cur = came_from[cur]
        path.append(cur)
        return path[::-1]

    def _build_waypoints(self, idx_path: List[int]) -> List[State]:
        if not idx_path:
            return []
        wp: List[State] = [self.nodes[idx_path[0]]]
        for a, b in zip(idx_path, idx_path[1:]):
            edge = next((e for e in self.graph.get(a, []) if e["to"] == b), None)
            wp.extend(edge["path"][1:] if edge else [self.nodes[b]])
        return wp