"""
firetruck_prm.py
================
Nonholonomic PRM planner for a firetruck.

Dubins curves are computed by an inlined pure-Python implementation —
no external `dubins` package required, works on any Python version.

Public State tuples are (x, y, theta_degrees) throughout.
The Dubins helpers work internally in radians.
"""

from __future__ import annotations

import heapq
import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TypeAlias

import numpy as np
from scipy.spatial import KDTree
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from Map_Generator import Map
from pathVisualizer import PlannerVisualizer

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

State:   TypeAlias = Tuple[float, float, float]   # (x, y, theta_degrees)
Point2D: TypeAlias = Tuple[float, float]

PRM_RANDOM = random.Random()

# ---------------------------------------------------------------------------
# Pure-Python Dubins implementation
# ---------------------------------------------------------------------------
# Mirrors the three-call API of the `dubins` Cython library:
#   path   = dubins_shortest_path(q0_rad, q1_rad, r_min)
#   length = path.path_length()
#   poses  = path.sample_many(step_size)   # list of (x, y, theta_rad)
# ---------------------------------------------------------------------------

_PATH_TYPES = ["LSL", "RSR", "LSR", "RSL", "RLR", "LRL"]


def _mod2pi(a: float) -> float:
    return a % (2.0 * math.pi)


def _dubins_compute(path_type: str, d: float, a: float, b: float):
    """Return (t, p, q) normalised segment lengths, or None if invalid."""
    sa, ca = math.sin(a), math.cos(a)
    sb, cb = math.sin(b), math.cos(b)

    if path_type == "LSL":
        p_sq = 2 + d*d - 2*math.cos(a - b) + 2*d*(sa - sb)
        if p_sq < 0: return None
        p = math.sqrt(p_sq)
        tmp = math.atan2(cb - ca, d + sa - sb)
        return _mod2pi(-a + tmp), p, _mod2pi(b - tmp)

    if path_type == "RSR":
        p_sq = 2 + d*d - 2*math.cos(a - b) + 2*d*(sb - sa)
        if p_sq < 0: return None
        p = math.sqrt(p_sq)
        tmp = math.atan2(ca - cb, d - sa + sb)
        return _mod2pi(a - tmp), p, _mod2pi(_mod2pi(-b) + tmp)

    if path_type == "LSR":
        p_sq = -2 + d*d + 2*math.cos(a - b) + 2*d*(sa + sb)
        if p_sq < 0: return None
        p = math.sqrt(p_sq)
        tmp = math.atan2(-ca - cb, d + sa + sb) - math.atan2(-2.0, p)
        return _mod2pi(-a + tmp), p, _mod2pi(-_mod2pi(b) + tmp)

    if path_type == "RSL":
        p_sq = -2 + d*d + 2*math.cos(a - b) - 2*d*(sa + sb)
        if p_sq < 0: return None
        p = math.sqrt(p_sq)
        tmp = math.atan2(ca + cb, d - sa - sb) - math.atan2(2.0, p)
        return _mod2pi(a - tmp), p, _mod2pi(b - tmp)

    if path_type == "RLR":
        val = (6 - d*d + 2*math.cos(a - b) + 2*d*(sa - sb)) / 8.0
        if abs(val) > 1: return None
        p = _mod2pi(2*math.pi - math.acos(val))
        t = _mod2pi(a - math.atan2(ca - cb, d - sa + sb) + p / 2.0)
        return t, p, _mod2pi(a - b - t + p)

    if path_type == "LRL":
        val = (6 - d*d + 2*math.cos(a - b) + 2*d*(-sa + sb)) / 8.0
        if abs(val) > 1: return None
        p = _mod2pi(2*math.pi - math.acos(val))
        t = _mod2pi(-a + math.atan2(-ca + cb, d + sa - sb) + p / 2.0)
        return t, p, _mod2pi(_mod2pi(b) - a - t + p)

    return None


class _DubinsPath:
    """Lightweight Dubins path object returned by dubins_shortest_path()."""

    def __init__(self, q0, r: float, path_type: str,
                 seg_lengths: Tuple[float, float, float]):
        self.q0          = q0            # (x, y, theta_rad)
        self.r           = r
        self.path_type   = path_type
        self.seg_lengths = seg_lengths   # physical lengths (meters)

    def path_length(self) -> float:
        return sum(self.seg_lengths)

    def sample_many(self, step_size: float) -> Tuple[List[Tuple], None]:
        """Return (list_of_(x,y,theta_rad), None) — mirrors pydubins API."""
        x, y, theta = self.q0
        r     = self.r
        poses = [(x, y, theta)]

        for seg_len, turn in zip(self.seg_lengths, self.path_type):
            remaining = seg_len
            while remaining > 1e-9:
                step = min(step_size, remaining)
                if turn == "S":
                    x     += step * math.cos(theta)
                    y     += step * math.sin(theta)
                elif turn == "L":
                    dth    = step / r
                    cx, cy = x - r*math.sin(theta), y + r*math.cos(theta)
                    theta += dth
                    x, y   = cx + r*math.sin(theta), cy - r*math.cos(theta)
                elif turn == "R":
                    dth    = step / r
                    cx, cy = x + r*math.sin(theta), y - r*math.cos(theta)
                    theta -= dth
                    x, y   = cx - r*math.sin(theta), cy + r*math.cos(theta)
                poses.append((x, y, theta))
                remaining -= step

        return poses, None


def dubins_shortest_path(
    q0: Tuple[float, float, float],
    q1: Tuple[float, float, float],
    r: float,
) -> Optional[_DubinsPath]:
    """
    Shortest Dubins path from q0 to q1 with turning radius r.
    q0, q1 = (x, y, theta_radians).  Returns None if configurations coincide.
    """
    dx, dy = q1[0] - q0[0], q1[1] - q0[1]
    D = math.hypot(dx, dy)
    if D < 1e-6:
        return None

    theta = math.atan2(dy, dx)
    d, a, b = D / r, _mod2pi(q0[2] - theta), _mod2pi(q1[2] - theta)

    best_len, best_type, best_segs = float("inf"), None, None
    for ptype in _PATH_TYPES:
        result = _dubins_compute(ptype, d, a, b)
        if result is None:
            continue
        t, p, q = result
        if t < 0 or q < 0 or (ptype in ("RLR", "LRL") and p < 0):
            continue
        total = (t + abs(p) + q) * r
        if total < best_len:
            best_len, best_type, best_segs = total, ptype, (t*r, abs(p)*r, q*r)

    return _DubinsPath(q0, r, best_type, best_segs) if best_type else None


# ===========================================================================
# CarModel
# ===========================================================================

@dataclass
class CarModel:
    """
    Geometric and kinematic description of the firetruck.
    Footprint origin: REAR AXLE centre (+x forward, +y left).
    """
    length:         float = 4.9
    width:          float = 2.2
    wheelbase:      float = 3.0
    r_min:          float = 13.0
    v_max:          float = 10.0
    front_overhang: Optional[float] = None
    rear_overhang:  Optional[float] = None

    def __post_init__(self):
        overhang = self.length - self.wheelbase
        if self.front_overhang is None:
            self.front_overhang = overhang / 2.0
        if self.rear_overhang is None:
            self.rear_overhang = overhang / 2.0
        hw = self.width / 2.0
        self._local_footprint = Polygon([
            (-self.rear_overhang,                   -hw),
            (-self.rear_overhang,                    hw),
            (self.wheelbase + self.front_overhang,   hw),
            (self.wheelbase + self.front_overhang,  -hw),
        ])

    def footprint_at(self, x: float, y: float, theta_deg: float) -> Polygon:
        rotated = rotate(self._local_footprint, theta_deg,
                         origin=(0.0, 0.0), use_radians=False)
        return translate(rotated, xoff=x, yoff=y)

    def __repr__(self) -> str:
        return (f"CarModel(length={self.length}m, width={self.width}m, "
                f"wheelbase={self.wheelbase}m, r_min={self.r_min}m)")


# ===========================================================================
# ConfigurationSpace
# ===========================================================================

class ConfigurationSpace:
    def __init__(self, car: CarModel, world_size: float,
                 obstacle_set, cell_size: float = 5.0):
        self.car        = car
        self._world_box = box(0.01, 0.01, world_size, world_size)
        polys = [
            box(r * cell_size, c * cell_size,
                r * cell_size + cell_size, c * cell_size + cell_size)
            for r, c in obstacle_set
        ]
        self.full_obstacle_geometry = unary_union(polys) if polys else None

    def is_free(self, x: float, y: float, theta_deg: float) -> bool:
        fp = self.car.footprint_at(x, y, theta_deg)
        if not fp.within(self._world_box):
            return False
        if (self.full_obstacle_geometry is not None
                and fp.intersects(self.full_obstacle_geometry)):
            return False
        return True

    def is_path_free(self, poses: List[State]) -> bool:
        return all(self.is_free(x, y, th) for x, y, th in poses)


# ===========================================================================
# Firetruck PRM
# ===========================================================================

class Firetruck:
    """PRM planner with Dubins-curve edges (pure-Python, no external dubins lib)."""

    def __init__(self, map: Map, plot: bool = False):
        self.map    = map
        self.car    = CarModel()
        world_size  = map.grid_num * map.cell_size
        self.cspace = ConfigurationSpace(
            car=self.car, world_size=world_size,
            obstacle_set=map.obstacle_set, cell_size=map.cell_size,
        )
        self.nodes: List[State]           = []
        self.graph: Dict[int, List[dict]] = {}   # edge = {"to", "cost", "path"}
        self._kd_tree: Optional[KDTree]   = None
        self._roadmap_size: int           = 0
        self.viz = PlannerVisualizer((self.car.width, self.car.length)) if plot else None

    # ------------------------------------------------------------------
    # Dubins helpers
    # ------------------------------------------------------------------

    def _dubins(self, q_start: State, q_end: State) -> Optional[_DubinsPath]:
        q0 = (q_start[0], q_start[1], math.radians(q_start[2]))
        q1 = (q_end[0],   q_end[1],   math.radians(q_end[2]))
        return dubins_shortest_path(q0, q1, self.car.r_min)

    def _dubins_length(self, q_start: State, q_end: State) -> float:
        path = self._dubins(q_start, q_end)
        return path.path_length() if path else float("inf")

    def _dubins_poses(self, q_start: State, q_end: State,
                      step_size: float = 1.0) -> List[State]:
        """Interpolated (x, y, theta_deg) poses along the shortest Dubins path."""
        path = self._dubins(q_start, q_end)
        if path is None:
            return []
        configs, _ = path.sample_many(step_size)
        return [(x, y, math.degrees(th)) for x, y, th in configs]

    # ------------------------------------------------------------------
    # BUILD PHASE
    # ------------------------------------------------------------------

    def build_tree(self, n_samples: int = 200) -> None:
        print("Sampling free configurations...")
        self._sample_points(n_samples)
        print(f"Connecting {len(self.nodes)} nodes...")
        self._connect_nodes()
        self._roadmap_size = len(self.nodes)
        n_edges = sum(len(v) for v in self.graph.values())
        print(f"PRM built: {self._roadmap_size} nodes, {n_edges} directed edges")

    def _sample_points(self, n_samples: int) -> None:
        limit = self.map.grid_num * self.map.cell_size
        self.nodes, self.graph = [], {}
        attempts, max_att = 0, n_samples * 20

        while len(self.nodes) < n_samples and attempts < max_att:
            attempts += 1
            tx     = PRM_RANDOM.uniform(5.0, limit - 5.0)
            ty     = PRM_RANDOM.uniform(5.0, limit - 5.0)
            ttheta = PRM_RANDOM.randrange(0, 360, 45)
            if self.cspace.is_free(tx, ty, ttheta):
                idx = len(self.nodes)
                self.nodes.append((tx, ty, ttheta))
                self.graph[idx] = []

        if len(self.nodes) < n_samples:
            print(f"  Warning: only {len(self.nodes)}/{n_samples} "
                  f"configs found after {max_att} attempts.")

        xy = np.array([(n[0], n[1]) for n in self.nodes])
        self._kd_tree = KDTree(xy)

    def _connect_nodes(self, k_neighbors: int = 20,
                       r_connect: float = 30.0, step_size: float = 1.0) -> None:
        if not self.nodes or self._kd_tree is None:
            return
        for i, q_i in enumerate(self.nodes):
            pos_i     = np.array([q_i[0], q_i[1]])
            _, k_idxs = self._kd_tree.query(pos_i, k=min(k_neighbors + 1, len(self.nodes)))
            r_idxs    = self._kd_tree.query_ball_point(pos_i, r=r_connect)
            candidates = (set(np.atleast_1d(k_idxs).tolist()) | set(r_idxs)) - {i}
            for j in candidates:
                poses = self._dubins_poses(q_i, self.nodes[j], step_size)
                if poses and self.cspace.is_path_free(poses):
                    cost = self._dubins_length(q_i, self.nodes[j])
                    self.graph[i].append({"to": j, "cost": cost, "path": poses})

    # ------------------------------------------------------------------
    # QUERY PHASE
    # ------------------------------------------------------------------

    def plan(self, goal_state: State,
             start_state: Optional[State] = None) -> Optional[List[State]]:
        if self._roadmap_size == 0:
            raise RuntimeError("Call build_tree() before plan().")
        if start_state is None:
            fp = self.map.firetruck_pose
            start_state = (float(fp[0]), float(fp[1]), 0.0)

        start_idx = self._inject_query_node(start_state, outgoing=True)
        goal_idx  = self._inject_query_node(goal_state,  outgoing=False)

        path_indices = None
        if start_idx is None:
            print("plan(): could not connect start pose to PRM.")
        elif goal_idx is None:
            print("plan(): could not connect goal pose to PRM.")
        else:
            path_indices = self._astar(start_idx, goal_idx)
            if path_indices is None:
                print("plan(): A* found no path through the PRM.")

        # Reconstruct BEFORE cleanup: path_indices reference temp nodes
        # (start_idx, goal_idx) which are still in self.nodes at this point.
        # Calling _cleanup_query_nodes() first truncates self.nodes back to
        # _roadmap_size, making those temp indices invalid → IndexError.
        waypoints = self._reconstruct_path(path_indices) if path_indices else None
        self._cleanup_query_nodes()
        return waypoints

    def _inject_query_node(self, q: State, outgoing: bool,
                           k: int = 10, r: float = 40.0) -> Optional[int]:
        idx = len(self.nodes)
        self.nodes.append(q)
        self.graph[idx] = []

        pos     = np.array([q[0], q[1]])
        n_query = min(k * 3, self._roadmap_size)
        if n_query == 0:
            return None

        _, k_idxs  = self._kd_tree.query(pos, k=n_query)
        r_idxs     = self._kd_tree.query_ball_point(pos, r=r)
        candidates = (set(np.atleast_1d(k_idxs).tolist()) | set(r_idxs)) - {idx}
        candidates = {c for c in candidates if c < self._roadmap_size}

        connected = False
        for j in candidates:
            q_src, q_dst = (q, self.nodes[j]) if outgoing else (self.nodes[j], q)
            src,   dst   = (idx, j)            if outgoing else (j, idx)
            poses = self._dubins_poses(q_src, q_dst)
            if poses and self.cspace.is_path_free(poses):
                cost = self._dubins_length(q_src, q_dst)
                self.graph[src].append({"to": dst, "cost": cost, "path": poses})
                connected = True

        return idx if connected else None

    def _cleanup_query_nodes(self) -> None:
        """Truncate temp nodes; no index shifting into permanent roadmap."""
        if len(self.nodes) <= self._roadmap_size:
            return
        temp = set(range(self._roadmap_size, len(self.nodes)))
        for idx in temp:
            self.graph.pop(idx, None)
        for i in range(self._roadmap_size):
            if i in self.graph:
                self.graph[i] = [e for e in self.graph[i]
                                  if e["to"] < self._roadmap_size]
        self.nodes = self.nodes[:self._roadmap_size]

    # ------------------------------------------------------------------
    # A* search
    # ------------------------------------------------------------------

    def _astar(self, start_idx: int, goal_idx: int) -> Optional[List[int]]:
        q_goal   = self.nodes[goal_idx]
        open_set = [(self._dubins_length(self.nodes[start_idx], q_goal), start_idx)]
        g_score  = {start_idx: 0.0}
        came_from: Dict[int, int] = {}
        visited  = set()

        while open_set:
            _, current = heapq.heappop(open_set)
            if current in visited:
                continue
            visited.add(current)
            if current == goal_idx:
                return self._unwind(came_from, current)
            for edge in self.graph.get(current, []):
                nbr   = edge["to"]
                new_g = g_score[current] + edge["cost"]
                if new_g < g_score.get(nbr, float("inf")):
                    g_score[nbr]   = new_g
                    came_from[nbr] = current
                    h = self._dubins_length(self.nodes[nbr], q_goal)
                    heapq.heappush(open_set, (new_g + h, nbr))
        return None

    @staticmethod
    def _unwind(came_from: Dict[int, int], current: int) -> List[int]:
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(current)
        return path[::-1]

    # ------------------------------------------------------------------
    # Path reconstruction
    # ------------------------------------------------------------------

    def _reconstruct_path(self, index_path: List[int]) -> List[State]:
        if not index_path:
            return []
        waypoints: List[State] = [self.nodes[index_path[0]]]
        for k in range(len(index_path) - 1):
            i, j = index_path[k], index_path[k + 1]
            edge = next((e for e in self.graph.get(i, []) if e["to"] == j), None)
            waypoints.extend(edge["path"][1:] if edge else [self.nodes[j]])
        return waypoints

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def main_run(self) -> None:
        build_time = time.time()
        self.build_tree()
        start_time = time.time()
        print(f"Build time: {start_time - build_time:.4f}s")
        goal = (40.0, 40.0, 45.0)
        path = self.plan(goal)
        print(f"Query time: {time.time() - start_time:.4f}s")
        if self.viz:
            self.viz.plot_prm(self.map, self.graph, self.nodes, path=path)
        print(f"Path found: {len(path)} waypoints." if path else "No path found.")