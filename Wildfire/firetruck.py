"""
firetruck_prm.py
================
Nonholonomic PRM planner for a firetruck.

Fix in this version
--------------------
  _cleanup_query_nodes used list.pop() which shifts all indices after the
  removed element, silently corrupting self.nodes for the next query.

  Solution: track the roadmap size at build time (_roadmap_size). At query
  time, temporary nodes are appended beyond this boundary. Cleanup simply
  truncates back to _roadmap_size and removes temp edges — no index shifting.
"""

from __future__ import annotations

import heapq
import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TypeAlias

import numpy as np
from scipy.spatial import KDTree
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from Map_Generator import Map, Status
from pathVisualizer import PlannerVisualizer
from pathSimulator import PathSimulator

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

State: TypeAlias = Tuple[float, float, float]   # (x, y, theta_degrees)
Point2D: TypeAlias = Tuple[float, float]

PRM_RANDOM = random.Random()


# ===========================================================================
# CarModel
# ===========================================================================

@dataclass
class CarModel:
    """
    Geometric and kinematic description of the firetruck.

    Footprint origin: REAR AXLE centre (Ackermann / Dubins convention).
      +x = forward, +y = left

    Overhang split assumes symmetric front/rear unless overridden.
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
            (-self.rear_overhang,                    -hw),
            (-self.rear_overhang,                     hw),
            ( self.wheelbase + self.front_overhang,   hw),
            ( self.wheelbase + self.front_overhang,  -hw),
        ])

    def footprint_at(self, x: float, y: float, theta_deg: float) -> Polygon:
        rotated = rotate(self._local_footprint, theta_deg,
                         origin=(0.0, 0.0), use_radians=False)
        return translate(rotated, xoff=x, yoff=y)

    @property
    def max_steering_angle_deg(self) -> float:
        return math.degrees(math.atan(self.wheelbase / self.r_min))

    def __repr__(self) -> str:
        return (
            f"CarModel(length={self.length}m, width={self.width}m, "
            f"wheelbase={self.wheelbase}m, r_min={self.r_min}m)"
        )


# ===========================================================================
# ConfigurationSpace
# ===========================================================================

class ConfigurationSpace:
    def __init__(
        self,
        car: CarModel,
        world_size: float,
        obstacle_set,
        cell_size: float = 5.0,
    ):
        self.car        = car
        self.world_size = world_size
        self.cell_size  = cell_size
        self._world_box = box(0.01, 0.01, world_size, world_size)
        self.full_obstacle_geometry: Optional[Polygon] = None
        self._prepare_obstacles(obstacle_set)

    def _prepare_obstacles(self, obstacle_set) -> None:
        polys = []
        for row, col in obstacle_set:
            x_min = row * self.cell_size
            y_min = col * self.cell_size
            polys.append(box(x_min, y_min,
                             x_min + self.cell_size,
                             y_min + self.cell_size))
        self.full_obstacle_geometry = unary_union(polys) if polys else None

    def is_free(self, x: float, y: float, theta_deg: float) -> bool:
        footprint = self.car.footprint_at(x, y, theta_deg)
        if not footprint.within(self._world_box):
            return False
        if (self.full_obstacle_geometry is not None
                and footprint.intersects(self.full_obstacle_geometry)):
            return False
        return True

    def is_path_free(self, poses: List[State]) -> bool:
        for x, y, theta in poses:
            if not self.is_free(x, y, theta):
                return False
        return True


# ===========================================================================
# DubinsPlanner
# ===========================================================================

@dataclass
class DubinsEdge:
    node_from:   int
    node_to:     int
    cost:        float
    path_type:   str
    seg_lengths: Tuple[float, float, float] = (0.0, 0.0, 0.0)


class DubinsPlanner:
    _PATH_TYPES = ['LSL', 'RSR', 'LSR', 'RSL', 'RLR', 'LRL']

    def __init__(self, car: CarModel, cspace: ConfigurationSpace):
        self.r      = car.r_min
        self.cspace = cspace

    def compute_edge(
        self,
        node_from: int, q_start: State,
        node_to:   int, q_end:   State,
        step_size: float = 1.0,
    ) -> Optional[DubinsEdge]:
        result = self._shortest_path(q_start, q_end)
        if result is None:
            return None
        path_type, seg_lengths, total_length = result
        poses = self._interpolate(q_start, seg_lengths, path_type, step_size)
        if not self.cspace.is_path_free(poses):
            return None
        return DubinsEdge(
            node_from=node_from,
            node_to=node_to,
            cost=total_length,
            path_type=path_type,
            seg_lengths=seg_lengths,
        )

    def interpolate_edge(
        self, q_start: State, edge: DubinsEdge, step_size: float = 4.0
    ) -> List[State]:
        return self._interpolate(q_start, edge.seg_lengths,
                                 edge.path_type, step_size)

    def path_length(self, q_start: State, q_end: State) -> float:
        result = self._shortest_path(q_start, q_end)
        return result[2] if result is not None else float('inf')

    # ------------------------------------------------------------------
    # Dubins geometry
    # ------------------------------------------------------------------

    def _shortest_path(self, q_start, q_end):
        x0, y0, h0 = q_start[0], q_start[1], math.radians(q_start[2])
        x1, y1, h1 = q_end[0],   q_end[1],   math.radians(q_end[2])

        dx, dy = x1 - x0, y1 - y0
        D = math.hypot(dx, dy)
        if D < 1e-6:
            return None

        theta = math.atan2(dy, dx)
        r = self.r
        d = D / r
        a = self._mod2pi(h0 - theta)
        b = self._mod2pi(h1 - theta)

        best, best_len = None, float('inf')
        for ptype in self._PATH_TYPES:
            result = self._compute_type(ptype, d, a, b)
            if result is None:
                continue
            t, p, q = result
            if t < 0 or q < 0:
                continue
            if ptype in ('RLR', 'LRL') and p < 0:
                continue
            total = (t + abs(p) + q) * r
            if total < best_len:
                best_len = total
                best = (ptype, (t * r, abs(p) * r, q * r), total)

        return best

    def _compute_type(self, ptype, d, a, b):
        sa, ca = math.sin(a), math.cos(a)
        sb, cb = math.sin(b), math.cos(b)

        if ptype == 'LSL':
            p_sq = 2 + d*d - 2*math.cos(a-b) + 2*d*(sa - sb)
            if p_sq < 0: return None
            p   = math.sqrt(p_sq)
            tmp = math.atan2(cb - ca, d + sa - sb)
            return self._mod2pi(-a + tmp), p, self._mod2pi(b - tmp)

        if ptype == 'RSR':
            p_sq = 2 + d*d - 2*math.cos(a-b) + 2*d*(sb - sa)
            if p_sq < 0: return None
            p   = math.sqrt(p_sq)
            tmp = math.atan2(ca - cb, d - sa + sb)
            return self._mod2pi(a - tmp), p, self._mod2pi(self._mod2pi(-b) + tmp)

        if ptype == 'LSR':
            p_sq = -2 + d*d + 2*math.cos(a-b) + 2*d*(sa + sb)
            if p_sq < 0: return None
            p   = math.sqrt(p_sq)
            tmp = math.atan2(-ca - cb, d + sa + sb) - math.atan2(-2.0, p)
            return self._mod2pi(-a + tmp), p, self._mod2pi(-self._mod2pi(b) + tmp)

        if ptype == 'RSL':
            p_sq = -2 + d*d + 2*math.cos(a-b) - 2*d*(sa + sb)
            if p_sq < 0: return None
            p   = math.sqrt(p_sq)
            tmp = math.atan2(ca + cb, d - sa - sb) - math.atan2(2.0, p)
            return self._mod2pi(a - tmp), p, self._mod2pi(b - tmp)

        if ptype == 'RLR':
            val = (6 - d*d + 2*math.cos(a-b) + 2*d*(sa - sb)) / 8.0
            if abs(val) > 1: return None
            p   = self._mod2pi(2*math.pi - math.acos(val))
            t   = self._mod2pi(a - math.atan2(ca-cb, d-sa+sb) + p/2.0)
            q   = self._mod2pi(a - b - t + p)
            return t, p, q

        if ptype == 'LRL':
            val = (6 - d*d + 2*math.cos(a-b) + 2*d*(-sa + sb)) / 8.0
            if abs(val) > 1: return None
            p   = self._mod2pi(2*math.pi - math.acos(val))
            t   = self._mod2pi(-a + math.atan2(-ca+cb, d+sa-sb) + p/2.0)
            q   = self._mod2pi(self._mod2pi(b) - a - t + p)
            return t, p, q

        return None

    def _interpolate(self, q_start, seg_lengths, path_type, step_size):
        x, y  = q_start[0], q_start[1]
        theta = math.radians(q_start[2])
        r     = self.r
        poses = [(x, y, q_start[2])]

        for seg_idx, seg_len in enumerate(seg_lengths):
            turn = path_type[seg_idx]
            dist_remaining = seg_len
            while dist_remaining > 1e-6:
                step = min(step_size, dist_remaining)
                if turn == 'S':
                    x += step * math.cos(theta)
                    y += step * math.sin(theta)
                elif turn == 'L':
                    d_theta = step / r
                    cx = x - r * math.sin(theta)
                    cy = y + r * math.cos(theta)
                    theta += d_theta
                    x = cx + r * math.sin(theta)
                    y = cy - r * math.cos(theta)
                elif turn == 'R':
                    d_theta = step / r
                    cx = x + r * math.sin(theta)
                    cy = y - r * math.cos(theta)
                    theta -= d_theta
                    x = cx - r * math.sin(theta)
                    y = cy + r * math.cos(theta)
                poses.append((x, y, math.degrees(theta)))
                dist_remaining -= step

        return poses

    @staticmethod
    def _mod2pi(angle: float) -> float:
        return angle % (2.0 * math.pi)


# ===========================================================================
# Firetruck PRM
# ===========================================================================

class Firetruck:
    def __init__(self, map: Map, plot: bool = False):
        self.map = map

        self.car = CarModel(
            length=4.9, width=2.2, wheelbase=3.0,
            r_min=13.0, v_max=10.0,
        )
        world_size = map.grid_num * map.cell_size
        self.cspace = ConfigurationSpace(
            car=self.car,
            world_size=world_size,
            obstacle_set=map.obstacle_set,
            cell_size=map.cell_size,
        )
        self.dubins = DubinsPlanner(self.car, self.cspace)

        self.nodes: List[State]                 = []
        self.graph: Dict[int, List[DubinsEdge]] = {}
        self._kd_tree: Optional[KDTree]         = None

        # Boundary between permanent roadmap nodes and temporary query nodes.
        # Set after build_tree(); never changes until the next build.
        self._roadmap_size: int = 0

        self.viz = PlannerVisualizer((self.car.width, self.car.length)) if plot else None

    # =======================================================================
    # BUILD PHASE
    # =======================================================================

    def build_tree(self, n_samples: int = 1000) -> None:
        print("Sampling free configurations...")
        self._sample_points(n_samples)

        print(f"Connecting {len(self.nodes)} nodes...")
        self._connect_nodes()

        self._roadmap_size = len(self.nodes)
        n_edges = sum(len(v) for v in self.graph.values())
        print(f"PRM built: {self._roadmap_size} nodes, {n_edges} directed edges")

    def _sample_points(self, n_samples: int = 1000) -> None:
        limit    = self.map.grid_num * self.map.cell_size
        self.nodes = []
        self.graph = {}
        attempts   = 0
        max_att    = n_samples * 20

        while len(self.nodes) < n_samples and attempts < max_att:
            attempts += 1
            tx     = PRM_RANDOM.uniform(5.0, limit - 5.0)
            ty     = PRM_RANDOM.uniform(5.0, limit - 5.0)
            ttheta = PRM_RANDOM.randrange(0, 360, 15)

            if self.cspace.is_free(tx, ty, ttheta):
                self.nodes.append((tx, ty, ttheta))
                self.graph[len(self.nodes) - 1] = []

        if len(self.nodes) < n_samples:
            print(f"  Warning: only found {len(self.nodes)}/{n_samples} "
                  f"free configs after {max_att} attempts.")

        xy = np.array([(n[0], n[1]) for n in self.nodes])
        self._kd_tree = KDTree(xy)

    def _connect_nodes(
        self,
        k_neighbors: int   = 20,
        r_connect:   float = 30.0,
        step_size:   float = 1.0,
    ) -> None:
        if not self.nodes or self._kd_tree is None:
            return

        for i, q_i in enumerate(self.nodes):
            pos_i = np.array([q_i[0], q_i[1]])

            _, k_idxs  = self._kd_tree.query(
                pos_i, k=min(k_neighbors + 1, len(self.nodes))
            )
            r_idxs     = self._kd_tree.query_ball_point(pos_i, r=r_connect)
            candidates = set(np.atleast_1d(k_idxs).tolist()) | set(r_idxs)
            candidates.discard(i)

            for j in candidates:
                edge = self.dubins.compute_edge(
                    i, self.nodes[i],
                    j, self.nodes[j],
                    step_size=step_size,
                )
                if edge is not None:
                    self.graph[i].append(edge)

    # =======================================================================
    # QUERY PHASE
    # =======================================================================

    def plan(
        self,
        goal_state:  State,
        start_state: Optional[State] = None,
    ) -> Optional[List[State]]:
        if self._roadmap_size == 0:
            raise RuntimeError("Call build_tree() before plan().")

        if start_state is None:
            fp = self.map.firetruck_pose
            start_state = (float(fp[0]), float(fp[1]), 0.0)

        # Inject start then goal — both land beyond _roadmap_size
        start_idx = self._inject_query_node(start_state, outgoing=True)
        goal_idx  = self._inject_query_node(goal_state,  outgoing=False)

        path_indices: Optional[List[int]] = None

        if start_idx is None:
            print("plan(): could not connect start pose to PRM.")
        elif goal_idx is None:
            print("plan(): could not connect goal pose to PRM.")
        else:
            path_indices = self._astar(start_idx, goal_idx)
            if path_indices is None:
                print("plan(): A* found no path through the PRM.")

        # Always clean up before returning
        self._cleanup_query_nodes()

        if not path_indices:
            return None

        # path_indices referenced nodes that existed before cleanup —
        # permanent roadmap indices are still valid; temp indices are gone.
        # _reconstruct_path only touches permanent indices, so this is safe.
        return self._reconstruct_path(path_indices)

    # ------------------------------------------------------------------
    # Query-node injection
    # ------------------------------------------------------------------

    def _inject_query_node(
        self,
        q: State,
        outgoing: bool,
        k: int   = 10,
        r: float = 40.0,
    ) -> Optional[int]:
        """
        Append q beyond _roadmap_size and connect it to the permanent roadmap.

        outgoing=True  → edges q → roadmap  (use for start)
        outgoing=False → edges roadmap → q  (use for goal)

        Returns the new index, or None if no valid Dubins connection was found.
        """
        idx = len(self.nodes)
        self.nodes.append(q)
        self.graph[idx] = []

        pos     = np.array([q[0], q[1]])
        n_query = min(k * 3, self._roadmap_size)
        if n_query == 0:
            return None

        _, k_idxs = self._kd_tree.query(pos, k=n_query)
        r_idxs    = self._kd_tree.query_ball_point(pos, r=r)

        candidates = set(np.atleast_1d(k_idxs).tolist()) | set(r_idxs)
        candidates.discard(idx)
        candidates = {c for c in candidates if c < self._roadmap_size}

        connected = False
        for j in candidates:
            if outgoing:
                edge = self.dubins.compute_edge(idx, q, j, self.nodes[j])
                if edge:
                    self.graph[idx].append(edge)
                    connected = True
            else:
                edge = self.dubins.compute_edge(j, self.nodes[j], idx, q)
                if edge:
                    self.graph[j].append(edge)
                    connected = True

        return idx if connected else None

    def _cleanup_query_nodes(self) -> None:
        """
        Remove all temporary nodes added since build_tree() completed.

        FIX: original used list.pop() which renumbers every subsequent
        index, corrupting the permanent roadmap for future queries.

        This version:
          1. Deletes temp indices from self.graph.
          2. Strips any edges from permanent nodes that point into
             the temp region (goal edges added during injection).
          3. Truncates self.nodes to _roadmap_size — O(1), no shifting.
        """
        if len(self.nodes) <= self._roadmap_size:
            return

        temp_indices = set(range(self._roadmap_size, len(self.nodes)))

        # Remove temp graph entries
        for idx in temp_indices:
            self.graph.pop(idx, None)

        # Strip goal-directed edges from permanent nodes
        for i in range(self._roadmap_size):
            if i in self.graph:
                self.graph[i] = [
                    e for e in self.graph[i]
                    if e.node_to < self._roadmap_size
                ]

        # Truncate node list — permanent indices untouched
        self.nodes = self.nodes[:self._roadmap_size]

    # ------------------------------------------------------------------
    # A* search
    # ------------------------------------------------------------------

    def _astar(
        self, start_idx: int, goal_idx: int
    ) -> Optional[List[int]]:
        q_goal   = self.nodes[goal_idx]
        h0       = self.dubins.path_length(self.nodes[start_idx], q_goal)
        open_set = [(h0, start_idx)]
        g_score  = {start_idx: 0.0}
        came_from: Dict[int, int] = {}
        visited  = set()

        while open_set:
            _, current = heapq.heappop(open_set)

            if current in visited:
                continue
            visited.add(current)

            if current == goal_idx:
                return self._reconstruct_index_path(came_from, current)

            for edge in self.graph.get(current, []):
                nbr   = edge.node_to
                new_g = g_score[current] + edge.cost

                if new_g < g_score.get(nbr, float('inf')):
                    g_score[nbr]   = new_g
                    came_from[nbr] = current
                    h = self.dubins.path_length(self.nodes[nbr], q_goal)
                    heapq.heappush(open_set, (new_g + h, nbr))

        return None

    # ------------------------------------------------------------------
    # Path reconstruction
    # ------------------------------------------------------------------

    def _reconstruct_index_path(
        self,
        came_from: Dict[int, int],
        current: int,
    ) -> List[int]:
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(current)
        return path[::-1]

    def _reconstruct_path(self, index_path: List[int]) -> List[State]:
        if not index_path:
            return []

        waypoints: List[State] = [self.nodes[index_path[0]]]

        for k in range(len(index_path) - 1):
            i, j = index_path[k], index_path[k + 1]

            edge = next(
                (e for e in self.graph.get(i, []) if e.node_to == j),
                None,
            )
            if edge is None:
                waypoints.append(self.nodes[j])
                continue

            poses = self.dubins.interpolate_edge(
                self.nodes[i], edge, step_size=0.5
            )
            waypoints.extend(poses[1:])

        return waypoints

    # =======================================================================
    # Main loop
    # =======================================================================

    def main_run(self) -> None:
        self.build_tree()

        start_time = time.time()
        goal = (240.0, 240.0, 0.0)
        path = self.plan(goal)
        elapsed = time.time() - start_time

        print(f"Query time: {elapsed:.4f}s")

        if self.viz:
            self.viz.plot_prm(self.map, self.graph, self.nodes, path=path)

        if path:
            print(f"Path found: {len(path)} waypoints.")
        else:
            print("No path found.")