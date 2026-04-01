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
from shapely.strtree import STRtree

from rsplan import planner

from Map_Generator import Map
from pathVisualizer import PlannerVisualizer

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

State:   TypeAlias = Tuple[float, float, float]   # (x, y, theta_degrees)
Point2D: TypeAlias = Tuple[float, float]

PRM_RANDOM = random.Random()

# ---------------------------------------------------------------------------
# Reeds-Shepp implementation (via rsplan)
# ---------------------------------------------------------------------------

class _ReedsSheppPath:
    """Wrapper to maintain compatibility with your existing PRM logic."""
    def __init__(self, q0: Tuple[float, float, float], r: float, q1: Tuple[float, float, float]):
        self.q0 = q0  # (x, y, theta_rad)
        self.r = r
        # rsplan computes the optimal path (shortest distance or fewest segments)
        # We set a large step_size initially; sampling happens in sample_many()
        self._path = planner.path(q0, q1, r, 0.0 ,2.0)

    def path_length(self) -> float:
        # Summing segment lengths from the internal rsplan path object
        return sum(abs(seg.length) for seg in self._path.segments)

    def sample_many(self, step_size: float) -> Tuple[List[Tuple], None]:
        """Returns (list_of_(x, y, theta_rad), None) to mirror your current API."""
        # rsplan's waypoints() generates the interpolated states
        poses = [(w.x, w.y, w.yaw) for w in self._path.waypoints()]
        return poses, None

def reeds_shepp_shortest_path(
    q0: Tuple[float, float, float],
    q1: Tuple[float, float, float],
    r: float,
) -> Optional[_ReedsSheppPath]:
    """Shortest Reeds-Shepp path from q0 to q1 with turning radius r."""
    dx, dy = q1[0] - q0[0], q1[1] - q0[1]
    if math.hypot(dx, dy) < 1e-6 and abs(q0[2] - q1[2]) < 1e-6:
        return None
    
    # Return our wrapper object which handles the rsplan logic internally
    return _ReedsSheppPath(q0, r, q1)

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
        self.polys = [
            box(r * cell_size, c * cell_size,
                r * cell_size + cell_size, c * cell_size + cell_size)
            for r, c in obstacle_set
        ]
        self.full_obstacle_geometry = STRtree(self.polys) if self.polys else None

    def is_free(self, x: float, y: float, theta_deg: float) -> bool:
        fp = self.car.footprint_at(x, y, theta_deg)
    
        # 1. Boundary check
        if not self._world_box.contains(fp):
            return False
    
        # 2. No obstacles at all — world is open
        if self.full_obstacle_geometry is None:
            return True
    
        # 3. Spatial broad-phase via STRtree bounding-box query
        possible_matches_indices = self.full_obstacle_geometry.query(fp)
    
        # 4. Narrow-phase exact intersection
        for idx in possible_matches_indices:
            if fp.intersects(self.polys[idx]):
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
    # Reeds-Shepp helpers (updated from Dubins)
    # ------------------------------------------------------------------

    def _dubins(self, q_start: State, q_end: State) -> Optional[_ReedsSheppPath]:
        q0 = (q_start[0], q_start[1], math.radians(q_start[2]))
        q1 = (q_end[0],   q_end[1],   math.radians(q_end[2]))
        # Renamed internal call to our new Reeds-Shepp wrapper
        return reeds_shepp_shortest_path(q0, q1, self.car.r_min)

    def _dubins_length(self, q_start: State, q_end: State) -> float:
        path = self._dubins(q_start, q_end)
        return path.path_length() if path else float("inf")

    def _dubins_poses(self, q_start: State, q_end: State,
                      step_size: float = 1.0) -> List[State]:
        """Interpolated (x, y, theta_deg) poses along the Reeds-Shepp path."""
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

    def _connect_nodes(self, k_neighbors: int = 30,
                       r_connect: float = 50.0, step_size: float = 1.0) -> None:
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

    def _fire_goal_nodes(self, fire_cell: Tuple[int, int],
                         radius: float = 10.0) -> List[int]:
        """
        Return indices of all roadmap nodes whose (x,y) position lies
        within `radius` metres of the centre of `fire_cell`.

        These nodes are used directly as goal candidates in plan_to_fire()
        without any injection — every one already has full graph connectivity
        from build time, so no new Dubins solves are needed on the goal side.
        """
        cs  = self.map.cell_size
        cx  = fire_cell[0] * cs + cs / 2.0
        cy  = fire_cell[1] * cs + cs / 2.0
        pos = np.array([cx, cy])
        idxs = self._kd_tree.query_ball_point(pos, r=radius)
        return [i for i in idxs if i < self._roadmap_size]

    def _astar_multi_goal(
        self,
        start_idx: int,
        goal_set:  set,
    ) -> Optional[List[int]]:
        """
        A* search that terminates at the CHEAPEST member of goal_set.

        Unlike single-goal A*, there is no single q_goal to compute a
        heuristic against.  We use the minimum Dubins distance to any
        goal node as the heuristic — this remains admissible because the
        true cost to reach the goal set is at least as large as the
        cheapest free-space arc to the nearest goal.

        The search pops nodes in order of f = g + h.  The first time any
        goal node is popped it is guaranteed to be the cheapest reachable
        goal (A* optimality).
        """
        # Precompute heuristic: min Dubins distance from node to any goal
        h_cache: Dict[int, float] = {}

        def h(idx: int) -> float:
            if idx not in h_cache:
                q_node = self.nodes[idx]
                h_cache[idx] = min(
                    self._dubins_length(q_node, self.nodes[g])
                    for g in goal_set
                )
            return h_cache[idx]

        open_set  = [(h(start_idx), start_idx)]
        g_score   = {start_idx: 0.0}
        came_from: Dict[int, int] = {}
        visited   = set()

        while open_set:
            _, current = heapq.heappop(open_set)
            if current in visited:
                continue
            visited.add(current)

            if current in goal_set:
                return self._unwind(came_from, current)

            for edge in self.graph.get(current, []):
                nbr   = edge["to"]
                new_g = g_score[current] + edge["cost"]
                if new_g < g_score.get(nbr, float("inf")):
                    g_score[nbr]   = new_g
                    came_from[nbr] = current
                    heapq.heappush(open_set, (new_g + h(nbr), nbr))

        return None

    def _astar_multi_goal_cost(
        self,
        start_idx: int,
        goal_set:  set,
    ) -> float:
        """
        Same as _astar_multi_goal but returns only the cost.
        Used by cost_to_fire() to rank fires without building the path list.
        """
        h_cache: Dict[int, float] = {}

        def h(idx: int) -> float:
            if idx not in h_cache:
                q_node = self.nodes[idx]
                h_cache[idx] = min(
                    self._dubins_length(q_node, self.nodes[g])
                    for g in goal_set
                )
            return h_cache[idx]

        open_set = [(h(start_idx), start_idx)]
        g_score  = {start_idx: 0.0}
        visited  = set()

        while open_set:
            _, current = heapq.heappop(open_set)
            if current in visited:
                continue
            visited.add(current)
            if current in goal_set:
                return g_score[current]
            for edge in self.graph.get(current, []):
                nbr   = edge["to"]
                new_g = g_score[current] + edge["cost"]
                if new_g < g_score.get(nbr, float("inf")):
                    g_score[nbr] = new_g
                    heapq.heappush(open_set, (new_g + h(nbr), nbr))

        return float("inf")

    def plan_to_fire(
        self,
        fire_cell:   Tuple[int, int],
        start_state: Optional[State] = None,
        radius:      float = 10.0,
    ) -> Optional[List[State]]:
        """
        Plan from start_state to the cheapest roadmap node within `radius`
        metres of `fire_cell`, injecting only the start node.

        Why no goal injection
        ---------------------
        The old plan(goal_state) approach:
          1. Computed a geometric stop-short point along the truck→fire vector.
          2. Injected it as a temp goal node, trying Dubins connections to
             the roadmap.
          3. Ran single-goal A* to that one point.

        Problem: the stop-short point was computed from the current truck
        position, so it always picked the face of the obstacle closest to
        where the truck currently is — not the face that is cheapest to
        actually reach via the road network.  A fire behind a wall cluster
        would produce a goal on the near side that required a long detour,
        when the far side was directly reachable via an existing corridor.

        This method instead:
          1. Finds all roadmap nodes already within `radius` metres of the
             fire centre (KD-tree lookup, no Dubins solves).
          2. Injects only the start node (half the Dubins overhead).
          3. Runs multi-goal A* — terminates at whichever goal node is
             cheapest via the existing graph.  The graph chooses the best
             approach angle automatically.
        """
        if self._roadmap_size == 0:
            raise RuntimeError("Call build_tree() before plan_to_fire().")

        if start_state is None:
            fp = self.map.firetruck_pose
            start_state = (float(fp[0]), float(fp[1]), float(fp[2]))

        goal_nodes = self._fire_goal_nodes(fire_cell, radius)
        if not goal_nodes:
            return None

        start_idx = self._inject_query_node(start_state, outgoing=True)
        if start_idx is None:
            self._cleanup_query_nodes()
            return None

        path_indices = self._astar_multi_goal(start_idx, set(goal_nodes))
        waypoints    = self._reconstruct_path(path_indices) if path_indices else None
        self._cleanup_query_nodes()
        return waypoints

    def cost_to_fire(
        self,
        fire_cell:   Tuple[int, int],
        start_state: Optional[State] = None,
        radius:      float = 10.0,
    ) -> float:
        """
        Cheapest A* cost (metres) from start_state to any roadmap node
        within `radius` metres of `fire_cell`.  Only injects start node.
        Used by the engine to rank fires before committing to a plan.
        Returns float('inf') if no goal node is reachable.
        """
        if self._roadmap_size == 0:
            return float("inf")

        if start_state is None:
            fp = self.map.firetruck_pose
            start_state = (float(fp[0]), float(fp[1]), float(fp[2]))

        goal_nodes = self._fire_goal_nodes(fire_cell, radius)
        if not goal_nodes:
            return float("inf")

        start_idx = self._inject_query_node(start_state, outgoing=True)
        if start_idx is None:
            self._cleanup_query_nodes()
            return float("inf")

        cost = self._astar_multi_goal_cost(start_idx, set(goal_nodes))
        self._cleanup_query_nodes()
        return cost

    def plan(self, goal_state: State,
             start_state: Optional[State] = None) -> Optional[List[State]]:
        """
        Single-goal planner — kept for wumpus chase and precise targets.
        Injects both start and goal as temp nodes.
        """
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

        waypoints = self._reconstruct_path(path_indices) if path_indices else None
        self._cleanup_query_nodes()
        return waypoints

    def _se2_distance(self, q1: State, q2: State) -> float:
        """
        SE(2)-aware distance combining position and heading.

        The KD-tree only indexes (x,y).  Two nodes at the same position
        but opposite headings are distance-0 in the tree but require a
        full U-turn in Dubins space.  Adding a heading term weighted by
        r_min (in metres) gives a unified distance that correctly
        deprioritises poorly-aligned neighbours.

            d = sqrt(dx² + dy² + (r_min × Δθ_wrapped_rad)²)
        """
        dx  = q1[0] - q2[0]
        dy  = q1[1] - q2[1]
        dth = abs(q1[2] - q2[2]) % 360
        if dth > 180:
            dth = 360 - dth
        dth_rad = math.radians(dth)
        return math.sqrt(dx*dx + dy*dy + (self.car.r_min * dth_rad) ** 2)

    def _inject_query_node(self, q: State, outgoing: bool,
                           k: int = 10, r: float = 40.0) -> Optional[int]:
        idx = len(self.nodes)
        self.nodes.append(q)
        self.graph[idx] = []

        pos     = np.array([q[0], q[1]])
        n_query = min(k * 3, self._roadmap_size)
        if n_query == 0:
            return None

        _, k_idxs = self._kd_tree.query(pos, k=n_query)
        r_idxs    = self._kd_tree.query_ball_point(pos, r=r)
        raw_cands = (set(np.atleast_1d(k_idxs).tolist()) | set(r_idxs)) - {idx}
        raw_cands = {c for c in raw_cands if c < self._roadmap_size}

        # Sort by SE(2) distance — try most heading-compatible nodes first
        candidates = sorted(raw_cands,
                            key=lambda j: self._se2_distance(q, self.nodes[j]))

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
                # Rebuild the edge list, filtering out any 'to' that is in temp_indices
                self.graph[i] = [
                    edge for edge in self.graph[i]
                    if edge["to"] < self._roadmap_size
                ]

        # 4. Truncate the nodes list back to permanent size
        self.nodes = self.nodes[:self._roadmap_size]

    # ------------------------------------------------------------------
    # A* search
    # ------------------------------------------------------------------

    def _astar(self, start_idx: int, goal_idx: int) -> Optional[List[int]]:
        q_goal   = self.nodes[goal_idx]
        h_cache: Dict[int, float] = {}

        def h(idx: int) -> float:
            if idx not in h_cache:
                h_cache[idx] = self._dubins_length(self.nodes[idx], q_goal)
            return h_cache[idx]

        open_set  = [(h(start_idx), start_idx)]
        g_score   = {start_idx: 0.0}
        came_from: Dict[int, int] = {}
        visited   = set()

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
                    heapq.heappush(open_set, (new_g + h(nbr), nbr))
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
        print('started building')
        self.build_tree()
        start_time = time.time()
        print(f"Build time: {start_time - build_time:.4f}s")
        goal = (40.0, 40.0, 45.0)
        path = self.plan(goal)
        print(f"Query time: {time.time() - start_time:.4f}s")
        if self.viz:
            self.viz.plot_prm(self.map, self.graph, self.nodes, path=path)
        print(f"Path found: {len(path)} waypoints." if path else "No path found.")