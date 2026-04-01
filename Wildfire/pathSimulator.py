"""
simulation_engine.py
====================
SimulationEngine — orchestrates Map, Firetruck, Wumpus, and SimVisualizer.

Architecture
------------
- Truck state machine: "idle" → "driving" → "suppressing" → "idle"
- Fire goal selection: Euclidean + burn-time triage, ranked candidate list
- Replanning runs in a daemon thread; main loop never blocks on A*
- Injected PRM nodes are kept for 2 goal cycles (deferred cleanup) so the
  truck stays graph-connected while switching targets
- Display is throttled to every N ticks for higher throughput
- Wumpus goal targets the closest PRM roadmap node to the wumpus position
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Deque, List, Optional, Set, Tuple

from Map_Generator import Map, Status
from firetruck import Firetruck
from wumpus import Wumpus
from pathVisualizer import SimVisualizer

State = Tuple[float, float, float]

_END_TIME   = "time_limit"
_END_WUMPUS = "wumpus_caught"
_END_MAP    = "map_done"


class SimulationEngine:
    """
    Parameters
    ----------
    grid_num, cell_size, fill_percent : map geometry
    firetruck_start : (x_m, y_m, theta_deg)
    wumpus_start    : (x_m, y_m)
    prm_nodes       : PRM roadmap sample count
    tick_real_time  : wall-clock sleep per tick (0.0 = max speed)
    display_every_n_ticks : render every N ticks (higher = faster sim)
    sim_duration    : hard stop in sim-seconds (default 300 = 5 min)
    wumpus_catch_radius : truck–wumpus distance that counts as a catch (m)
    flood_fill_radius   : BFS depth for connected-obstacle extinguish
    extinguish_margin   : extra seconds of burn budget required to target a fire
    burn_lifetime       : total seconds a cell burns before self-extinguishing
    """

    def __init__(
        self,
        grid_num:              int   = 50,
        cell_size:             float = 5.0,
        fill_percent:          float = 0.15,
        firetruck_start:       State = (25.0, 25.0, 0.0),
        wumpus_start:          Tuple[float, float] = (220.0, 220.0),
        prm_nodes:             int   = 500,
        tick_real_time:        float = 0.0,
        display_every_n_ticks: int   = 5,
        plot:                  bool  = True,
        plot_prm:              bool  = False,
        sim_duration:          float = 300.0,
        wumpus_catch_radius:   float = 5.0,
        flood_fill_radius:     int   = 4,
        extinguish_margin:     float = 5.0,
        burn_lifetime:         float = 30.0,
    ):
        self.tick_real_time        = tick_real_time
        self.display_every_n_ticks = max(1, display_every_n_ticks)
        self.sim_duration          = sim_duration
        self.wumpus_catch_radius   = wumpus_catch_radius
        self.flood_fill_radius     = flood_fill_radius
        self.extinguish_margin     = extinguish_margin
        self.burn_lifetime         = burn_lifetime

        self._end_reason:    Optional[str] = None
        self._tick_counter:  int           = 0

        # Path written by bg thread, read by main — always hold _path_lock
        self._path_lock:      threading.Lock         = threading.Lock()
        self._firetruck_path: Optional[List[State]] = None
        self._pending_path:   Optional[List[State]] = None
        self._replan_pending: bool                  = False

        self._wumpus_path:       Optional[List]           = None
        self._target_fire_cell:  Optional[Tuple[int, int]] = None
        self._fire_candidates:   List[Tuple[int, int]]     = []
        self.approach_radius:    float = 10.0

        self._truck_state:       str   = "idle"
        self._suppress_start:    float = 0.0
        self.suppress_duration:  float = 8.0

        self._proximity_timers:  dict  = {}
        self.proximity_radius:   float = 10.0
        self.proximity_duration: float = 5.0

        # Start at 0 so the first wumpus move happens after wumpus_move_interval ticks
        self._wumpus_tick_counter: int = 0
        self.wumpus_move_interval: int = 10

        # Deferred PRM node cleanup: retain last 2 batches of temp-node indices
        # so the truck stays graph-connected when switching goals
        self._pending_cleanup: Deque[List[int]] = deque(maxlen=2)
        # Thread-safe accumulators for rolling time
        self._total_truck_replan_time: float = 0.0
        self._total_wumpus_replan_time: float = 0.0
        self._timer_lock = threading.Lock()

        print("[Engine] Building map...")
        self.map = Map(
            Grid_num       = grid_num,
            cell_size      = cell_size,
            fill_percent   = fill_percent,
            wumpus         = None,
            firetruck      = None,
            firetruck_pose = firetruck_start,
            wumpus_pose    = wumpus_start,
        )

        print("[Engine] Initialising agents...")
        self.firetruck = Firetruck(self.map, plot=plot_prm)
        self.wumpus    = Wumpus(self.map)
        self.map.firetruck = self.firetruck
        self.map.wumpus    = self.wumpus

        print(f"[Engine] Building PRM roadmap ({prm_nodes} nodes)...")
        # Start the rolling timer for the initial tree build
        start_tree_t = time.perf_counter()
        self.firetruck.build_tree(n_samples=prm_nodes)
        duration = time.perf_counter() - start_tree_t
        with self._timer_lock:
            # We add this to the truck's rolling replan time 
            # as it is essentially the "global" plan.
            self._total_truck_replan_time += duration

        print(f"[Engine] Roadmap ready in {duration:.4f}s.")
        if plot_prm and self.firetruck.viz is not None:
            self.firetruck.viz.plot_prm(
                self.map, self.firetruck.graph, self.firetruck.nodes, path=None
            )

        self.viz: Optional[SimVisualizer] = None
        if plot:
            self.viz = SimVisualizer(self.map, figsize=(16, 11))

        print("[Engine] Ready.")

    # =======================================================================
    # Public API
    # =======================================================================

    def run(self) -> None:
        print(f"[Engine] Starting — duration={self.sim_duration}s, "
              f"display_every={self.display_every_n_ticks} ticks")
        self._refresh_goal()
        while self.map.sim_time <= self.sim_duration and self._end_reason is None:
            self._tick()
        print(f"[Engine] Ended at t={self.map.sim_time:.1f}s "
              f"[{self._end_reason or _END_TIME}]")
        self._shutdown()

    def step(self) -> bool:
        if self.map.sim_time > self.sim_duration or self._end_reason is not None:
            return False
        self._tick()
        return self.map.sim_time <= self.sim_duration and self._end_reason is None

    # =======================================================================
    # Core tick
    # =======================================================================

    def _tick(self) -> None:
        """
        One simulation step (0.1 s of sim time).

        Ordering per tick
        -----------------
        1. Advance map clock + fire spread (map.main)
        2. Wumpus-catch termination check
        3. Swap any newly-computed background path into active use
        4. Truck state machine (idle / driving / suppressing)
        5. Wumpus movement (every wumpus_move_interval ticks)
        6. Wumpus burns; if new fires lit, replan wumpus immediately
        7. Throttled display update
        8. Optional real-time pacing sleep
        """
        self._tick_counter += 1

        # 1. Advance sim clock and fire-spread
        if self.map.main() == "Done":
            self._end_reason = _END_MAP
            return

        # 2. Wumpus-catch check — inlined for one fewer function call per tick
        ft, wp = self.map.firetruck_pose, self.map.wumpus_pose
        if math.hypot(ft[0] - wp[0], ft[1] - wp[1]) <= self.wumpus_catch_radius * 2 and len(self.map.active_fires) == 0:
            print(f"[Engine] WUMPUS CAUGHT at t={self.map.sim_time:.1f}s")
            self._end_reason = _END_WUMPUS
            return

        # 3. Pick up any path the background thread just finished
        self._swap_pending_path()

        # 4. Truck state machine
        if self._truck_state == "idle":
            if not self._replan_pending:
                self._refresh_goal()
                self._launch_replan_thread()
            # Pick up path immediately if the planner finished synchronously
            self._swap_pending_path()
            if self._firetruck_path:
                self._truck_state = "driving"

        elif self._truck_state == "driving":
            self._advance_firetruck()

        elif self._truck_state == "suppressing":
            extinguished = self._check_proximity_extinguish()
            if extinguished:
                for cell in extinguished:
                    self._extinguish_connected(cell)
                self._clear_goal()
            elif self._target_fire_cell and self._fire_burned_out(self._target_fire_cell):
                print(f"[Engine] Fire {self._target_fire_cell} burned out — replanning")
                self._clear_goal()
            elif self.map.sim_time - self._suppress_start >= self.suppress_duration:
                self._finish_suppression()

        # 5. Wumpus moves at fixed intervals
        self._wumpus_tick_counter += 1
        if self._wumpus_tick_counter >= self.wumpus_move_interval:
            self._advance_wumpus()
            self._wumpus_tick_counter = 0

        # 6. Wumpus burns; replan immediately if new fires were lit
        if self._wumpus_act():
            self._replan_wumpus()

        # 7. Display (throttled)
        if self.viz and self._tick_counter % self.display_every_n_ticks == 0:
            with self._path_lock:
                display_path = list(self._firetruck_path) if self._firetruck_path else None
            self.viz.update(display_path, self._wumpus_path)

        # 8. Real-time pacing
        if self.tick_real_time > 0:
            time.sleep(self.tick_real_time)

    # =======================================================================
    # Goal selection
    # =======================================================================

    def _refresh_goal(self) -> None:
        """
        Pick the best fire target (or the closest roadmap node to the wumpus)
        and cache the full ranked candidate list for fallback replanning.
        """
        if self.map.active_fires:
            ranked = self._rank_fire_candidates()
            self._fire_candidates  = [cell for _, cell in ranked]
            self._target_fire_cell = self._fire_candidates[0] if ranked else None
            if self._target_fire_cell:
                cs  = self.map.cell_size
                r, c = self._target_fire_cell
                self.map.update_goal((r*cs + cs/2, c*cs + cs/2, 0.0))
        else:
            self._target_fire_cell = None
            self._fire_candidates  = []
            # Wumpus chase: target the permanent PRM node that is closest to
            # the wumpus position.  Using a roadmap node (rather than the raw
            # wumpus world-metre pose) ensures the goal is always reachable
            # via the graph — identical to how fire goals work.
            wp  = self.map.wumpus_pose
            cs  = self.map.cell_size
            # Convert wumpus world position to the grid cell it occupies
            wumpus_cell = (int(wp[0] / cs), int(wp[1] / cs))
            # Search for permanent roadmap nodes near the wumpus using the
            # same approach_radius used for fire goals; fall back to a wider
            # search radius if none are found close enough.
            goal_nodes = self.firetruck._fire_goal_nodes(
                wumpus_cell, radius=self.approach_radius
            )
            if not goal_nodes:
                goal_nodes = self.firetruck._fire_goal_nodes(
                    wumpus_cell, radius=self.approach_radius * 2
                )

            if goal_nodes:
                # _fire_goal_nodes returns indices from a ball query — order is
                # arbitrary, so find the one whose world position is closest to
                # the wumpus to minimise approach distance.
                wx, wy = float(wp[0]), float(wp[1])
                best_idx = min(
                    goal_nodes,
                    key=lambda i: math.hypot(
                        self.firetruck.nodes[i][0] - wx,
                        self.firetruck.nodes[i][1] - wy,
                    ),
                )
                gx, gy, gth = self.firetruck.nodes[best_idx]
                self.map.update_goal((gx, gy, gth))
            else:
                # Absolute fallback: no roadmap nodes anywhere near the wumpus,
                # drive to the raw world-metre pose
                print('Warning no avaliable spot near wumpus')
                self.map.update_goal((float(wp[0]), float(wp[1]), 0.0))

    def _rank_fire_candidates(self) -> List[Tuple[float, Tuple[int, int]]]:
        """
        Sort active fires: viable (enough burn time remaining) first, then
        fallback fires.  Both groups sorted by Euclidean distance ascending.
        O(F log F) where F = |active_fires|.
        """
        ft     = self.map.firetruck_pose
        tx, ty = float(ft[0]), float(ft[1])
        cs     = self.map.cell_size
        v_max  = self.firetruck.car.v_max
        budget = self.proximity_duration + self.extinguish_margin
        now = self.map.sim_time

        viable, fallback = [], []
        for cell in self.map.active_fires:
            cx   = cell[0] * cs + cs / 2.0
            cy   = cell[1] * cs + cs / 2.0
            dist = math.hypot(tx - cx, ty - cy)
            data       = self.map.obstacle_coordinate_dict.get(cell, {})
            burn_start = data.get("burn_time") if data else 0.0
            remaining = self.burn_lifetime - (now - burn_start)
            # 3. Triage: If time to get there + spray < fire life, it's viable
            if remaining >= (dist / v_max) + budget:
                viable.append((dist, cell))
            else:
                fallback.append((dist, cell))

        viable.sort(key=lambda t: t[0])
        fallback.sort(key=lambda t: t[0])
        return viable + fallback

    # =======================================================================
    # Background replanning
    # =======================================================================

    def _launch_replan_thread(self) -> None:
        """
        Flush the oldest stale temp-node batch, snapshot planning inputs,
        and start the background replan thread.

        Node lifetime
        -------------
        _pending_cleanup holds at most 2 batches (maxlen=2).  We flush the
        oldest only when there are already 2 entries, guaranteeing the
        *previous* goal's injected nodes remain in the graph while the truck
        is still traversing the path that was routed through them.
        """
        if self._replan_pending:
            return

        if len(self._pending_cleanup) >= 2:
            self._delete_temp_nodes(self._pending_cleanup.popleft())

        pose  = self.map.firetruck_pose
        start: State = (float(pose[0]), float(pose[1]), float(pose[2]))

        self._replan_pending = True
        self._pending_path   = None
        threading.Thread(
            target=self._bg_replan,
            args=(start, self._target_fire_cell,
                  list(self._fire_candidates),
                  self._normalize_goal(self.map.firetruck_goal)),
            daemon=True,
        ).start()

    def _bg_replan(self, start: State, target_cell, candidates, goal_state) -> None:
        """
        Background thread: run A* + Reeds-Shepp, deposit result atomically.

        Thread-safety
        -------------
        - Reads firetruck.nodes/graph during A* (read-only after build_tree).
        - Writes only _pending_path, _target_fire_cell (under _path_lock),
          and _pending_cleanup (under _path_lock).
        - Never writes map.sim_time, map.firetruck_pose, or fire state.

        Temp-node accounting
        --------------------
        plan_to_fire() / plan() inject a start node into the graph.
        We record the index range [_roadmap_size, len(nodes)) immediately
        after the call succeeds, before attempting the next candidate.
        This prevents index accumulation from multiple failed attempts.
        """
        new_path:     Optional[List[State]] = None
        temp_indices: List[int]             = []


        replan_start = time.perf_counter()
    
        new_path: Optional[List[State]] = None
        temp_indices: List[int] = []
        
        try:
            if target_cell is not None:
                for cell in candidates:
                    nodes_before = len(self.firetruck.nodes)
                    path = self.firetruck.plan_to_fire(
                        fire_cell=cell, start_state=start, radius=self.approach_radius
                    )
                    if path:
                        new_path     = path
                        # Capture only the nodes injected by this successful call
                        temp_indices = list(range(self.firetruck._roadmap_size,
                                                  len(self.firetruck.nodes)))
                        with self._path_lock:
                            self._target_fire_cell = cell
                            cs = self.map.cell_size
                            self.map.update_goal(
                                (cell[0]*cs + cs/2, cell[1]*cs + cs/2, 0.0)
                            )
                        if cell != candidates[0]:
                            # print(f"[BG] Fallback succeeded: planned to fire {cell}")
                            pass 
                        break
                    else:
                        # Inject failed — clean up any orphaned node from this attempt
                        # so indices stay tidy for the next candidate
                        if len(self.firetruck.nodes) > nodes_before:
                            orphan = list(range(nodes_before, len(self.firetruck.nodes)))
                            self._delete_temp_nodes(orphan)
                        print(f"[BG] plan_to_fire failed for {cell}")

                if new_path is None:
                    print(f"[BG] All {len(candidates)} candidates unreachable")

            elif goal_state is not None:
                path = self.firetruck.plan(goal_state=goal_state, start_state=start)
                if path:
                    new_path     = path
                    temp_indices = list(range(self.firetruck._roadmap_size,
                                             len(self.firetruck.nodes)))
                else:
                    print("[BG] Wumpus-chase plan failed")

        except Exception as e:
            print(f"[BG] Error: {type(e).__name__}: {e}")
        finally:
            # 2. Calculate how long THIS specific replan took
            duration = time.perf_counter() - replan_start
            
            # 3. "Roll" it into the total safely
            with self._timer_lock:
                self._total_truck_replan_time += duration

            with self._path_lock:
                self._pending_path = new_path
                if temp_indices:
                    self._pending_cleanup.append(temp_indices)
                self._replan_pending = False

    def _delete_temp_nodes(self, indices: List[int]) -> None:
        """
        Surgically remove a specific batch of temp nodes from the PRM.

        Permanent edges never point to temp nodes (enforced at inject time),
        so the only work is:
          1. Strip edges FROM permanent nodes TO the dead indices.
          2. Drop their graph entries.
          3. Remap the node list so surviving temp nodes keep valid indices.

        O(permanent_nodes + surviving_temp_edges)
        """
        if not indices:
            return
        dead = set(indices)
        rs   = self.firetruck._roadmap_size

        # 1. Strip permanent→dead edges
        for i in range(rs):
            edges = self.firetruck.graph.get(i)
            if edges:
                self.firetruck.graph[i] = [e for e in edges if e["to"] not in dead]

        # 2. Drop dead graph entries
        for idx in dead:
            self.firetruck.graph.pop(idx, None)

        # 3. Rebuild node list (permanent + surviving temps) with remapped indices
        remap     = {}
        new_nodes = []
        for old_idx, node in enumerate(self.firetruck.nodes):
            if old_idx < rs or old_idx not in dead:
                remap[old_idx] = len(new_nodes)
                new_nodes.append(node)

        self.firetruck.nodes = new_nodes
        self.firetruck.graph = {
            remap[i]: [{**e, "to": remap[e["to"]]}
                       for e in edges if e["to"] in remap]
            for i, edges in self.firetruck.graph.items()
            if i in remap
        }

    def _swap_pending_path(self) -> None:
        """Atomically move a background-thread result into active use."""
        with self._path_lock:
            if self._pending_path is not None:
                self._firetruck_path = self._pending_path
                self._pending_path   = None

    # Synchronous replan — used by tests and non-threaded contexts
    def _replan_firetruck(self) -> None:
        pose  = self.map.firetruck_pose
        start: State = (float(pose[0]), float(pose[1]), float(pose[2]))
        if self._target_fire_cell is not None:
            for cell in (self._fire_candidates or [self._target_fire_cell]):
                path = self.firetruck.plan_to_fire(
                    fire_cell=cell, start_state=start, radius=self.approach_radius
                )
                if path:
                    self._firetruck_path   = path
                    self._target_fire_cell = cell
                    cs = self.map.cell_size
                    self.map.update_goal((cell[0]*cs + cs/2, cell[1]*cs + cs/2, 0.0))
                    return
                print(f"[Engine] plan_to_fire failed for {cell}")
            print("[Engine] All candidates unreachable (sync replan)")
        else:
            goal = self._normalize_goal(self.map.firetruck_goal)
            if goal:
                path = self.firetruck.plan(goal_state=goal, start_state=start)
                if path:
                    self._firetruck_path = path
                else:
                    print(f"[Engine] Wumpus-chase plan failed at t={self.map.sim_time:.1f}s")

    def _replan_wumpus(self) -> None:
        start_t = time.perf_counter()
        path = self.wumpus.plan()
        duration = time.perf_counter() - start_t
        with self._timer_lock:
            self._total_wumpus_replan_time += duration
        if path is not None:
            self._wumpus_path = path

    # =======================================================================
    # Agent advancement
    # =======================================================================

    def _advance_firetruck(self) -> None:
        with self._path_lock:
            if not self._firetruck_path or len(self._firetruck_path) < 2:
                # Path exhausted — commit to suppression if targeting a fire,
                # else go idle so a new goal can be selected next tick
                if self._target_fire_cell:
                    self._suppress_start = self.map.sim_time
                    self._truck_state    = "suppressing"
                else:
                    self._truck_state = "idle"
                return
            self._firetruck_path.pop(0)
            next_pose = self._firetruck_path[0]

        self.map.firetruck_pose = next_pose

        if self._target_fire_cell:
            cs   = self.map.cell_size
            r, c = self._target_fire_cell
            if math.hypot(next_pose[0] - (r*cs + cs/2),
                          next_pose[1] - (c*cs + cs/2)) <= self.approach_radius:
                print(f"[Engine] Arrived at fire {self._target_fire_cell} "
                      f"at t={self.map.sim_time:.1f}s — suppressing")
                self._suppress_start = self.map.sim_time
                self._truck_state    = "suppressing"
                with self._path_lock:
                    self._firetruck_path = [next_pose]
        

    def _advance_wumpus(self) -> None:
        if self._wumpus_path and len(self._wumpus_path) >= 2:
            self._wumpus_path.pop(0)
            r, c = self._wumpus_path[0]
            cs = self.map.cell_size
            self.map.wumpus_pose = (r*cs + cs/2, c*cs + cs/2)
        else:
            # PATH EXHAUSTED: Replan even if no new fire was lit
            self._replan_wumpus()

    # =======================================================================
    # Fire suppression
    # =======================================================================

    def _check_proximity_extinguish(self) -> Set[Tuple[int, int]]:
        """
        Track burning cells within proximity_radius.  Extinguish any that
        have been in range >= proximity_duration seconds.
        Iterates active_fires only — O(F) not O(all obstacles).
        """
        ft     = self.map.firetruck_pose
        tx, ty = float(ft[0]), float(ft[1])
        cs, now = self.map.cell_size, self.map.sim_time
        extinguished: Set[Tuple[int, int]] = set()

        in_range: set = {
            cell for cell in self.map.active_fires
            if math.hypot(tx - (cell[0]*cs + cs/2),
                          ty - (cell[1]*cs + cs/2)) <= self.proximity_radius
        }

        # Drop cells that left range
        for cell in list(self._proximity_timers):
            if cell not in in_range:
                del self._proximity_timers[cell]

        # Start timers for newly in-range cells
        for cell in in_range:
            self._proximity_timers.setdefault(cell, now)

        # Extinguish cells that have been in range long enough
        for cell, start_t in list(self._proximity_timers.items()):
            if now - start_t >= self.proximity_duration:
                data = self.map.obstacle_coordinate_dict.get(cell)
                if data and data["status"] == Status.BURNING:
                    self.map.set_status_on_obstacles([cell], Status.EXTINGUISHED)
                    print(f"[Engine] Extinguish {cell} after {now-start_t:.1f}s "
                          f"at t={now:.1f}s")
                    extinguished.add(cell)
                del self._proximity_timers[cell]

        return extinguished

    def _extinguish_connected(self, origin: Tuple[int, int]) -> None:
        """BFS flood-fill extinguish connected BURNING cells up to flood_fill_radius steps."""
        visited: Set[Tuple[int, int]] = {origin}
        queue = deque([(origin, 0)])
        while queue:
            (r, c), depth = queue.popleft()
            if depth >= self.flood_fill_radius:
                continue
            for nb in ((r, c+1), (r, c-1), (r+1, c), (r-1, c)):
                if nb in visited:
                    continue
                visited.add(nb)
                data = self.map.obstacle_coordinate_dict.get(nb)
                if data and data["status"] == Status.BURNING:
                    self.map.set_status_on_obstacles([nb], Status.EXTINGUISHED)
                    queue.append((nb, depth + 1))

    def _finish_suppression(self) -> None:
        """Extinguish 3×3 neighbourhood, flood-fill from each cell, then go idle."""
        pose = self.map.firetruck_pose
        cs   = self.map.cell_size
        cx, cy = int(pose[0] / cs), int(pose[1] / cs)
        extinguished = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                cell = (cx+dr, cy+dc)
                data = self.map.obstacle_coordinate_dict.get(cell)
                if data and data["status"] == Status.BURNING:
                    self.map.set_status_on_obstacles([cell], Status.EXTINGUISHED)
                    extinguished.append(cell)
        if extinguished:
            print(f"[Engine] Suppressed {extinguished} at t={self.map.sim_time:.1f}s")
            for cell in extinguished:
                self._extinguish_connected(cell)
        self._clear_goal()

    def _fire_burned_out(self, cell: Tuple[int, int]) -> bool:
        data = self.map.obstacle_coordinate_dict.get(cell)
        return data is None or data["status"] != Status.BURNING

    def _clear_goal(self) -> None:
        """Reset truck to idle and clear all goal/path state."""
        self.map.firetruck_goal = None
        self._firetruck_path    = None
        self._proximity_timers  = {}
        self._target_fire_cell  = None
        self._truck_state       = "idle"

    def _wumpus_act(self) -> bool:
        """Burn adjacent obstacles; return True if new fires were lit."""
        before = len(self.map.active_fires)
        try:
            self.wumpus.burn()
        except Exception as e:
            print(f"[Engine] Wumpus burn() error: {e}")
            return False
        return len(self.map.active_fires) > before

    # =======================================================================
    # Utilities
    # =======================================================================

    @staticmethod
    def _normalize_goal(goal) -> Optional[State]:
        """Coerce any goal representation to (x, y, theta_deg) or None."""
        if not goal or goal == "ERROR CANT GO HERE":
            return None
        return (float(goal[0]), float(goal[1]),
                float(goal[2]) if len(goal) > 2 else 0.0)

    # =======================================================================
    # Shutdown
    # =======================================================================

    def _shutdown(self) -> None:
        counts = {s.name: 0 for s in Status}
        for data in self.map.obstacle_coordinate_dict.values():
            counts[data["status"].name] += 1
        labels = {
            _END_TIME:   "Time limit (5 min)",
            _END_WUMPUS: "Wumpus caught",
            _END_MAP:    "Map complete",
            None:        "Unknown",
        }
        print(f"\n[Engine] ── Final Stats ─────────────────────────")
        print(f"  End    : {labels.get(self._end_reason, self._end_reason)}")
        print(f"  Time   : {self.map.sim_time:.1f}s")
        print(f"  Rolling Truck Replan Time  : {self._total_truck_replan_time:.4f}s")
        print(f"  Rolling Wumpus Replan Time : {self._total_wumpus_replan_time:.4f}s")

        for k, v in counts.items():
            print(f"  {k:<13}: {v}")
        print("─────────────────────────────────────────────────\n")
        if self.viz:
            self.viz.close()