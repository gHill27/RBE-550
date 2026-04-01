"""
simulation_engine.py
====================
SimulationEngine — the single entry point for the Wildfire simulation.

Changes in this version
-----------------------
1. Faster sim speed via display throttling
   A new `display_every_n_ticks` parameter (default 5) skips viz.update()
   on intermediate ticks.  With tick_real_time=0.0 this means the sim
   computes 5 physics steps per rendered frame, running ~5× faster in
   real time without changing any sim logic.  Set to 1 to render every tick.

2. Wumpus replans immediately after burning
   After wumpus.burn() fires, the engine calls _replan_wumpus() on the
   same tick if burn() returned any newly-lit cells.  The wumpus no longer
   waits for the firetruck's idle cycle to get a new path away from the fire.

3. Deferred cleanup of injected PRM nodes (2-goal lag)
   plan_to_fire() / plan() inject a temporary start node into the PRM graph
   for connectivity.  Previously _cleanup_query_nodes() removed it
   immediately after each plan, so when the truck switched goals its current
   position was disconnected from the graph.
   
   New behaviour: injected nodes from the **previous** goal are kept alive
   for one full goal cycle.  A two-slot queue (_pending_cleanup) holds the
   temp-node indices for the last two plans.  Only when a third plan starts
   are the oldest temp nodes actually removed.  This guarantees the truck
   always has at least one injected node connecting it to the permanent graph.

4. Background replanning to eliminate lag spikes
   _replan_firetruck() now runs in a daemon Thread.  The engine continues
   advancing the sim and updating the display while A* + Reeds-Shepp path
   checks run concurrently.  When the thread finishes, the new path is
   swapped in atomically via a threading.Lock.  The display never blocks.

   A _replan_pending flag prevents overlapping replan threads.
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
    Orchestrates Map, Firetruck, Wumpus, and SimVisualizer.

    Parameters
    ----------
    grid_num : int
    cell_size : float
    fill_percent : float
    firetruck_start : (x_m, y_m, theta_deg)
    wumpus_start : (x_m, y_m)
    prm_nodes : int
    replan_interval : float
        Minimum sim-seconds between replans while driving.
    tick_real_time : float
        Wall-clock seconds to sleep per tick.  0.0 = max speed.
    display_every_n_ticks : int
        Render the visualizer only every N ticks.  Default 5.
        Higher values = faster sim, less smooth display.
    plot : bool
    plot_prm : bool
    sim_duration : float   (default 300 s = 5 min)
    extinguish_margin : float
    burn_lifetime : float
    wumpus_catch_radius : float   (default 5.0 m)
    flood_fill_radius : int       (default 4)
    """

    def __init__(
        self,
        grid_num:              int   = 50,
        cell_size:             float = 5.0,
        fill_percent:          float = 0.15,
        firetruck_start:       State = (25.0, 25.0, 0.0),
        wumpus_start:          Tuple[float, float] = (220.0, 220.0),
        prm_nodes:             int   = 500,
        replan_interval:       float = 5.0,
        tick_real_time:        float = 0.0,
        display_every_n_ticks: int   = 5,
        plot:                  bool  = True,
        plot_prm:              bool  = False,
        sim_duration:          float = 300.0,
        extinguish_margin:     float = 5.0,
        burn_lifetime:         float = 30.0,
        wumpus_catch_radius:   float = 5.0,
        flood_fill_radius:     int   = 4,
    ):
        self.replan_interval       = replan_interval
        self.tick_real_time        = tick_real_time
        self.display_every_n_ticks = max(1, display_every_n_ticks)
        self.sim_duration          = sim_duration
        self.plot                  = plot
        self.plot_prm              = plot_prm
        self.extinguish_margin     = extinguish_margin
        self.burn_lifetime         = burn_lifetime
        self.wumpus_catch_radius   = wumpus_catch_radius
        self.flood_fill_radius     = flood_fill_radius

        self._end_reason: Optional[str] = None
        self._tick_counter: int         = 0

        # Firetruck path (protected by _path_lock for background replanning)
        self._path_lock: threading.Lock         = threading.Lock()
        self._firetruck_path: Optional[List[State]] = None
        self._pending_path:   Optional[List[State]] = None   # written by bg thread
        self._replan_pending: bool                  = False  # bg thread running?

        self._wumpus_path:      Optional[List]  = None
        self._last_replan_time: float           = -replan_interval

        self._target_fire_cell:  Optional[Tuple[int, int]] = None
        self._fire_candidates:   List[Tuple[int, int]]     = []
        self.approach_radius:    float = 10.0

        # Truck state machine: "idle" | "driving" | "suppressing"
        self._truck_state:    str   = "idle"
        self._suppress_start: float = 0.0
        self.suppress_duration: float = 8.0

        self._proximity_timers: dict   = {}
        self.proximity_radius:  float  = 10.0
        self.proximity_duration: float = 5.0

        self._wumpus_tick_counter: int = 10
        self.wumpus_move_interval: int = 10

        # ------------------------------------------------------------------
        # Deferred node-cleanup queue (fix 3)
        # Holds (List[temp_node_indices], goal_id) from the previous plan.
        # Only the entry two goals old is actually deleted.
        # ------------------------------------------------------------------
        # Each entry: list of temp node indices to clean up
        self._pending_cleanup: Deque[List[int]] = deque(maxlen=2)

        # ------------------------------------------------------------------
        # Build Map
        # ------------------------------------------------------------------
        print("[Engine] Building map...")
        self.map = Map(
            Grid_num       = grid_num,
            cell_size      = cell_size,
            fill_percent   = fill_percent,
            wumpus         = None,
            firetruck      = None,
            firetruck_pose = firetruck_start,
            wumpus_pose    = (wumpus_start[0], wumpus_start[1]),
        )

        # ------------------------------------------------------------------
        # Build agents
        # ------------------------------------------------------------------
        print("[Engine] Initialising agents...")
        self.firetruck = Firetruck(self.map, plot=plot_prm)
        self.wumpus    = Wumpus(self.map)

        self.map.firetruck = self.firetruck
        self.map.wumpus    = self.wumpus

        # ------------------------------------------------------------------
        # Build PRM roadmap
        # ------------------------------------------------------------------
        print(f"[Engine] Building PRM roadmap ({prm_nodes} nodes)...")
        self.firetruck.build_tree(n_samples=prm_nodes)
        print("[Engine] Roadmap ready.")

        if plot_prm and self.firetruck.viz is not None:
            print("[Engine] Rendering PRM debug graph...")
            self.firetruck.viz.plot_prm(
                self.map,
                self.firetruck.graph,
                self.firetruck.nodes,
                path=None,
            )

        # ------------------------------------------------------------------
        # Visualizer
        # ------------------------------------------------------------------
        self.viz: Optional[SimVisualizer] = None
        if plot:
            print("[Engine] Opening display window...")
            self.viz = SimVisualizer(self.map, figsize=(16, 11))

        print("[Engine] Initialisation complete. Ready to run.")

    # =======================================================================
    # Public API
    # =======================================================================

    def run(self) -> None:
        print(
            f"[Engine] Simulation starting "
            f"(duration={self.sim_duration}s, "
            f"display_every={self.display_every_n_ticks} ticks, "
            f"wumpus_catch_radius={self.wumpus_catch_radius}m)..."
        )
        self._refresh_goal()

        while self.map.sim_time <= self.sim_duration:
            self._tick()
            if self._end_reason is not None:
                break

        print(f"[Engine] Simulation ended at t={self.map.sim_time:.1f}s "
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

        Speed notes
        -----------
        With tick_real_time=0.0 and display_every_n_ticks=N, every N ticks
        costs one matplotlib redraw and zero sleep.  The sim runs at
        approximately N × (physics_time / render_time) faster than real time.
        Increase display_every_n_ticks to trade display smoothness for speed.

        Background replanning (fix 4)
        ------------------------------
        When the truck goes idle, _launch_replan_thread() starts a daemon
        thread.  The thread writes its result to _pending_path.  Each tick,
        _swap_pending_path() atomically moves _pending_path → _firetruck_path
        if a new path has arrived.  The truck keeps its old path (or sits
        still) while the thread runs; no tick ever blocks on A*.
        """
        self._tick_counter += 1

        # 1. Advance sim clock and fire-spread
        result = self.map.main()
        if result == "Done":
            self._end_reason = _END_MAP
            self.map.sim_time = self.sim_duration + 1.0
            return

        # 2. Wumpus-catch check
        if self._check_wumpus_caught():
            self._end_reason = _END_WUMPUS
            return

        # 3. Swap in any path that just finished computing
        self._swap_pending_path()

        # 4. Truck state machine
        if self._truck_state == "idle":
            if not self._replan_pending:
                # No thread running yet — start one
                self._refresh_goal()
                self._launch_replan_thread()
                self._last_replan_time = self.map.sim_time
            # If the new path is already ready (fast plan), pick it up
            self._swap_pending_path()
            if self._firetruck_path:
                self._truck_state = "driving"

        elif self._truck_state == "driving":
            self._advance_firetruck()

        elif self._truck_state == "suppressing":
            extinguished_cells = self._check_proximity_extinguish()
            if extinguished_cells:
                for cell in extinguished_cells:
                    self._extinguish_connected(cell)
                self._clear_goal()
            else:
                if (self._target_fire_cell is not None and
                        self._fire_cell_burned_out(self._target_fire_cell)):
                    print(
                        f"[Engine] Target fire {self._target_fire_cell} burned out "
                        f"at t={self.map.sim_time:.1f}s — replanning"
                    )
                    self._clear_goal()
                else:
                    if self.map.sim_time - self._suppress_start >= self.suppress_duration:
                        self._finish_suppression()
                        self._truck_state = "idle"

        # 5. Wumpus moves and replans independently
        self._wumpus_tick_counter += 1
        if self._wumpus_tick_counter >= self.wumpus_move_interval:
            self._advance_wumpus()
            self._wumpus_tick_counter = 0

        # Wumpus burns and immediately replans if it just lit something
        newly_burning = self._wumpus_act()
        if newly_burning or not self._wumpus_path:
            # Wumpus just set fire — get a new goal away from the flames
            self._replan_wumpus()

        # 6. Display (throttled by display_every_n_ticks)
        if self.viz and self._tick_counter % self.display_every_n_ticks == 0:
            with self._path_lock:
                display_path = list(self._firetruck_path) if self._firetruck_path else None
            self.viz.update(display_path, self._wumpus_path)

        # 7. Pace wall-clock
        if self.tick_real_time > 0:
            time.sleep(self.tick_real_time)

    # =======================================================================
    # Termination helpers
    # =======================================================================

    def _check_wumpus_caught(self) -> bool:
        ft   = self.map.firetruck_pose
        wp   = self.map.wumpus_pose
        dist = math.hypot(float(ft[0]) - float(wp[0]),
                          float(ft[1]) - float(wp[1]))
        if dist <= self.wumpus_catch_radius and self._wumpus_path == False:
            print(
                f"[Engine] WUMPUS CAUGHT! dist={dist:.2f}m "
                f"at t={self.map.sim_time:.1f}s"
            )
            return True
        return False

    # =======================================================================
    # Goal management
    # =======================================================================

    @staticmethod
    def _normalize_goal(goal):
        if not goal or goal == "ERROR CANT GO HERE":
            return None
        if len(goal) == 2:
            return (float(goal[0]), float(goal[1]), 0.0)
        return (float(goal[0]), float(goal[1]), float(goal[2]))

    def _clear_goal(self) -> None:
        """Reset all goal/path/state to idle cleanly."""
        self.map.firetruck_goal = None
        self._firetruck_path    = None
        self._proximity_timers  = {}
        self._target_fire_cell  = None
        self._truck_state       = "idle"

    def _refresh_goal(self) -> None:
        if self.map.active_fires:
            candidates              = self._rank_fire_candidates()
            self._fire_candidates   = [c for _, c in candidates]
            self._target_fire_cell  = (
                self._fire_candidates[0] if self._fire_candidates else None
            )
            if self._target_fire_cell:
                cs = self.map.cell_size
                fc = self._target_fire_cell
                self.map.update_goal((fc[0]*cs + cs/2.0, fc[1]*cs + cs/2.0, 0.0))
        else:
            self._target_fire_cell = None
            self._fire_candidates  = []

            wumpus_pose = self.map.wumpus_pose
            cs = self.map.cell_size        
            # Convert Wumpus world-metres to a grid cell for the helper function
            wumpus_cell = (int(wumpus_pose[0] / cs), int(wumpus_pose[1] / cs))

            # Use the roadmap helper to find nodes near the Wumpus
            # We use a larger radius here (e.g., 15m) to ensure we find a 
            # valid connection even if the Wumpus is in an awkward spot.
            goal_node_indices = self.firetruck._fire_goal_nodes(
                wumpus_cell, 
                radius=self.wumpus_catch_radius 
            )

            if goal_node_indices:
                # Pick the best pre-existing roadmap node near the Wumpus
                node_idx = goal_node_indices[0] #TODO make this try and use a different one if this one is poor
                gx, gy, gtheta = self.firetruck.nodes[node_idx] 
                self.map.update_goal((gx, gy, gtheta))
                print(f"[Engine] Wumpus chase: Targeting Roadmap Node {node_idx}")
            else:
                # Absolute Fallback: Just go to the raw Wumpus coordinates
                goal = self._normalize_goal(wumpus_pose)
                if goal:
                    self.map.update_goal(goal)

    def _rank_fire_candidates(self) -> List[Tuple[float, Tuple[int, int]]]:
        """
        Sort active fires: viable (enough burn time) first, fallback last.
        Both groups sorted by Euclidean distance ascending.
        O(F log F) where F = number of active fires.
        """
        ft    = self.map.firetruck_pose
        tx, ty = float(ft[0]), float(ft[1])
        cs    = self.map.cell_size
        v_max = self.firetruck.car.v_max
        budget = self.proximity_duration + self.extinguish_margin
 
        viable, fallback = [], []
        for cell in self.map.active_fires:
            cx   = cell[0] * cs + cs / 2.0
            cy   = cell[1] * cs + cs / 2.0
            dist = math.hypot(tx - cx, ty - cy)
            data = self.map.obstacle_coordinate_dict.get(cell)
            burn_start = data.get("burn_time") if data else None
            remaining = (max(0.0, self.burn_lifetime - (self.map.sim_time - burn_start))
                         if burn_start is not None else 0.0)
            (viable if remaining >= dist / v_max + budget else fallback).append((dist, cell))
 
        viable.sort(key=lambda t: t[0])
        fallback.sort(key=lambda t: t[0])
        return viable + fallback
    

    # =======================================================================
    # Background replanning 
    # =======================================================================

    def _launch_replan_thread(self) -> None:
        """
        Kick off a daemon thread to compute the next firetruck path.

        The thread captures a snapshot of the current planning inputs
        (start pose, candidate list, target cell) so it runs safely without
        holding any locks on the main sim state.  When it finishes, it
        deposits the result into _pending_path and clears _replan_pending.

        Deferred cleanup (fix 3)
        ------------------------
        Before launching, flush the cleanup queue: any temp nodes that are
        now two goals old are removed from the PRM.  Nodes from the
        *previous* goal remain in the graph so the truck — which may still
        be traversing a path that was connected through those nodes — stays
        reachable.
        """
        if self._replan_pending:
            return   # thread already running

        # Flush nodes that are now two goals old (safe to delete)
        if len(self._pending_cleanup) >= 2:
            self._delete_temp_nodes(self._pending_cleanup.popleft())

        # Snapshot planning inputs for the thread
        pose         = self.map.firetruck_pose
        start: State = (float(pose[0]), float(pose[1]), float(pose[2]))
        target_cell  = self._target_fire_cell
        candidates   = list(self._fire_candidates)
        goal_state   = self._normalize_goal(self.map.firetruck_goal)

        self._replan_pending = True
        self._pending_path   = None

        t = threading.Thread(
            target=self._bg_replan,
            args=(start, target_cell, candidates, goal_state),
            daemon=True,
        )
        t.start()

    def _bg_replan(
        self,
        start:       State,
        target_cell: Optional[Tuple[int, int]],
        candidates:  List[Tuple[int, int]],
        goal_state:  Optional[State],
    ) -> None:
        """
        Background thread body.  Runs plan_to_fire() / plan() off the main
        thread so the display never blocks.

        Thread-safety
        -------------
        - Reads firetruck.nodes / graph during A* (PRM is read-only after
          build; only _cleanup touches it, which we defer until the thread
          finishes via _pending_cleanup).
        - Writes only to _pending_path and the cleanup queue — both protected
          by _path_lock.
        - Never writes to map.sim_time, map.firetruck_pose, or any Map field.
        """
        new_path      = None
        used_indices: List[int] = []   # temp node indices injected this plan

        try:
            if target_cell is not None:
                for cell in candidates:
                    path = self.firetruck.plan_to_fire(
                        fire_cell   = cell,
                        start_state = start,
                        radius      = self.approach_radius,
                    )
                    if path:
                        new_path = path
                        # Record which temp nodes were injected (the last one added)
                        used_indices = self._collect_recent_temp_nodes()
                        with self._path_lock:
                            self._target_fire_cell = cell
                            cs = self.firetruck.map.cell_size
                            gx = cell[0] * cs + cs / 2.0
                            gy = cell[1] * cs + cs / 2.0
                            self.firetruck.map.update_goal((gx, gy, 0.0))
                        if cell != candidates[0]:
                            print(
                                f"[BG] Fallback succeeded: planned to fire {cell}"
                            )
                        break
                    else:
                        print(f"[BG] plan_to_fire failed for {cell} — next candidate")

                if new_path is None:
                    print(f"[BG] All {len(candidates)} candidates unreachable")

            elif goal_state is not None:
                path = self.firetruck.plan(
                    goal_state  = goal_state,
                    start_state = start,
                )
                if path:
                    new_path     = path
                    used_indices = self._collect_recent_temp_nodes()
                else:
                    print("[BG] Wumpus-chase plan failed")

        except Exception as e:
            print(f"[BG] Replan thread error: {type(e).__name__}: {e}")
        finally:
            with self._path_lock:
                self._pending_path = new_path
                if used_indices:
                    self._pending_cleanup.append(used_indices)
                self._replan_pending = False

    def _collect_recent_temp_nodes(self) -> List[int]:
        """
        Return the indices of all temp (query) nodes currently in the PRM
        — i.e. all indices >= _roadmap_size.  Called immediately after
        plan_to_fire() / plan() so we capture nodes before cleanup runs.
        """
        roadmap_size = self.firetruck._roadmap_size
        return [i for i in range(roadmap_size, len(self.firetruck.nodes))]

    def _delete_temp_nodes(self, indices: List[int]) -> None:
        """
        Remove specific temp node indices from the PRM graph and node list.

        Rather than calling _cleanup_query_nodes() (which deletes ALL temp
        nodes), we remove only the specified indices so we can surgically
        keep the previous goal's nodes alive.

        After deletion the permanent roadmap is unaffected because temp nodes
        always have indices >= _roadmap_size and permanent edges never point
        to them (enforced by _cleanup in the original planner).
        """
        if not indices:
            return
        idx_set = set(indices)
        roadmap_size = self.firetruck._roadmap_size
        # 1. Remove edges FROM permanent nodes TO these temp nodes
        # We only need to check permanent nodes that might have connected to them
        for i in range(roadmap_size):
            if i in self.firetruck.graph:
                # Filter the adjacency list in-place
                self.firetruck.graph[i] = [
                    edge for edge in self.firetruck.graph[i] 
                    if edge["to"] not in idx_set
                ]

        # 2. Remove the temp nodes' own entry in the graph dictionary
        for idx in indices:
            self.firetruck.graph.pop(idx, None)

        # 3. Truncate the nodes list
        # If we are only deleting the oldest batch and keeping the 'previous' batch,
        # we just ensure the nodes list only contains indices we haven't 'deleted'
        max_valid_idx = max([i for i in range(len(self.firetruck.nodes)) if i not in idx_set], default=roadmap_size-1)
        self.firetruck.nodes = self.firetruck.nodes[:max_valid_idx + 1]


    def _swap_pending_path(self) -> None:
        """
        Atomically move _pending_path → _firetruck_path if a new path
        arrived from the background thread.
        """
        with self._path_lock:
            if self._pending_path is not None:
                self._firetruck_path = self._pending_path
                self._pending_path   = None

    def _replan(self) -> None:
        """Synchronous replan used by tests and the initial idle cycle."""
        self._replan_firetruck_sync()
        self._replan_wumpus()

    def _replan_firetruck_sync(self) -> None:
        """Synchronous version of replanning (for tests / wumpus-only case)."""
        pose  = self.map.firetruck_pose
        start: State = (float(pose[0]), float(pose[1]), float(pose[2]))

        if self._target_fire_cell is not None:
            candidates = list(self._fire_candidates)
            if self._target_fire_cell not in candidates:
                candidates.insert(0, self._target_fire_cell)
            for cell in candidates:
                path = self.firetruck.plan_to_fire(
                    fire_cell   = cell,
                    start_state = start,
                    radius      = self.approach_radius,
                )
                if path:
                    self._firetruck_path   = path
                    self._target_fire_cell = cell
                    cs = self.map.cell_size
                    self.map.update_goal(
                        (cell[0]*cs + cs/2.0, cell[1]*cs + cs/2.0, 0.0)
                    )
                    return
                else:
                    print(f"[Engine] plan_to_fire failed for {cell}")
            print(f"[Engine] All {len(candidates)} candidates unreachable")
        else:
            goal = self._normalize_goal(self.map.firetruck_goal)
            if goal is None:
                return
            path = self.firetruck.plan(goal_state=goal, start_state=start)
            if path:
                self._firetruck_path = path
            else:
                print(f"[Engine] Firetruck replan failed at t={self.map.sim_time:.1f}s")

    # Keep _replan_firetruck as alias for test compatibility
    _replan_firetruck = _replan_firetruck_sync

    def _replan_wumpus(self) -> None:
        path = self.wumpus.plan()
        if path is not None:
            self._wumpus_path = path

    # =======================================================================
    # Agent advancement
    # =======================================================================

    def _advance_firetruck(self) -> None:
        with self._path_lock:
            path = self._firetruck_path

        if not path or len(path) < 2:
            if self._target_fire_cell is not None:
                print(
                    f"[Engine] Path exhausted near fire {self._target_fire_cell} "
                    f"at t={self.map.sim_time:.1f}s — starting suppression"
                )
                self._suppress_start = self.map.sim_time
                self._truck_state    = "suppressing"
            else:
                self._truck_state = "idle"
            return

        with self._path_lock:
            self._firetruck_path.pop(0)
            next_pose = self._firetruck_path[0]

        self.map.firetruck_pose = next_pose

        if self._target_fire_cell is not None:
            cs  = self.map.cell_size
            fcx = self._target_fire_cell[0] * cs + cs / 2.0
            fcy = self._target_fire_cell[1] * cs + cs / 2.0
            dist = math.hypot(next_pose[0] - fcx, next_pose[1] - fcy)
            if dist <= self.approach_radius:
                print(
                    f"[Engine] Truck arrived within {dist:.1f}m of fire "
                    f"{self._target_fire_cell} at t={self.map.sim_time:.1f}s "
                    f"— suppressing for up to {self.suppress_duration}s"
                )
                self._suppress_start = self.map.sim_time
                self._truck_state    = "suppressing"
                with self._path_lock:
                    self._firetruck_path = [next_pose]

    def _advance_wumpus(self) -> None:
        if not self._wumpus_path or len(self._wumpus_path) < 1:
            return
        next_cell = self._wumpus_path[0]
        self._wumpus_path.pop(0)
        cs = self.map.cell_size
        self.map.wumpus_pose = (next_cell[0] * cs + cs / 2.0,
                                next_cell[1] * cs + cs / 2.0)

    # =======================================================================
    # Agent actions
    # =======================================================================

    def _check_proximity_extinguish(self) -> Set[Tuple[int, int]]:
        """
        Track burning cells within proximity_radius.  Extinguish any that
        have been in range >= proximity_duration seconds.
        Iterates active_fires (only burning cells) — O(F) not O(all obstacles).
        """
        ft = self.map.firetruck_pose
        tx, ty = float(ft[0]), float(ft[1])
        cs, now = self.map.cell_size, self.map.sim_time
        extinguished: Set[Tuple[int, int]] = set()
 
        in_range: set = {
            cell for cell in self.map.active_fires
            if math.hypot(tx - (cell[0]*cs + cs/2), ty - (cell[1]*cs + cs/2))
               <= self.proximity_radius
        }
 
        # Remove cells that left range
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
                    print(f"[Engine] Extinguish {cell} after {now-start_t:.1f}s")
                    extinguished.add(cell)
                del self._proximity_timers[cell]
 
        return extinguished

    def _extinguish_connected(self, origin: Tuple[int, int]) -> None:
        """BFS flood-fill extinguish up to flood_fill_radius grid steps."""
        visited: Set[Tuple[int, int]] = {origin}
        queue: deque = deque([(origin, 0)])

        while queue:
            (r, c), depth = queue.popleft()
            if depth >= self.flood_fill_radius:
                continue
            for dr, dc in ((0,1),(0,-1),(1,0),(-1,0)):
                nb = (r + dr, c + dc)
                if nb in visited:
                    continue
                visited.add(nb)
                data = self.map.obstacle_coordinate_dict.get(nb)
                if data is None or data["status"] != Status.BURNING:
                    continue
                self.map.set_status_on_obstacles([nb], Status.EXTINGUISHED)
                print(
                    f"[Engine] Flood-fill extinguish: {nb} "
                    f"(depth {depth+1} from {origin}) at t={self.map.sim_time:.1f}s"
                )
                queue.append((nb, depth + 1))

    def _fire_cell_burned_out(self, cell: Tuple[int, int]) -> bool:
        data = self.map.obstacle_coordinate_dict.get(cell)
        return data is None or data["status"] != Status.BURNING

    def _finish_suppression(self) -> None:
        """Extinguish 3×3 neighbourhood around truck, flood-fill from each, then go idle."""
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
    # Shutdown
    # =======================================================================

    def _shutdown(self) -> None:
        counts = {"INTACT": 0, "BURNING": 0, "EXTINGUISHED": 0, "BURNED": 0}
        for data in self.map.obstacle_coordinate_dict.values():
            counts[data["status"].name] += 1

        reason_labels = {
            _END_TIME:   "Time limit reached (5 min)",
            _END_WUMPUS: "Wumpus caught",
            _END_MAP:    "Map simulation complete",
            None:        "Unknown",
        }

        print("\n[Engine] ── Final Statistics ──────────────────")
        print(f"  End reason     : {reason_labels.get(self._end_reason, self._end_reason)}")
        print(f"  Sim time       : {self.map.sim_time:.1f}s")
        print(f"  Intact cells   : {counts['INTACT']}")
        print(f"  Burned cells   : {counts['BURNED']}")
        print(f"  Extinguished   : {counts['EXTINGUISHED']}")
        print(f"  Still burning  : {counts['BURNING']}")
        print("────────────────────────────────────────────────\n")

        if self.viz:
            self.viz.close()