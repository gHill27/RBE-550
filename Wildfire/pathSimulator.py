"""
simulation_engine.py
====================
SimulationEngine — the single entry point for the Wildfire simulation.

Changes in this version
-----------------------
1. Fallback goal on plan failure
   _replan_firetruck() now walks the triage-sorted candidate list and
   tries each fire cell in turn until plan_to_fire() succeeds, rather
   than giving up after one failure.

2. Simulation ends at 5 min sim time OR wumpus catch (≤ 5 m)
   _tick() checks truck↔wumpus distance every step.  sim_duration
   defaults to 3600 s .  A "wumpus caught" result triggers an
   immediate clean shutdown.

3. Flood-fill extinguish within 4 tiles
   _extinguish_connected() performs a BFS from each newly-extinguished
   cell, extinguishing every connected BURNING obstacle within 4 grid
   steps.  Called from both _check_proximity_extinguish() and
   _finish_suppression().

4. PRM debug visualizer plots nodes + edges
   After build_tree() completes, if plot_prm=True, the engine calls
   firetruck.viz.plot_prm() with the full graph and node list so the
   roadmap is visible for debugging.

5. Gaussian sampling for tighter PRM paths
   _sample_points() now draws (x, y) from a Gaussian centred on the
   map midpoint with σ = world_size / 4.  Samples are clamped to the
   valid range [margin, limit-margin].  This concentrates nodes in the
   navigable interior and produces shorter, more consistent paths.
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import List, Optional, Set, Tuple

from Map_Generator import Map, Status
from firetruck import Firetruck
from wumpus import Wumpus
from pathVisualizer import SimVisualizer

State = Tuple[float, float, float]

# Simulation end reasons
_END_TIME    = "time_limit"
_END_WUMPUS  = "wumpus_caught"
_END_MAP     = "map_done"


class SimulationEngine:
    """
    Orchestrates Map, Firetruck, Wumpus, and SimVisualizer for one
    complete simulation run.

    Parameters
    ----------
    grid_num : int
        Number of grid cells per side.
    cell_size : float
        Metres per grid cell.
    fill_percent : float
        Fraction of grid cells occupied by obstacles (0.0–1.0).
    firetruck_start : tuple (x_m, y_m, theta_deg)
        Initial firetruck pose in world metres.
    wumpus_start : tuple (x_m, y_m)
        Initial wumpus position in world metres.
    prm_nodes : int
        Number of nodes to sample when building the PRM roadmap.
    replan_interval : float
        Minimum simulated seconds between replans while driving.
    tick_real_time : float
        Wall-clock seconds to sleep per tick.  0.0 = as fast as possible.
    plot : bool
        Whether to open the main simulation display window.
    plot_prm : bool
        Whether to show the PRM debug window (nodes + edges).
    sim_duration : float
        Hard time limit in simulated seconds.  Default 3600 s.
    extinguish_margin : float
        Extra seconds beyond proximity_duration required for a fire to be
        considered reachable.  Default 5.0 s.
    burn_lifetime : float
        Total seconds a cell burns before it self-extinguishes.  Default 30 s.
    wumpus_catch_radius : float
        Truck–wumpus distance (metres) that counts as a catch.  Default 5.0 m.
    flood_fill_radius : int
        BFS depth for connected-obstacle extinguishing after a cell is put out.
        Default 4 grid steps.
    """

    def __init__(
        self,
        grid_num:            int   = 50,
        cell_size:           float = 5.0,
        fill_percent:        float = 0.15,
        firetruck_start:     State = (25.0, 25.0, 0.0),
        wumpus_start:        Tuple[float, float] = (220.0, 220.0),
        prm_nodes:           int   = 500,
        replan_interval:     float = 5.0,
        tick_real_time:      float = 0.05,
        plot:                bool  = True,
        plot_prm:            bool  = True,
        sim_duration:        float = 3600.0,   
        extinguish_margin:   float = 5.0,
        burn_lifetime:       float = 30.0,
        wumpus_catch_radius: float = 5.0,
        flood_fill_radius:   int   = 4,
    ):
        self.replan_interval     = replan_interval
        self.tick_real_time      = tick_real_time
        self.sim_duration        = sim_duration
        self.plot                = plot
        self.plot_prm            = plot_prm
        self.extinguish_margin   = extinguish_margin
        self.burn_lifetime       = burn_lifetime
        self.wumpus_catch_radius = wumpus_catch_radius
        self.flood_fill_radius   = flood_fill_radius

        # End-reason recorded when the sim stops
        self._end_reason: Optional[str] = None

        # Internal state
        self._firetruck_path: Optional[List[State]] = None
        self._wumpus_path:    Optional[List]        = None
        self._last_replan_time: float               = -replan_interval
        self._last_goal:        Optional[State]     = None

        self._target_fire_cell: Optional[Tuple[int, int]] = None
        self.approach_radius: float = 10.0

        # Truck state machine: "idle" | "driving" | "suppressing"
        self._truck_state: str      = "idle"
        self._suppress_start: float = 0.0
        self.suppress_duration: float = 8.0

        self._proximity_timers: dict   = {}
        self.proximity_radius: float   = 10.0
        self.proximity_duration: float = 5.0

        self._wumpus_tick_counter: int = 0
        self.wumpus_move_interval: int = 10

        # ------------------------------------------------------------------
        # Step 1 — Build Map
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
        # Step 2 — Build agents
        # ------------------------------------------------------------------
        print("[Engine] Initialising agents...")
        self.firetruck = Firetruck(self.map, plot=plot_prm)
        self.wumpus    = Wumpus(self.map)

        # ------------------------------------------------------------------
        # Step 3 — Patch circular dependency
        # ------------------------------------------------------------------
        self.map.firetruck = self.firetruck
        self.map.wumpus    = self.wumpus

        # ------------------------------------------------------------------
        # Step 4 — Build PRM roadmap (Gaussian-sampled)
        # ------------------------------------------------------------------
        print(f"[Engine] Building PRM roadmap ({prm_nodes} nodes, Gaussian sampling)...")
        self.firetruck.build_tree(n_samples=prm_nodes)
        print("[Engine] Roadmap ready.")

        # ------------------------------------------------------------------
        # Step 4b — Show PRM debug visualizer (nodes + edges)
        # ------------------------------------------------------------------
        if plot_prm and self.firetruck.viz is not None:
            print("[Engine] Rendering PRM debug graph...")
            self.firetruck.viz.plot_prm(
                self.map,
                self.firetruck.graph,
                self.firetruck.nodes,
                path=None,
            )

        # ------------------------------------------------------------------
        # Step 5 — Simulation visualizer
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
        """Run until time limit, wumpus caught, or map signals done."""
        print(
            f"[Engine] Simulation starting "
            f"(duration={self.sim_duration}s, "
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
        """
        Advance one tick.  Returns False when the simulation should stop.
        """
        if self.map.sim_time > self.sim_duration or self._end_reason is not None:
            return False
        self._tick()
        return (
            self.map.sim_time <= self.sim_duration and
            self._end_reason is None
        )

    # =======================================================================
    # Core tick
    # =======================================================================

    def _tick(self) -> None:
        """
        One simulation step (0.1 s of sim time).

        Termination checks (evaluated every tick before agent logic):
          • Wumpus catch: truck within wumpus_catch_radius → _END_WUMPUS.
          • Time limit: sim_time > sim_duration → _END_TIME (handled by loop).

        Truck state machine
        -------------------
        "idle"      Pick best fire via triage; try each candidate in
                    Euclidean order until plan_to_fire() succeeds.
                    If all fail, retry next tick.
        "driving"   Follow Reeds-Shepp path.  Arrival = within
                    approach_radius of fire cell centre OR path exhausted.
        "suppressing" Proximity + flood-fill extinguish; burn-out check;
                    suppress_duration fallback.
        """

        # 1. Advance sim clock and fire-spread
        result = self.map.main()
        if result == "Done":
            self._end_reason = _END_MAP
            self.map.sim_time = self.sim_duration + 1.0
            return

        # 2. Wumpus-catch check
        # if self._check_wumpus_caught():
        #     self._end_reason = _END_WUMPUS
        #     return

        # 3. Truck state machine
        if self._truck_state == "idle":
            self._refresh_goal()
            self._replan()
            self._last_replan_time = self.map.sim_time
            if self._firetruck_path:
                self._truck_state = "driving"
            else:
                print(
                    f"[Engine] All planning attempts failed at "
                    f"t={self.map.sim_time:.1f}s — retrying next tick"
                )

        elif self._truck_state == "driving":
            self._advance_firetruck()

        elif self._truck_state == "suppressing":
            extinguished_cells = self._check_proximity_extinguish()
            if extinguished_cells:
                for cell in extinguished_cells:
                    self._extinguish_connected(cell)
                self.map.firetruck_goal = None
                self._firetruck_path    = None
                self._proximity_timers  = {}
                self._target_fire_cell  = None
                self._truck_state       = "idle"
            else:
                if (self._target_fire_cell is not None and
                        self._fire_cell_burned_out(self._target_fire_cell)):
                    print(
                        f"[Engine] Target fire {self._target_fire_cell} burned out "
                        f"at t={self.map.sim_time:.1f}s — replanning"
                    )
                    self.map.firetruck_goal = None
                    self._firetruck_path    = None
                    self._proximity_timers  = {}
                    self._target_fire_cell  = None
                    self._truck_state       = "idle"
                else:
                    elapsed = self.map.sim_time - self._suppress_start
                    if elapsed >= self.suppress_duration:
                        self._finish_suppression()
                        self._truck_state = "idle"

        # 4. Wumpus moves once per sim-second
        self._wumpus_tick_counter += 1
        if self._wumpus_tick_counter >= self.wumpus_move_interval:
            self._advance_wumpus()
            self._wumpus_tick_counter = 0
        self._wumpus_act()

        # 5. Redraw
        if self.viz:
            self.viz.update(self._firetruck_path, self._wumpus_path)

        # 6. Pace wall-clock
        if self.tick_real_time > 0:
            time.sleep(self.tick_real_time)

    # =======================================================================
    # Termination helpers
    # =======================================================================

    def _check_wumpus_caught(self) -> bool:
        """
        Return True if the truck is within wumpus_catch_radius metres of
        the wumpus.  Prints a message on first catch.
        """
        ft = self.map.firetruck_pose
        wp = self.map.wumpus_pose
        dist = math.hypot(float(ft[0]) - float(wp[0]),
                          float(ft[1]) - float(wp[1]))
        if dist <= self.wumpus_catch_radius:
            print(
                f"[Engine] WUMPUS CAUGHT! Truck reached wumpus at "
                f"dist={dist:.2f}m, t={self.map.sim_time:.1f}s"
            )
            return True
        return False

    # =======================================================================
    # Goal management
    # =======================================================================

    @staticmethod
    def _normalize_goal(goal):
        """
        Guarantee goal is a 3-tuple (x, y, theta_deg).
        2-tuples → theta=0.0; invalid strings / None → None.
        """
        if not goal or goal == "ERROR CANT GO HERE":
            return None
        if len(goal) == 2:
            return (float(goal[0]), float(goal[1]), 0.0)
        return (float(goal[0]), float(goal[1]), float(goal[2]))

    def _refresh_goal(self) -> None:
        """
        Select the target fire cell (or wumpus fallback) and store it.
        The triage list is stored on self._fire_candidates so
        _replan_firetruck() can walk it if the first choice fails.
        """
        if self.map.active_fires:
            candidates = self._rank_fire_candidates()
            # _target_fire_cell = first candidate; full list available for fallback
            self._fire_candidates: List[Tuple[int, int]] = [c for _, c in candidates]
            self._target_fire_cell = (
                self._fire_candidates[0] if self._fire_candidates else None
            )
            if self._target_fire_cell:
                cs = self.map.cell_size
                fc = self._target_fire_cell
                gx = fc[0] * cs + cs / 2.0
                gy = fc[1] * cs + cs / 2.0
                self.map.update_goal((gx, gy, 0.0))
        else:
            self._target_fire_cell  = None
            self._fire_candidates   = []
            goal = self._normalize_goal(self.map.find_firetruck_goal())
            if goal:
                self.map.update_goal(goal)

    def _fire_remaining_burn_time(self, cell: Tuple[int, int]) -> float:
        """Seconds of burn time left for a BURNING cell.  0.0 if unknown."""
        data = self.map.obstacle_coordinate_dict.get(cell)
        if data is None or data.get("status") != Status.BURNING:
            return 0.0
        burn_start = data.get("burn_time")
        if burn_start is None:
            return 0.0
        return max(0.0, self.burn_lifetime - (self.map.sim_time - burn_start))

    def _rank_fire_candidates(self) -> List[Tuple[float, Tuple[int, int]]]:
        """
        Return all active fires sorted by Euclidean distance, with viable
        fires (passing burn-time triage) listed before fallback-only fires.

        Returns list of (distance, cell) pairs — viable first, then fallback.
        This order is used by _replan_firetruck() to try candidates in sequence.
        """
        pose = self.map.firetruck_pose
        tx, ty = float(pose[0]), float(pose[1])
        cs     = self.map.cell_size
        v_max  = self.firetruck.car.v_max
        time_at_fire = self.proximity_duration + self.extinguish_margin

        viable:   List[Tuple[float, Tuple[int, int]]] = []
        fallback: List[Tuple[float, Tuple[int, int]]] = []

        for cell in self.map.active_fires:
            cx   = cell[0] * cs + cs / 2.0
            cy   = cell[1] * cs + cs / 2.0
            dist = math.hypot(tx - cx, ty - cy)
            remaining = self._fire_remaining_burn_time(cell)
            required  = dist / v_max + time_at_fire

            if remaining >= required:
                viable.append((dist, cell))
            else:
                fallback.append((dist, cell))

        viable.sort(key=lambda t: t[0])
        fallback.sort(key=lambda t: t[0])
        return viable + fallback

    def _select_best_fire_goal(self) -> Optional[Tuple[int, int]]:
        """Return the top-ranked fire cell (or None).  Used by tests."""
        ranked = self._rank_fire_candidates()
        return ranked[0][1] if ranked else None

    def _goal_has_changed(self) -> bool:
        """True if current goal position moved > 1 m since last replan."""
        current = self.map.firetruck_goal
        if current is None and self._last_goal is None:
            return False
        if current is None or self._last_goal is None:
            return True
        return (
            abs(current[0] - self._last_goal[0]) > 1.0 or
            abs(current[1] - self._last_goal[1]) > 1.0
        )

    # =======================================================================
    # Replanning
    # =======================================================================

    def _replan(self) -> None:
        self._replan_firetruck()
        self._replan_wumpus()
        self._last_goal = self.map.firetruck_goal

    def _replan_firetruck(self) -> None:
        """
        Plan toward the target fire cell.  If plan_to_fire() fails for
        the primary candidate, walk the full ranked candidate list and try
        each in turn (fallback goal selection).

        For wumpus chase (no active fires), uses the single-goal planner.
        """
        pose  = self.map.firetruck_pose
        start: State = (float(pose[0]), float(pose[1]), float(pose[2]))

        if self._target_fire_cell is not None:
            # Attempt primary candidate first, then walk the ranked list
            candidates = list(getattr(self, "_fire_candidates", []))
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
                    # Update the displayed goal to the cell we actually planned to
                    cs = self.map.cell_size
                    gx = cell[0] * cs + cs / 2.0
                    gy = cell[1] * cs + cs / 2.0
                    self.map.update_goal((gx, gy, 0.0))
                    if cell is not candidates[0]:
                        print(
                            f"[Engine] Fallback succeeded: planning to fire {cell} "
                            f"after {candidates.index(cell)} failure(s)"
                        )
                    return
                else:
                    print(
                        f"[Engine] plan_to_fire failed for cell {cell} "
                        f"at t={self.map.sim_time:.1f}s — trying next candidate"
                    )

            # All candidates exhausted
            print(
                f"[Engine] All {len(candidates)} fire candidate(s) unreachable "
                f"at t={self.map.sim_time:.1f}s"
            )

        else:
            # Wumpus chase
            goal = self._normalize_goal(self.map.firetruck_goal)
            if goal is None:
                return
            path = self.firetruck.plan(goal_state=goal, start_state=start)
            if path:
                self._firetruck_path = path
            else:
                print(f"[Engine] Firetruck replan failed at t={self.map.sim_time:.1f}s")

    def _replan_wumpus(self) -> None:
        path = self.wumpus.plan()
        if path is not None:
            self._wumpus_path = path

    # =======================================================================
    # Agent advancement
    # =======================================================================

    def _advance_firetruck(self) -> None:
        """
        Move one waypoint along the Reeds-Shepp path.
        Arrival: within approach_radius of fire cell centre OR path exhausted.
        """
        if not self._firetruck_path or len(self._firetruck_path) < 2:
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
                self._firetruck_path = [next_pose]

    def _advance_wumpus(self) -> None:
        if not self._wumpus_path or len(self._wumpus_path) < 2:
            return
        self._wumpus_path.pop(0)
        next_cell = self._wumpus_path[0]
        cs = self.map.cell_size
        self.map.wumpus_pose = (next_cell[0] * cs + cs / 2.0,
                                next_cell[1] * cs + cs / 2.0)

    # =======================================================================
    # Agent actions
    # =======================================================================

    def _check_proximity_extinguish(self) -> Set[Tuple[int, int]]:
        """
        Check burning cells within proximity_radius.  Extinguish any that
        have been in range for >= proximity_duration seconds.

        Returns the set of cells that were extinguished this tick (empty if none).
        Callers are responsible for triggering flood-fill on each returned cell.
        """
        pose = self.map.firetruck_pose
        tx, ty = float(pose[0]), float(pose[1])
        cs     = self.map.cell_size
        now    = self.map.sim_time
        extinguished: Set[Tuple[int, int]] = set()

        in_range_now: set = set()
        for cell, data in list(self.map.obstacle_coordinate_dict.items()):
            if data["status"] != Status.BURNING:
                continue
            cx = cell[0] * cs + cs / 2.0
            cy = cell[1] * cs + cs / 2.0
            if math.hypot(tx - cx, ty - cy) <= self.proximity_radius:
                in_range_now.add(cell)

        for cell in list(self._proximity_timers):
            if cell not in in_range_now:
                del self._proximity_timers[cell]

        for cell in in_range_now:
            if cell not in self._proximity_timers:
                self._proximity_timers[cell] = now

        for cell, start_t in list(self._proximity_timers.items()):
            if now - start_t >= self.proximity_duration:
                data = self.map.obstacle_coordinate_dict.get(cell)
                if data and data["status"] == Status.BURNING:
                    self.map.set_status_on_obstacles([cell], Status.EXTINGUISHED)
                    print(
                        f"[Engine] Proximity extinguish: {cell} after "
                        f"{now - start_t:.1f}s at t={now:.1f}s"
                    )
                    extinguished.add(cell)
                del self._proximity_timers[cell]

        return extinguished

    def _extinguish_connected(self, origin: Tuple[int, int]) -> None:
        """
        Flood-fill BFS from `origin` up to `flood_fill_radius` grid steps.
        Every BURNING obstacle cell reachable within that radius is
        immediately extinguished.

        "Connected" means the two cells share a grid edge (4-connectivity).
        The flood fill stops at non-obstacle cells (open ground) and at
        already-extinguished or burned cells, so it only propagates through
        the actual burning obstacle cluster.

        Parameters
        ----------
        origin : (row, col)
            The cell that was just extinguished, used as the BFS root.
        """
        visited: Set[Tuple[int, int]] = {origin}
        queue: deque = deque()
        queue.append((origin, 0))

        while queue:
            (r, c), depth = queue.popleft()
            if depth >= self.flood_fill_radius:
                continue
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                neighbour = (r + dr, c + dc)
                if neighbour in visited:
                    continue
                visited.add(neighbour)
                data = self.map.obstacle_coordinate_dict.get(neighbour)
                if data is None or data["status"] != Status.BURNING:
                    continue
                self.map.set_status_on_obstacles([neighbour], Status.EXTINGUISHED)
                print(
                    f"[Engine] Flood-fill extinguish: {neighbour} "
                    f"(depth {depth+1} from {origin}) "
                    f"at t={self.map.sim_time:.1f}s"
                )
                queue.append((neighbour, depth + 1))

    def _fire_cell_burned_out(self, cell: Tuple[int, int]) -> bool:
        """Return True if the cell is no longer BURNING."""
        data = self.map.obstacle_coordinate_dict.get(cell)
        return data is None or data["status"] != Status.BURNING

    def _target_fire_burned_out(self, goal: State) -> bool:
        """Legacy helper: True if no BURNING cell in 3×3 neighbourhood of goal."""
        cs = self.map.cell_size
        gx, gy = int(goal[0] / cs), int(goal[1] / cs)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                data = self.map.obstacle_coordinate_dict.get((gx + dr, gy + dc))
                if data and data["status"] == Status.BURNING:
                    return False
        return True

    def _finish_suppression(self) -> None:
        """
        Suppress_duration expired.  Extinguish all BURNING cells in the
        3×3 neighbourhood, then flood-fill from each extinguished cell.
        """
        pose = self.map.firetruck_pose
        cs   = self.map.cell_size
        cx   = int(pose[0] / cs)
        cy   = int(pose[1] / cs)

        extinguished = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                cell = (cx + dr, cy + dc)
                data = self.map.obstacle_coordinate_dict.get(cell)
                if data and data["status"] == Status.BURNING:
                    self.map.set_status_on_obstacles([cell], Status.EXTINGUISHED)
                    extinguished.append(cell)

        if extinguished:
            print(
                f"[Engine] Suppression complete at t={self.map.sim_time:.1f}s "
                f"— extinguished {extinguished}"
            )
            for cell in extinguished:
                self._extinguish_connected(cell)
        else:
            print(
                f"[Engine] Suppression complete at t={self.map.sim_time:.1f}s "
                f"— no burning cells found nearby"
            )

        self.map.firetruck_goal = None
        self._firetruck_path    = None
        self._target_fire_cell  = None

    def _wumpus_act(self) -> None:
        try:
            self.wumpus.burn()
        except Exception as e:
            print(f"[Engine] Wumpus burn() error: {e}")

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