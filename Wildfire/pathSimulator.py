"""
simulation_engine.py
====================
SimulationEngine — the single entry point for the Wildfire simulation.

Responsibilities
----------------
  - Owns Map, Firetruck, and Wumpus construction in the right order,
    resolving the circular dependency (agents need map, map needs agents).
  - Drives the per-tick loop: clock advance → fire spread → goal selection
    → replanning → pose updates → visualizer redraw.
  - Exposes clean hooks so each agent stays focused on its own logic:
      Firetruck  → path planning (PRM + Reeds-Shepp)
      Wumpus     → path planning (A*)
      Map        → world state (obstacles, fire, time)
      Visualizer → display only
  - Handles replan throttling so the PRM isn't hammered every 0.1 s tick.
  - Provides a simple run() call to start the full simulation.

Usage
-----
    from simulation_engine import SimulationEngine

    engine = SimulationEngine(
        grid_num        = 50,
        cell_size       = 5.0,
        fill_percent    = 0.15,
        firetruck_start = (25.0, 25.0, 0.0),   # world metres (x, y, theta°)
        wumpus_start    = (220.0, 220.0),        # world metres (x, y)
        prm_nodes       = 500,
        replan_interval = 5.0,   # seconds of sim time between replans
        tick_real_time  = 0.05,  # wall-clock seconds per tick (display speed)
        plot            = True,
        plot_prm        = False, # set True to open the PRM debug visualizer
    )
    engine.run()
"""

from __future__ import annotations

import math
import time
from typing import List, Optional, Tuple

from Map_Generator import Map, Status
from firetruck import Firetruck
from wumpus import Wumpus
from pathVisualizer import SimVisualizer

# Type alias kept consistent with firetruck_prm.py
State = Tuple[float, float, float]


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
        More nodes → better paths, slower build.
    replan_interval : float
        Minimum simulated seconds between full replanning calls.
        Replanning is triggered immediately whenever the active goal
        changes (new fire, fire extinguished, wumpus becomes the goal).
    tick_real_time : float
        Wall-clock seconds to sleep between sim ticks.
        0.0 = run as fast as possible.
    plot : bool
        Whether to open the main simulation matplotlib display window.
    plot_prm : bool
        Whether to open the PRM debug visualizer window (PlannerVisualizer).
        Useful for inspecting the roadmap graph and sampled nodes during
        development.  Has no effect on simulation logic.
    sim_duration : float
        Maximum simulated seconds before the engine stops (default 3600).
    extinguish_margin : float
        Extra sim-seconds of buffer required beyond proximity_duration when
        pre-filtering reachable fires.  A fire is only targeted if the truck
        can reach it AND still have (proximity_duration + extinguish_margin)
        seconds of burn time remaining.  Default 5.0 s.
    burn_lifetime : float
        Total sim-seconds a cell burns before it burns out naturally.
        Used to compute remaining burn time for triage.  Default 30.0 s.
    """

    def __init__(
        self,
        grid_num:          int   = 50,
        cell_size:         float = 5.0,
        fill_percent:      float = 0.15,
        firetruck_start:   State = (25.0, 25.0, 0.0),
        wumpus_start:      Tuple[float, float] = (220.0, 220.0),
        prm_nodes:         int   = 500,
        replan_interval:   float = 5.0,
        tick_real_time:    float = 0.05,
        plot:              bool  = True,
        plot_prm:          bool  = False,
        sim_duration:      float = 3600.0,
        extinguish_margin: float = 5.0,
        burn_lifetime:     float = 30.0,
    ):
        self.replan_interval   = replan_interval
        self.tick_real_time    = tick_real_time
        self.sim_duration      = sim_duration
        self.plot              = plot
        self.plot_prm          = plot_prm
        self.extinguish_margin = extinguish_margin
        self.burn_lifetime     = burn_lifetime

        # Internal state
        self._firetruck_path: Optional[List[State]] = None
        self._wumpus_path:    Optional[List]        = None
        self._last_replan_time: float               = -replan_interval
        self._last_goal:        Optional[State]     = None

        # Fire targeting — grid cell currently being driven to
        self._target_fire_cell: Optional[Tuple[int, int]] = None
        # Roadmap nodes within this radius of the fire centre are goal candidates
        self.approach_radius: float = 10.0

        # Truck state machine
        # Three states:
        #   "idle"        — No active path.  Pick the best fire via Euclidean
        #                   + burn-time triage, plan a path, and transition to
        #                   "driving".  If planning fails, log and retry.
        #   "driving"     — Following the Reeds-Shepp path waypoint by waypoint.
        #                   Arrival is detected when the truck enters
        #                   approach_radius of the target fire cell centre, or
        #                   when the path is exhausted.
        #   "suppressing" — Sitting beside the fire, running proximity timers.
        self._truck_state: str      = "idle"
        self._suppress_start: float = 0.0    # sim_time when suppression began
        self.suppress_duration: float = 8.0  # max sim-seconds to dwell at goal

        # Proximity extinguish tracking
        # Keys = (row,col) grid cell.  Values = sim_time when proximity started.
        self._proximity_timers: dict   = {}
        self.proximity_radius: float   = 10.0  # metres — extinguish range
        self.proximity_duration: float = 5.0   # sim-seconds within range to extinguish

        # Wumpus movement pacing — 1 cell per sim-second = every 10 ticks
        self._wumpus_tick_counter: int = 0
        self.wumpus_move_interval: int = 10    # ticks between wumpus moves

        # ------------------------------------------------------------------
        # Step 1 — Build Map WITHOUT agents (they don't exist yet)
        # ------------------------------------------------------------------
        print("[Engine] Building map...")
        self.map = Map(
            Grid_num       = grid_num,
            cell_size      = cell_size,
            fill_percent   = fill_percent,
            wumpus         = None,   # patched in step 3
            firetruck      = None,   # patched in step 3
            firetruck_pose = firetruck_start,
            wumpus_pose    = (wumpus_start[0], wumpus_start[1]),
        )

        # ------------------------------------------------------------------
        # Step 2 — Build agents (they read map at construction time)
        # ------------------------------------------------------------------
        print("[Engine] Initialising agents...")
        self.firetruck = Firetruck(self.map, plot=plot_prm)
        self.wumpus    = Wumpus(self.map)

        # ------------------------------------------------------------------
        # Step 3 — Patch agents back into the map (resolves circular dep)
        # ------------------------------------------------------------------
        self.map.firetruck = self.firetruck
        self.map.wumpus    = self.wumpus

        # ------------------------------------------------------------------
        # Step 4 — Build PRM roadmap (one-time, slow operation)
        # ------------------------------------------------------------------
        print(f"[Engine] Building PRM roadmap ({prm_nodes} nodes)...")
        self.firetruck.build_tree(n_samples=prm_nodes)
        print("[Engine] Roadmap ready.")

        # ------------------------------------------------------------------
        # Step 5 — Simulation visualizer (needs map fully configured)
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
        """
        Start the simulation loop and block until it finishes.
        The loop runs until sim_time exceeds sim_duration or Map.main()
        signals completion.
        """
        print(f"[Engine] Simulation starting (duration={self.sim_duration}s)...")
        self._refresh_goal()

        while self.map.sim_time <= self.sim_duration:
            self._tick()

        print(f"[Engine] Simulation ended at t={self.map.sim_time:.1f}s")
        self._shutdown()

    def step(self) -> bool:
        """
        Advance the simulation by exactly one tick.
        Returns False when the simulation should stop, True otherwise.

        The duration check happens AFTER the tick so that the post-tick
        sim_time (updated inside map.main()) is the value tested.
        """
        if self.map.sim_time > self.sim_duration:
            return False
        self._tick()
        return self.map.sim_time <= self.sim_duration

    # =======================================================================
    # Core tick
    # =======================================================================

    def _tick(self) -> None:
        """
        One simulation step (0.1 s of sim time).

        Truck state machine
        -------------------
        "idle"
            Pick the best reachable fire using Euclidean-distance + burn-time
            triage (_select_best_fire_goal), then run plan_to_fire().
            Transition to "driving" if a path is found; otherwise log and
            retry next tick.  There is no reverse-escape: Reeds-Shepp curves
            handle any required reversing natively.

        "driving"
            Follow the Reeds-Shepp path one waypoint per tick.
            Arrival is detected by _advance_firetruck() when:
              (a) the truck enters approach_radius of the target fire cell, OR
              (b) the path is exhausted.
            On arrival → transition to "suppressing".

        "suppressing"
            Sit beside the fire.  Exit conditions (checked every tick):
              (a) Proximity extinguish: a burning cell has been within
                  proximity_radius for >= proximity_duration seconds
                  → extinguish it and go idle.
              (b) Target burned out naturally → go idle.
              (c) suppress_duration timer expires → _finish_suppression(),
                  extinguish the 3x3 neighbourhood, go idle.
        """

        # 1. Advance sim clock and fire-spread events
        result = self.map.main()
        if result == "Done":
            self.map.sim_time = self.sim_duration + 1.0
            return

        # 2. Truck state machine
        if self._truck_state == "idle":
            self._refresh_goal()
            self._replan()
            self._last_replan_time = self.map.sim_time
            if self._firetruck_path:
                self._truck_state = "driving"
            else:
                print(
                    f"[Engine] Planning failed at t={self.map.sim_time:.1f}s "
                    f"— will retry next tick"
                )

        elif self._truck_state == "driving":
            self._advance_firetruck()

        elif self._truck_state == "suppressing":
            if self._check_proximity_extinguish():
                # A nearby cell was extinguished — done with this fire
                self.map.firetruck_goal = None
                self._firetruck_path    = None
                self._proximity_timers  = {}
                self._target_fire_cell  = None
                self._truck_state       = "idle"
            else:
                # Check if fire burned out on its own using the cell directly
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

        # 3. Wumpus moves once per sim-second (every wumpus_move_interval ticks)
        self._wumpus_tick_counter += 1
        if self._wumpus_tick_counter >= self.wumpus_move_interval:
            self._advance_wumpus()
            self._wumpus_tick_counter = 0
        self._wumpus_act()

        # 4. Redraw
        if self.viz:
            self.viz.update(self._firetruck_path, self._wumpus_path)

        # 5. Pace wall-clock time
        if self.tick_real_time > 0:
            time.sleep(self.tick_real_time)

    # =======================================================================
    # Goal management
    # =======================================================================

    @staticmethod
    def _normalize_goal(goal):
        """
        Guarantee the goal is always a 3-tuple (x, y, theta_deg).

        find_firetruck_goal() can return:
          - (x, y, theta)  heading toward a fire        <- already correct
          - (x, y)         falling back to wumpus_pose  <- missing theta
          - "ERROR CANT GO HERE"                        <- invalid, discard
          - None                                        <- nothing to do

        2-tuples get theta=0.0 appended so the Reeds-Shepp planner always
        receives a valid State and never raises IndexError on goal[2].
        """
        if not goal or goal == "ERROR CANT GO HERE":
            return None
        if len(goal) == 2:
            return (float(goal[0]), float(goal[1]), 0.0)
        return (float(goal[0]), float(goal[1]), float(goal[2]))

    def _refresh_goal(self) -> None:
        """
        Select the target fire cell (or wumpus fallback) and store it.

        For fires: _select_best_fire_goal() filters by Euclidean reachability
        and burn-time budget, then returns the closest surviving candidate.

        For wumpus: normalises the 2-tuple return from find_firetruck_goal().
        """
        if self.map.active_fires:
            self._target_fire_cell = self._select_best_fire_goal()
            if self._target_fire_cell:
                cs = self.map.cell_size
                fc = self._target_fire_cell
                gx = fc[0] * cs + cs / 2.0
                gy = fc[1] * cs + cs / 2.0
                self.map.update_goal((gx, gy, 0.0))
        else:
            self._target_fire_cell = None
            goal = self._normalize_goal(self.map.find_firetruck_goal())
            if goal:
                self.map.update_goal(goal)

    def _fire_remaining_burn_time(self, cell: Tuple[int, int]) -> float:
        """
        Return the number of sim-seconds this cell has left before it burns
        out naturally.

        Uses burn_time recorded in obstacle_coordinate_dict (the sim_time
        when the cell started burning) and self.burn_lifetime (total burn
        duration).  Returns 0.0 if the cell is not burning or has no
        burn_time recorded.
        """
        data = self.map.obstacle_coordinate_dict.get(cell)
        if data is None or data.get("status") != Status.BURNING:
            return 0.0
        burn_start = data.get("burn_time")
        if burn_start is None:
            return 0.0
        elapsed = self.map.sim_time - burn_start
        return max(0.0, self.burn_lifetime - elapsed)

    def _select_best_fire_goal(self) -> Optional[Tuple[int, int]]:
        """
        Choose the best burning cell to target using a fast Euclidean
        distance + burn-time pre-filter.

        Algorithm
        ---------
        For each active fire cell:
          1. Compute Euclidean distance from the truck to the fire centre.
          2. Estimate travel time = distance / car.v_max   (optimistic lower bound).
          3. Time needed at fire  = proximity_duration + extinguish_margin.
          4. Required budget      = travel_time + time_needed_at_fire.
          5. Remaining burn time from the Map's burn_time record.

          Accept the cell only if remaining_burn_time >= required_budget.

        Among all accepted cells, return the one with the smallest
        Euclidean distance (fastest to reach under the optimistic model).

        If NO cell passes the filter (all fires are too far gone or burn_time
        is not yet recorded), fall back to the single closest fire so the
        truck is never left idle when fires still exist.

        This replaces the expensive cost_to_fire() PRM query that was being
        called for every active fire on every goal refresh tick.
        """
        pose = self.map.firetruck_pose
        tx, ty = float(pose[0]), float(pose[1])
        cs    = self.map.cell_size
        v_max = self.firetruck.car.v_max

        # Seconds the truck must spend at the fire once it arrives
        time_at_fire = self.proximity_duration + self.extinguish_margin

        viable:   list = []   # (distance, cell) — pass the burn-time filter
        fallback: list = []   # (distance, cell) — all cells regardless

        for cell in self.map.active_fires:
            cx   = cell[0] * cs + cs / 2.0
            cy   = cell[1] * cs + cs / 2.0
            dist = math.hypot(tx - cx, ty - cy)

            travel_time = dist / v_max          # optimistic lower bound
            required    = travel_time + time_at_fire
            remaining   = self._fire_remaining_burn_time(cell)

            fallback.append((dist, cell))

            if remaining >= required:
                viable.append((dist, cell))

        if viable:
            best_dist, best_cell = min(viable, key=lambda t: t[0])
            print(
                f"[Engine] Target fire: cell={best_cell} "
                f"euclid={best_dist:.1f}m  t={self.map.sim_time:.1f}s"
            )
            return best_cell

        # All fires are nearly burned out — go for the closest as fallback
        if fallback:
            best_dist, best_cell = min(fallback, key=lambda t: t[0])
            print(
                f"[Engine] Fallback fire (no viable): cell={best_cell} "
                f"euclid={best_dist:.1f}m  t={self.map.sim_time:.1f}s"
            )
            return best_cell

        return None

    def _goal_has_changed(self) -> bool:
        """
        Return True if the current goal differs meaningfully from the last
        planned goal.  Triggers an immediate replan when a new fire starts
        or is extinguished.
        """
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
        """
        Recompute both the firetruck and wumpus paths.
        Stores results internally; the visualizer reads them on the next
        update() call.
        """
        self._replan_firetruck()
        self._replan_wumpus()
        self._last_goal = self.map.firetruck_goal

    def _replan_firetruck(self) -> None:
        """
        Plan a path to the target fire cell using plan_to_fire(), or fall
        back to the single-goal planner for the wumpus-chase case.
        """
        pose  = self.map.firetruck_pose
        start: State = (float(pose[0]), float(pose[1]), float(pose[2]))

        if self._target_fire_cell is not None:
            path = self.firetruck.plan_to_fire(
                fire_cell   = self._target_fire_cell,
                start_state = start,
                radius      = self.approach_radius,
            )
            if path:
                self._firetruck_path = path
            else:
                print(
                    f"[Engine] plan_to_fire failed for cell "
                    f"{self._target_fire_cell} at t={self.map.sim_time:.1f}s"
                )
        else:
            goal = self._normalize_goal(self.map.firetruck_goal)
            if goal is None:
                return
            path = self.firetruck.plan(goal_state=goal, start_state=start)
            if path:
                self._firetruck_path = path
            else:
                print(
                    f"[Engine] Firetruck replan failed at t={self.map.sim_time:.1f}s"
                )

    def _replan_wumpus(self) -> None:
        path = self.wumpus.plan()
        if path is not None:
            self._wumpus_path = path

    # =======================================================================
    # Agent advancement
    # =======================================================================

    def _advance_firetruck(self) -> None:
        """
        Move the firetruck one waypoint along its current Reeds-Shepp path
        and check for arrival at the target fire cell.

        Arrival is defined as the truck being within approach_radius metres
        of the target fire cell's world-metre centre.  We intentionally do
        NOT compare against map.firetruck_goal because the PRM planner
        routes to a roadmap NODE near the fire centre — the path ends at
        the node, not at the fire centre itself.  Comparing against the
        fire cell directly keeps the trigger consistent with plan_to_fire()'s
        radius parameter and prevents the bounce loop (plan → arrive at
        roadmap node → not within cell_size of goal → path exhausts → replan
        to same fire → repeat).

        Exit paths
        ----------
        (a) Path exhausted → commit to suppression wherever the truck ended up.
        (b) Truck enters approach_radius of fire cell → immediate suppression.
        """
        if not self._firetruck_path or len(self._firetruck_path) < 2:
            # Path exhausted — transition to suppression at current position
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

        # Advance one step along the Reeds-Shepp path
        self._firetruck_path.pop(0)
        next_pose = self._firetruck_path[0]
        self.map.firetruck_pose = next_pose

        # Arrival check: within approach_radius of the target fire cell centre
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
                # Freeze path so truck stays put during suppression
                self._firetruck_path = [next_pose]

    def _advance_wumpus(self) -> None:
        """
        Move the wumpus one grid step along its current path.
        Wumpus path is in grid (row, col) tuples.
        """
        if not self._wumpus_path or len(self._wumpus_path) < 2:
            return

        self._wumpus_path.pop(0)
        next_cell = self._wumpus_path[0]

        cs = self.map.cell_size
        wx = next_cell[0] * cs + cs / 2.0
        wy = next_cell[1] * cs + cs / 2.0
        self.map.wumpus_pose = (wx, wy)

    # =======================================================================
    # Agent actions
    # =======================================================================

    def _check_proximity_extinguish(self) -> bool:
        """
        Check every burning cell within proximity_radius metres of the truck.
        Track how long each one has been in range via _proximity_timers.
        If any cell reaches proximity_duration, extinguish it and return True.
        """
        pose = self.map.firetruck_pose
        tx, ty = float(pose[0]), float(pose[1])
        cs     = self.map.cell_size
        now    = self.map.sim_time
        extinguished_any = False

        in_range_now: set = set()
        for cell, data in list(self.map.obstacle_coordinate_dict.items()):
            if data["status"] != Status.BURNING:
                continue
            cx = cell[0] * cs + cs / 2.0
            cy = cell[1] * cs + cs / 2.0
            if math.hypot(tx - cx, ty - cy) <= self.proximity_radius:
                in_range_now.add(cell)

        # Drop cells that left range
        for cell in list(self._proximity_timers):
            if cell not in in_range_now:
                del self._proximity_timers[cell]

        # Start clock for newly in-range cells
        for cell in in_range_now:
            if cell not in self._proximity_timers:
                self._proximity_timers[cell] = now

        # Extinguish cells that have been in range long enough
        for cell, start_t in list(self._proximity_timers.items()):
            if now - start_t >= self.proximity_duration:
                data = self.map.obstacle_coordinate_dict.get(cell)
                if data and data["status"] == Status.BURNING:
                    self.map.set_status_on_obstacles([cell], Status.EXTINGUISHED)
                    print(
                        f"[Engine] Proximity extinguish: {cell} after "
                        f"{now - start_t:.1f}s within {self.proximity_radius}m "
                        f"at t={now:.1f}s"
                    )
                    extinguished_any = True
                del self._proximity_timers[cell]

        return extinguished_any

    def _fire_cell_burned_out(self, cell: Tuple[int, int]) -> bool:
        """
        Return True if the target fire cell is no longer BURNING —
        either burned out naturally or already extinguished.
        """
        data = self.map.obstacle_coordinate_dict.get(cell)
        if data is None:
            return True
        return data["status"] != Status.BURNING

    def _target_fire_burned_out(self, goal: State) -> bool:
        """
        Legacy helper: return True if no BURNING cell remains in the
        3x3 neighbourhood of a world-metre goal position.
        Kept for test compatibility; prefer _fire_cell_burned_out(cell).
        """
        cs = self.map.cell_size
        gx = int(goal[0] / cs)
        gy = int(goal[1] / cs)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                cell = (gx + dr, gy + dc)
                data = self.map.obstacle_coordinate_dict.get(cell)
                if data and data["status"] == Status.BURNING:
                    return False
        return True

    def _finish_suppression(self) -> None:
        """
        Called when suppress_duration expires.  Extinguishes all BURNING
        cells in the 3x3 neighbourhood around the truck's current position,
        then clears the goal so the engine picks a new target next tick.
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
        else:
            print(
                f"[Engine] Suppression complete at t={self.map.sim_time:.1f}s "
                f"— no burning cells found nearby"
            )

        self.map.firetruck_goal = None
        self._firetruck_path    = None
        self._target_fire_cell  = None

    def _wumpus_act(self) -> None:
        """Let the wumpus burn adjacent obstacles each tick."""
        try:
            self.wumpus.burn()
        except Exception as e:
            print(f"[Engine] Wumpus burn() error: {e}")

    # =======================================================================
    # Shutdown
    # =======================================================================

    def _shutdown(self) -> None:
        """Print final stats and keep the display open."""
        counts = {"INTACT": 0, "BURNING": 0, "EXTINGUISHED": 0, "BURNED": 0}
        for data in self.map.obstacle_coordinate_dict.values():
            counts[data["status"].name] += 1

        print("\n[Engine] ── Final Statistics ──────────────────")
        print(f"  Sim time       : {self.map.sim_time:.1f}s")
        print(f"  Intact cells   : {counts['INTACT']}")
        print(f"  Burned cells   : {counts['BURNED']}")
        print(f"  Extinguished   : {counts['EXTINGUISHED']}")
        print(f"  Still burning  : {counts['BURNING']}")
        print("────────────────────────────────────────────────\n")

        if self.viz:
            self.viz.close()