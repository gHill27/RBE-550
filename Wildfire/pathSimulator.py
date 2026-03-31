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
      Firetruck  → path planning (PRM + Dubins)
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
        Whether to open the matplotlib display window.
    sim_duration : float
        Maximum simulated seconds before the engine stops (default 3600).
    """

    def __init__(
        self,
        grid_num:        int   = 50,
        cell_size:       float = 5.0,
        fill_percent:    float = 0.15,
        firetruck_start: State = (25.0, 25.0, 0.0),
        wumpus_start:    Tuple[float, float] = (220.0, 220.0),
        prm_nodes:       int   = 500,
        replan_interval: float = 5.0,
        tick_real_time:  float = 0.05,
        plot:            bool  = True,
        sim_duration:    float = 3600.0,
    ):
        self.replan_interval = replan_interval
        self.tick_real_time  = tick_real_time
        self.sim_duration    = sim_duration
        self.plot            = plot

        # Internal state
        self._firetruck_path: Optional[List[State]] = None
        self._wumpus_path:    Optional[List]        = None
        self._last_replan_time: float               = -replan_interval
        self._last_goal:        Optional[State]     = None

        # Fire targeting — grid cell currently being driven to
        self._target_fire_cell: Optional[Tuple[int,int]] = None
        # Roadmap nodes within this radius of the fire centre are goal candidates
        self.approach_radius: float = 10.0

        # Truck state machine
        # Three states:
        #   "driving"     — path exists, truck is moving toward goal
        #   "suppressing" — truck arrived, sitting at goal extinguishing fire
        #   "idle"        — suppression done, needs a new goal and plan
        self._truck_state: str            = "idle"
        self._suppress_start: float       = 0.0   # sim_time when suppression began
        self.suppress_duration: float     = 8.0   # max sim-seconds to wait at goal

        # Proximity extinguish tracking
        # Tracks how long each burning cell has been within 10m of the truck.
        # Keys = (row,col) grid cell.  Values = sim_time when proximity started.
        self._proximity_timers: dict      = {}
        self.proximity_radius: float      = 10.0  # metres — extinguish range
        self.proximity_duration: float    = 5.0   # sim-seconds within range to extinguish

        # Wumpus movement pacing — 1 cell per sim-second = every 10 ticks
        self._wumpus_tick_counter: int    = 0
        self.wumpus_move_interval: int    = 10    # ticks between wumpus moves

        # Reverse-escape state
        # When the planner fails, the truck backs up straight for
        # reverse_duration sim-seconds then retries planning.
        self._reversing: bool             = False
        self._reverse_start: float        = 0.0
        self.reverse_duration: float      = 2.0   # sim-seconds of reverse
        self.reverse_speed: float         = 3.0   # m/s backward speed

        # ------------------------------------------------------------------
        # Step 1 — Build Map WITHOUT agents (they don't exist yet)
        # ------------------------------------------------------------------
        print("[Engine] Building map...")
        self.map = Map(
            Grid_num       = grid_num,
            cell_size      = cell_size,
            fill_percent   = fill_percent,
            wumpus         = None,          # patched in step 3
            firetruck      = None,          # patched in step 3
            firetruck_pose = firetruck_start,
            wumpus_pose    = (wumpus_start[0], wumpus_start[1]),
        )

        # ------------------------------------------------------------------
        # Step 2 — Build agents (they read map at construction time)
        # ------------------------------------------------------------------
        print("[Engine] Initialising agents...")
        self.firetruck = Firetruck(self.map, plot=False)
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
        # Step 5 — Visualizer (needs map fully configured)
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

        # Seed the first goal before the first tick
        self._refresh_goal()

        while self.map.sim_time <= self.sim_duration:
            self._tick()

        print(f"[Engine] Simulation ended at t={self.map.sim_time:.1f}s")
        self._shutdown()

    def step(self) -> bool:
        """
        Advance the simulation by exactly one tick.
        Returns False when the simulation should stop, True otherwise.
        Useful for external loops or testing.

        The duration check happens AFTER the tick so that the post-tick
        sim_time (updated inside map.main()) is the value tested.
        Checking before the tick would use the pre-tick time, causing
        step() to return True one tick past the deadline.
        """
        if self.map.sim_time > self.sim_duration:
            return False
        self._tick()
        # Re-check after the tick — map.main() advanced sim_time inside
        return self.map.sim_time <= self.sim_duration

    # =======================================================================
    # Core tick
    # =======================================================================

    def _tick(self) -> None:
        """
        One simulation step (0.1 s of sim time).

        Truck state machine
        -------------------
        "idle"        — No active path.  If currently reversing, continue
                        the reverse maneuver; otherwise pick the best fire,
                        plan a path, and transition to "driving".
                        If planning fails, start a reverse-escape maneuver
                        so the truck can free itself from a wall and retry.
        "driving"     — Path exists, truck is moving forward.  No replanning
                        mid-journey.  _advance_firetruck() detects arrival
                        and transitions to "suppressing".
        "suppressing" — Truck is stationary beside the fire.  Two exit
                        conditions are checked every tick:
                          a) A nearby burning cell has been within
                             proximity_radius for >= proximity_duration
                             seconds → extinguish it immediately (early exit).
                          b) The target fire burned out on its own before
                             the truck could extinguish it → give up and
                             transition to "idle" to replan.
                          c) suppress_duration timer expires → run
                             _finish_suppression() and go idle.
        """

        # 1. Advance sim clock and fire-spread events
        result = self.map.main()
        if result == "Done":
            self.map.sim_time = self.sim_duration + 1.0
            return

        # 2. Truck state machine
        if self._truck_state == "idle":
            if self._reversing:
                # Continue backing up until reverse_duration expires
                elapsed = self.map.sim_time - self._reverse_start
                if elapsed < self.reverse_duration:
                    self._step_reverse()
                else:
                    # Reverse complete — stop reversing and try to plan again
                    self._reversing = False
                    print(f"[Engine] Reverse complete at t={self.map.sim_time:.1f}s — replanning")
            else:
                # Normal idle: pick best goal and plan
                self._refresh_goal()
                self._replan()
                self._last_replan_time = self.map.sim_time
                if self._firetruck_path:
                    self._truck_state = "driving"
                else:
                    # Planning failed — start reverse-escape maneuver
                    print(
                        f"[Engine] Planning failed at t={self.map.sim_time:.1f}s "
                        f"— starting {self.reverse_duration}s reverse escape"
                    )
                    self._reversing      = True
                    self._reverse_start  = self.map.sim_time

        elif self._truck_state == "driving":
            self._advance_firetruck()

        elif self._truck_state == "suppressing":
            # Check proximity extinguish (5s within 10m)
            if self._check_proximity_extinguish():
                # Fire extinguished early — done
                self.map.firetruck_goal  = None
                self._firetruck_path     = None
                self._proximity_timers   = {}
                self._truck_state        = "idle"
            else:
                # Check if the target fire burned out on its own
                goal = self.map.firetruck_goal
                if goal and self._target_fire_burned_out(goal):
                    print(
                        f"[Engine] Target fire burned out at t={self.map.sim_time:.1f}s "
                        f"— giving up suppression, replanning"
                    )
                    self.map.firetruck_goal  = None
                    self._firetruck_path     = None
                    self._proximity_timers   = {}
                    self._truck_state        = "idle"
                else:
                    # Check max-dwell timer
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

        2-tuples get theta=0.0 appended so the Dubins planner always
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

        For fires: picks the cheapest-to-reach burning cell via
        _select_best_fire_goal(), stores the cell on _target_fire_cell
        so _replan_firetruck knows which fire to drive to.

        For wumpus: falls back to a normalised 3-tuple goal as before.
        """
        if self.map.active_fires:
            self._target_fire_cell = self._select_best_fire_goal()
            # Store a rough world-metre goal on the map for the visualiser
            if self._target_fire_cell:
                cs  = self.map.cell_size
                fc  = self._target_fire_cell
                gx  = fc[0] * cs + cs / 2.0
                gy  = fc[1] * cs + cs / 2.0
                self.map.update_goal((gx, gy, 0.0))
        else:
            self._target_fire_cell = None
            goal = self._normalize_goal(self.map.find_firetruck_goal())
            if goal:
                self.map.update_goal(goal)

    def _select_best_fire_goal(self) -> Optional[Tuple[int, int]]:
        """
        Rank every active fire by its true driving cost via the roadmap
        and return the grid cell of the cheapest reachable fire.

        Uses cost_to_fire() which injects only the start node and runs
        multi-goal A* to any roadmap node within the approach radius —
        no geometric stop-short computation, no goal node injection.
        The best approach angle is chosen by the graph automatically.

        Returns the (row, col) grid cell of the best fire, or None.
        """
        pose  = self.map.firetruck_pose
        start = (float(pose[0]), float(pose[1]), float(pose[2]))

        best_cost: float            = float("inf")
        best_cell: Optional[Tuple[int,int]] = None

        for cell in self.map.active_fires:
            cost = self.firetruck.cost_to_fire(
                fire_cell   = cell,
                start_state = start,
                radius      = self.approach_radius,
            )
            if cost < best_cost:
                best_cost = cost
                best_cell = cell

        if best_cell is not None:
            print(
                f"[Engine] Best fire: cell={best_cell} "
                f"cost={best_cost:.1f}m  t={self.map.sim_time:.1f}s"
            )
        return best_cell

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
        # Compare x, y positions (ignore heading for change detection)
        return (
            abs(current[0] - self._last_goal[0]) > 1.0 or
            abs(current[1] - self._last_goal[1]) > 1.0
        )

    # =======================================================================
    # Replanning
    # =======================================================================

    def _replan(self) -> None:
        """
        Recompute both the firetruck and wumpus paths using their
        respective planners.  Stores results internally; the visualizer
        reads them on the next update() call.
        """
        self._replan_firetruck()
        self._replan_wumpus()
        self._last_goal = self.map.firetruck_goal

    def _replan_firetruck(self) -> None:
        """
        Plan a path to the target fire cell using plan_to_fire(), or fall
        back to the single-goal planner for the wumpus-chase case.

        For fires: uses plan_to_fire(fire_cell, radius=approach_radius).
          - Only injects the start node.
          - Multi-goal A* finds the cheapest roadmap node near the fire.
          - No stop-short computation, no geometric goal point.

        For wumpus (no active fire): uses the old plan(goal_state) path
        which injects both start and goal.
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
            # Wumpus chase — use original point-goal planner
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
        Move the firetruck one waypoint along its current path.

        Arrival detection: when the truck comes within cell_size metres
        of the goal, it is considered arrived.  Instead of immediately
        replanning, we transition to "suppressing" and start the timer.
        The truck stays stationary for suppress_duration sim-seconds
        before _finish_suppression() is called and state goes to "idle".
        """
        if not self._firetruck_path or len(self._firetruck_path) < 2:
            # Path exhausted without triggering arrival — go idle so
            # a fresh plan can be requested next tick.
            self._truck_state = "idle"
            return

        # Advance one step
        self._firetruck_path.pop(0)
        next_pose = self._firetruck_path[0]
        self.map.firetruck_pose = next_pose

        # Arrival check
        goal = self.map.firetruck_goal
        if goal:
            dist = math.hypot(
                next_pose[0] - goal[0],
                next_pose[1] - goal[1],
            )
            if dist < self.map.cell_size:
                print(
                    f"[Engine] Truck arrived at goal {goal} "
                    f"at t={self.map.sim_time:.1f}s — suppressing for "
                    f"{self.suppress_duration}s"
                )
                self._suppress_start = self.map.sim_time
                self._truck_state    = "suppressing"
                # Clear the path so the truck sits still during suppression
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

        # Convert grid cell to world-metre centre
        cs = self.map.cell_size
        wx = next_cell[0] * cs + cs / 2.0
        wy = next_cell[1] * cs + cs / 2.0
        self.map.wumpus_pose = (wx, wy)

    # =======================================================================
    # Agent actions
    # =======================================================================

    def _step_reverse(self) -> None:
        """
        Move the firetruck straight backward by one tick's worth of distance.

        The truck drives at reverse_speed m/s for reverse_duration sim-seconds.
        Each tick is 0.1s, so distance per tick = reverse_speed * 0.1.

        Direction is directly opposite the current heading — no steering during
        reverse so the truck escapes whatever wall it is facing.
        """
        pose    = self.map.firetruck_pose
        x, y, theta_deg = float(pose[0]), float(pose[1]), float(pose[2])
        theta_rad = math.radians(theta_deg)

        # Move backward along the current heading direction
        dist = self.reverse_speed * 0.1   # metres this tick (0.1s tick)
        nx   = x - dist * math.cos(theta_rad)
        ny   = y - dist * math.sin(theta_rad)

        # Only apply if the new position is collision-free
        if self.firetruck.cspace.is_free(nx, ny, theta_deg):
            self.map.firetruck_pose = (nx, ny, theta_deg)

    def _check_proximity_extinguish(self) -> bool:
        """
        Check every burning cell within proximity_radius metres of the truck.
        For each one, track how long it has been within range using
        _proximity_timers.  If any cell has been within range for
        >= proximity_duration sim-seconds, extinguish it and return True.

        Returns True if at least one cell was extinguished this tick.

        This replaces the old all-or-nothing 8s dwell timer: the truck
        now extinguishes fires as soon as it has been close enough for
        5 sim-seconds, regardless of the max suppress_duration.
        """
        pose = self.map.firetruck_pose
        tx, ty = float(pose[0]), float(pose[1])
        cs     = self.map.cell_size
        now    = self.map.sim_time
        extinguished_any = False

        # Identify all burning cells currently within proximity_radius
        in_range_now: set = set()
        for cell, data in list(self.map.obstacle_coordinate_dict.items()):
            if data["status"] != Status.BURNING:
                continue
            # Cell centre in world metres
            cx = cell[0] * cs + cs / 2.0
            cy = cell[1] * cs + cs / 2.0
            if math.hypot(tx - cx, ty - cy) <= self.proximity_radius:
                in_range_now.add(cell)

        # Update timers — start clock for newly in-range cells,
        # remove cells that left range
        for cell in list(self._proximity_timers):
            if cell not in in_range_now:
                del self._proximity_timers[cell]

        for cell in in_range_now:
            if cell not in self._proximity_timers:
                self._proximity_timers[cell] = now   # start the clock

        # Extinguish any cell that has been in range long enough
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

    def _target_fire_burned_out(self, goal: State) -> bool:
        """
        Return True if the fire the truck is suppressing has burned out
        (status BURNED or no longer in active_fires) before being extinguished.

        This is the "give up early" check: if the fire burned itself out
        naturally during the suppress_duration wait, there is nothing left
        to extinguish, so the truck should immediately replan to the next fire.
        """
        cs = self.map.cell_size
        # The goal is parked stop_distance short of the fire cell —
        # check the 3×3 neighbourhood for any remaining BURNING cell.
        gx = int(goal[0] / cs)
        gy = int(goal[1] / cs)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                cell = (gx + dr, gy + dc)
                data = self.map.obstacle_coordinate_dict.get(cell)
                if data and data["status"] == Status.BURNING:
                    return False   # fire still alive — keep suppressing
        # No burning cell found near the goal
        return True

    def _finish_suppression(self) -> None:
        """
        Called when the suppress_duration timer expires.

        Scans the 3×3 neighbourhood around the current truck position for
        any BURNING obstacles and extinguishes them.  The search radius is
        intentionally generous because the goal was parked stop_distance
        short of the fire cell — the fire is adjacent, not directly below.

        After extinguishing, clears the goal so the engine transitions to
        "idle" and selects the next fire on the following tick.
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
                f"— no burning cells found nearby (fire may have already burned out)"
            )

        # Clear goal so idle state picks the next target
        self.map.firetruck_goal = None
        self._firetruck_path    = None

    def _wumpus_act(self) -> None:
        """
        Let the wumpus burn adjacent obstacles each tick.
        Delegates to Wumpus.burn() which sets nearby cells to BURNING.
        """
        try:
            self.wumpus.burn()
        except Exception as e:
            # burn() may fail if wumpus is in an edge cell — log and continue
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