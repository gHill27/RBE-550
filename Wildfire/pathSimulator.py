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

import time
from typing import List, Optional, Tuple

from Map_Generator import Map, Status
from firetruck import Firetruck
from wumpus import Wumpus
from pathVisualizer import SimVisualizer
import math
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
        fill_percent:    float = 0.05,
        firetruck_start: State = (25.0, 25.0, 0.0),
        wumpus_start:    Tuple[float, float] = (220.0, 220.0),
        prm_nodes:       int   = 500,
        replan_interval: float = 30.0,
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
        self._last_replan_time: float               = -replan_interval  # force immediate replan
        self._last_goal:        Optional[State]     = None              # detect goal changes

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
        """One simulation step — called every 0.1 s of sim time."""

        # 1. Advance sim clock and run Map-level events (fire spread, burn-out)
        #    map.main() calls wumpus.plan() and firetruck.plan() internally.
        #    We guard the return value so a "Done" signal exits cleanly.
        result = self.map.main()
        if result == "Done":
            # Force sim_time past sim_duration so run() and step() both stop
            self.map.sim_time = self.sim_duration + 1.0
            return

        # 2. Decide whether to replan
        time_since_replan = self.map.sim_time - self._last_replan_time
        goal_changed      = self._goal_has_changed()

        if goal_changed or time_since_replan >= self.replan_interval:
            self._refresh_goal()
            self._replan()
            self._last_replan_time = self.map.sim_time

        # 3. Advance agent positions along their current paths
        self._advance_firetruck()
        self._advance_wumpus()

        # 4. Let the wumpus act on the world (burn nearby obstacles)
        self._wumpus_act()

        # 5. Redraw
        if self.viz:
            self.viz.update(self._firetruck_path, self._wumpus_path)

        # 6. Pace wall-clock time
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
        Select the best reachable fire goal and store it on the map.
 
        When fires are active, calls _select_best_fire_goal() to rank every
        burning cell by its true Dubins path cost from the current truck pose,
        and picks the cheapest reachable one.
 
        Falls back to the wumpus position (normalised to a 3-tuple) when
        there are no active fires.
        """
        if self.map.active_fires:
            goal = self._select_best_fire_goal()
        else:
            goal = self._normalize_goal(self.map.find_firetruck_goal())
 
        if goal:
            self.map.update_goal(goal)


    def _select_best_fire_goal(self) -> Optional[State]:
        """
        Rank every active fire by its Dubins path cost from the current
        truck pose and return a goal state for the cheapest reachable one.
 
        Why this matters
        ----------------
        Dubins curves are asymmetric: a fire that is physically "close"
        but lies directly behind the truck requires a wide U-turn arc
        before the truck can approach it forward.  Euclidean distance
        completely misses this — it treats all fires at the same radius
        identically regardless of heading.
 
        This method probes the PRM planner for each candidate fire and
        ranks them by what it would actually cost to drive there.  Fires
        that are behind the truck, across obstacles, or in dead-ends are
        naturally deprioritised even if they are metrically "closer".
 
        Implementation
        --------------
        Calling firetruck.path_cost() per fire runs a lightweight A* probe
        (inject two temp nodes, run search, clean up) without touching the
        stored firetruck path or roadmap.  For a typical simulation with
        3-10 active fires the overhead is negligible compared to the full
        plan() call that follows.
 
        Returns None if no fire is reachable from the current pose.
        """
        cs    = self.map.cell_size
        pose  = self.map.firetruck_pose
        start = (float(pose[0]), float(pose[1]), float(pose[2]))
        stop  = cs * 1.5   # park stop_distance short of the fire cell edge
 
        best_cost = float("inf")
        best_goal: Optional[State] = None
 
        for cell in self.map.active_fires:
            # Convert grid cell to world-metre centre
            fire_wx = cell[0] * cs + cs / 2.0
            fire_wy = cell[1] * cs + cs / 2.0
 
            # Direction from truck to fire
            dx   = fire_wx - start[0]
            dy   = fire_wy - start[1]
            dist = math.hypot(dx, dy)
 
            if dist < 1e-3:
                # Already on top of this fire — zero cost, take it immediately
                return (start[0], start[1], start[2])
 
            angle_deg = math.degrees(math.atan2(dy, dx)) % 360.0
 
            if dist <= stop:
                # Already within stop distance — face the fire, cost = 0
                goal: State = (start[0], start[1], angle_deg)
                return goal
 
            # Goal = stop_distance short along truck→fire vector
            ratio  = (dist - stop) / dist
            goal_x = start[0] + dx * ratio
            goal_y = start[1] + dy * ratio
 
            # Reject goals that land inside an obstacle
            goal_cell = (int(goal_x / cs), int(goal_y / cs))
            if goal_cell in self.map.obstacle_set:
                continue
 
            candidate: State = (goal_x, goal_y, angle_deg)
 
            # Probe the PRM planner for the real driving cost
            cost = self.firetruck.path_cost(
                goal_state  = candidate,
                start_state = start,
            )
 
            if cost < best_cost:
                best_cost = cost
                best_goal = candidate
 
        if best_goal is None:
            # No fire was reachable — fall back to wumpus or centre
            return self._normalize_goal(self.map.find_firetruck_goal())
 
        print(
            f"[Engine] Best fire goal: {best_goal} "
            f"(Dubins cost={best_cost:.1f}m, t={self.map.sim_time:.1f}s)"
        )
        return best_goal

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
        # Normalize defensively — goal may have been set externally as a 2-tuple
        goal = self._normalize_goal(self.map.firetruck_goal)
        if goal is None:
            return

        pose = self.map.firetruck_pose
        start: State = (float(pose[0]), float(pose[1]), float(pose[2]))

        path = self.firetruck.plan(goal_state=goal, start_state=start)
        if path:
            self._firetruck_path = path
        else:
            # Keep the old path rather than showing nothing
            print(
                f"[Engine] Firetruck replan failed at t={self.map.sim_time:.1f}s "
                f"(start={start}, goal={goal})"
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
        Move the firetruck one step along its current path.
        Each tick represents 0.1 s of sim time; we move one waypoint
        per tick (the path is dense at ~0.5–1.0 m spacing from Dubins).
        """
        if not self._firetruck_path or len(self._firetruck_path) < 2:
            return

        # Pop the first waypoint — the truck is now at the second
        self._firetruck_path.pop(0)
        next_pose = self._firetruck_path[0]
        self.map.firetruck_pose = next_pose

        # Check if the truck has reached the goal (within one cell)
        goal = self.map.firetruck_goal
        if goal:
            dist = (
                (next_pose[0] - goal[0]) ** 2 +
                (next_pose[1] - goal[1]) ** 2
            ) ** 0.5
            if dist < self.map.cell_size:
                self._on_firetruck_reached_goal()

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

    def _on_firetruck_reached_goal(self) -> None:
        """
        Called when the firetruck arrives within stop_distance of its goal.
        Scans nearby grid cells for any BURNING obstacle and extinguishes it.

        We search a small radius around the goal rather than converting the
        goal exactly to one cell, because the goal is intentionally placed
        stop_distance metres short of the fire — the fire cell itself is
        adjacent, not directly under the goal point.
        """
        goal = self.map.firetruck_goal
        if goal is None:
            return

        cs       = self.map.cell_size
        gx, gy   = goal[0], goal[1]

        # Check the goal cell and its 8 neighbours for a burning obstacle
        goal_cell = (int(gx / cs), int(gy / cs))
        candidates = [
            goal_cell,
            (goal_cell[0] + 1, goal_cell[1]),
            (goal_cell[0] - 1, goal_cell[1]),
            (goal_cell[0],     goal_cell[1] + 1),
            (goal_cell[0],     goal_cell[1] - 1),
            (goal_cell[0] + 1, goal_cell[1] + 1),
            (goal_cell[0] - 1, goal_cell[1] - 1),
            (goal_cell[0] + 1, goal_cell[1] - 1),
            (goal_cell[0] - 1, goal_cell[1] + 1),
        ]

        extinguished_any = False
        for cell in candidates:
            if cell in self.map.obstacle_coordinate_dict:
                status = self.map.obstacle_coordinate_dict[cell]["status"]
                if status == Status.BURNING:
                    print(
                        f"[Engine] Firetruck extinguishing {cell} "
                        f"at t={self.map.sim_time:.1f}s"
                    )
                    self.map.set_status_on_obstacles([cell], Status.EXTINGUISHED)
                    extinguished_any = True

        if extinguished_any:
            print(f"[Engine] Fire suppressed near goal {goal_cell}")

        # Goal is consumed — clear it so _goal_has_changed triggers a replan
        self.map.firetruck_goal = None

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