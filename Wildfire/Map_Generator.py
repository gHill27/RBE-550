"""
Map_Generator.py
================
World state: obstacle grid, fire spread, and goal management.

╔══════════════════════════════════════════════════════════════════════════╗
║                          AI USAGE DISCLOSURE                             ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Tool      : Claude (Anthropic) — claude-sonnet-4-6                      ║
║  Role      : Refactoring, bug-fixing, and performance partner            ║
║  Scope     : Partially AI-assisted (original logic is human-authored)    ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Contributions                                                           ║
║  ─ Fixed three bugs in the original check_time_events():                 ║
║      • find_burnable_obstacles() called without the required coordinate  ║
║        argument, causing TypeError at t≈4.9s on every run.               ║
║      • set_status_on_obstacles() called with a bare tuple instead of     ║
║        [coordinate], silently skipping the BURNED transition.            ║
║      • Removed the list() copy on obstacle_coordinate_dict.items()       ║
║        (safe because keys are not added/removed during iteration).       ║
║  ─ Rewrote set_status_on_obstacles() as a clean state-machine:           ║
║    EXTINGUISHED/BURNED are terminal; active_fires uses discard().        ║
║  ─ Collapsed eight map-generation helpers into two (_generate_safe_map,  ║
║    _random_tetromino), eliminating the N²-alloc generate_random_coord    ║
║    and the never-terminating find_neighbor_obstacles (Queue bug).        ║
║  ─ Fixed find_firetruck_goal() to convert grid cells to world metres     ║
║    (coord * cell_size + cell_size/2) rather than using raw indices.      ║
╠══════════════════════════════════════════════════════════════════════════╣
"""

from __future__ import annotations

import math
import random
from enum import Enum, auto
from typing import Optional

from shapely.geometry import Point, box


class Status(Enum):
    INTACT      = auto()
    BURNING     = auto()
    EXTINGUISHED = auto()
    BURNED      = auto()


class Map:
    def __init__(
        self,
        Grid_num:      int,
        cell_size:     float,
        fill_percent:  float,
        wumpus,
        firetruck,
        firetruck_pose: Optional[tuple] = None,
        wumpus_pose:    Optional[tuple] = None,
    ):
        self.grid_num    = Grid_num
        self.cell_size   = cell_size
        self.fill_percent = fill_percent

        # Obstacle state: coord → {status, burn_time, extinguish_time}
        self.obstacle_coordinate_dict: dict = {}
        self.obstacle_set:  set = set()   # fast membership tests
        self.active_fires:  set = set()   # subset of obstacle_set that are BURNING

        self.firetruck_goal: Optional[tuple] = None
        self.sim_time        = 0.0
        self.firetruck_pose  = firetruck_pose
        self.wumpus_pose     = wumpus_pose
        self.firetruck       = firetruck
        self.wumpus          = wumpus

        try:
            self._generate_safe_map(firetruck_pose, wumpus_pose, buffer_radius=6)
        except Exception:
            print(f"Cannot generate obstacles around poses {firetruck_pose}, {wumpus_pose}")

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------

    def main(self) -> Optional[str]:
        """Advance sim by 0.1 s, spread fire, return 'Done' after 3600 s."""
        self.sim_time += 0.1
        self._check_time_events()
        if self.sim_time > 3600:
            return "Done"
        return None

    # ------------------------------------------------------------------
    # Fire spread
    # ------------------------------------------------------------------

    def _check_time_events(self) -> None:
        """
        Advance fire state for every burning obstacle.
        Iterates obstacle_coordinate_dict directly (no list() copy needed
        because we never add/remove keys here — only mutate values).
        """
        for coord, data in self.obstacle_coordinate_dict.items():
            burn_time = data["burn_time"]
            if burn_time is None:
                continue
            elapsed = self.sim_time - burn_time
            if elapsed > 30:
                self.set_status_on_obstacles([coord], Status.BURNED)
            elif elapsed > 10:
                nearby = self._find_burnable_obstacles(coord)
                if nearby:
                    self.set_status_on_obstacles(nearby, Status.BURNING)

    def _find_burnable_obstacles(self, coord: tuple, radius: int = 6) -> list:
        """All obstacle cells within `radius` grid steps of coord."""
        cx, cy = coord
        r2     = radius * radius
        return [
            (x, y)
            for x in range(cx - radius, cx + radius + 1)
            for y in range(cy - radius, cy + radius + 1)
            if (x-cx)**2 + (y-cy)**2 <= r2 and (x, y) in self.obstacle_set
        ]

    # ------------------------------------------------------------------
    # Status management
    # ------------------------------------------------------------------

    def set_status_on_obstacles(self, coordinates: list, status: Status) -> None:
        """
        Transition obstacle cells to `status`, enforcing the state machine:
          INTACT    → BURNING or any
          BURNING   → EXTINGUISHED or BURNED
          EXTINGUISHED / BURNED → (terminal, no further transitions)
        """
        for coord in coordinates:
            data = self.obstacle_coordinate_dict.get(coord)
            if data is None:
                continue
            cur = data["status"]
            if cur in (Status.EXTINGUISHED, Status.BURNED) or cur == status:
                continue
            data["status"] = status
            if status == Status.BURNING and cur == Status.INTACT:
                data["burn_time"] = self.sim_time
                self.active_fires.add(coord)
            elif status in (Status.EXTINGUISHED, Status.BURNED):
                data["burn_time"]      = None
                data["extinguish_time"] = None
                self.active_fires.discard(coord)

    def get_status(self, coordinate: tuple) -> Status:
        return self.obstacle_coordinate_dict[coordinate]["status"]

    def update_goal(self, goal: tuple) -> None:
        self.firetruck_goal = goal

    # ------------------------------------------------------------------
    # Goal computation (used as fallback by engine)
    # ------------------------------------------------------------------

    def find_firetruck_goal(self):
        """
        Return (x_m, y_m, theta_deg) toward the nearest fire, or
        (x_m, y_m) of the wumpus if no fires are active.
        """
        if not self.active_fires:
            wp = self.wumpus_pose
            return (float(wp[0]), float(wp[1]))

        fx, fy = float(self.firetruck_pose[0]), float(self.firetruck_pose[1])
        cs     = self.cell_size
        closest_dist, closest = float("inf"), None

        for cell in self.active_fires:
            fx_c = cell[0] * cs + cs / 2.0
            fy_c = cell[1] * cs + cs / 2.0
            d    = math.hypot(fx - fx_c, fy - fy_c)
            if d < closest_dist:
                closest_dist, closest = d, cell

        tx = closest[0] * cs + cs / 2.0
        ty = closest[1] * cs + cs / 2.0
        dx, dy   = tx - fx, ty - fy
        distance = math.hypot(dx, dy)
        angle    = math.degrees(math.atan2(dy, dx)) % 360
        stop     = cs * 1.5

        if distance <= stop:
            return (fx, fy, angle)

        ratio = (distance - stop) / distance
        gx, gy = fx + dx * ratio, fy + dy * ratio
        if (int(gx / cs), int(gy / cs)) in self.obstacle_set:
            return "ERROR CANT GO HERE"
        return (gx, gy, angle)

    # ------------------------------------------------------------------
    # Map generation
    # ------------------------------------------------------------------

    def _generate_safe_map(self, start_pos, wumpus_pos, buffer_radius: float = 6.0) -> None:
        """Generate tetromino obstacles, keeping start/wumpus zones clear."""
        safe_ft  = Point(start_pos[0], start_pos[1]).buffer(buffer_radius)
        safe_wu  = Point(wumpus_pos[0], wumpus_pos[1]).buffer(buffer_radius)
        cs       = self.cell_size
        target   = int(self.grid_num * self.grid_num * self.fill_percent)
        placed   = []
        is_full  = False

        while not is_full and len(placed) < target:
            shape = self._random_tetromino()
            if shape is None:
                is_full = True
                break
            for r, c in shape:
                obs = box(r*cs, c*cs, (r+1)*cs, (c+1)*cs)
                if not (obs.intersects(safe_ft) or obs.intersects(safe_wu)):
                    placed.append((r, c))

        for coord in placed:
            self._append_obstacle(coord)

    def _random_tetromino(self, attempts: int = 0) -> Optional[list]:
        """Return a valid random tetromino, or None after 100 failed attempts."""
        if attempts > 100:
            return None
        x = random.randint(0, self.grid_num - 1)
        y = random.randint(0, self.grid_num - 1)
        shape_id = random.randint(1, 4)
        if   shape_id == 1: coords = [(x,y),(x,y-1),(x,y+1)]
        elif shape_id == 2: coords = [(x,y),(x-1,y-1),(x,y-1),(x,y+1)]
        elif shape_id == 3: coords = [(x,y),(x-1,y-1),(x-1,y),(x,y+1)]
        else:               coords = [(x,y),(x,y-1),(x,y+1),(x-1,y)]

        for r, c in coords:
            if not (0 <= r < self.grid_num and 0 <= c < self.grid_num):
                return self._random_tetromino(attempts + 1)
            if (r, c) in self.obstacle_set:
                return self._random_tetromino(attempts + 1)
        return coords

    def _append_obstacle(self, coordinate: tuple) -> None:
        self.obstacle_coordinate_dict[coordinate] = {
            "status": Status.INTACT,
            "burn_time": None,
            "extinguish_time": None,
        }
        self.obstacle_set.add(coordinate)

    def _delete_obstacle(self, coordinate: tuple) -> None:
        del self.obstacle_coordinate_dict[coordinate]
        self.obstacle_set.discard(coordinate)
        self.active_fires.discard(coordinate)

    # Keep for test compatibility
    def _append_new_obstacle(self, coordinate: tuple) -> None:
        self._append_obstacle(coordinate)

    def check_valid_cell(self, coordinate: tuple) -> bool:
        r, c = coordinate
        return (0 <= r < self.grid_num and 0 <= c < self.grid_num
                and (r, c) not in self.obstacle_set)