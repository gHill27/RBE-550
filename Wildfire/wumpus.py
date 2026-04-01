"""
wumpus.py
=========
Wumpus agent — uses A* on the grid to navigate toward the nearest obstacle,
then burns adjacent cells each tick.

Bugs fixed in this version
--------------------------
1. Infinite A* loop: neighbour coords were never bounds-checked, so the
   search expanded into negative/out-of-range grid indices forever.
   Fix: skip any neighbour outside [0, grid_num).

2. _unwind was defined as a @staticmethod but its first parameter was
   named like an instance method (came_from, current).  It was also not
   decorated, so self._unwind() would fail.  Fixed: explicit @staticmethod.

3. Wumpus.burn() called self.map.obstacle_set() with parentheses — but
   obstacle_set is a plain set, not a method.  Fixed: removed ().
"""

from __future__ import annotations

import math
import heapq
from typing import Optional

from Map_Generator import Map, Status


class Wumpus:
    def __init__(self, map: Map):
        self.map = map
        # 8-directional movement (cardinal + diagonal)
        self.directions = [
            (1, 0), (1, 1), (0, 1), (-1, 1),
            (-1, -1), (1, -1), (-1, 0), (0, -1),
        ]

    # ------------------------------------------------------------------
    # Path planning — A* on the grid
    # ------------------------------------------------------------------

    def plan(self):
        """
        Return a list of (row, col) grid tuples from the current wumpus
        grid cell to the nearest obstacle, or None if no obstacle exists.
        """
        goal = self.find_closest_obstacle()
        if goal is None:
            return None  # no obstacles left — nothing to navigate toward

        # Convert world-metre wumpus_pose to grid cell
        cs         = self.map.cell_size
        start_node = (
            int(self.map.wumpus_pose[0] / cs),
            int(self.map.wumpus_pose[1] / cs),
        )

        if start_node == goal:
            return [start_node]  # already there

        open_list: list = []
        heapq.heappush(
            open_list,
            (self._heuristic(start_node, goal), start_node),
        )

        cost_so_far: dict[tuple, float] = {start_node: 0.0}
        came_from:   dict[tuple, tuple] = {}
        visited:     set                = set()

        while open_list:
            _, current = heapq.heappop(open_list)

            if current == goal:
                return self._unwind(came_from, current)

            if current in visited:
                continue
            visited.add(current)

            for direction in self.directions:
                neighbour = (
                    current[0] + direction[0],
                    current[1] + direction[1],
                )

                # BUG FIX 1: bounds check — without this the search expands
                # into negative/infinite grid indices, looping forever.
                if not self._in_bounds(neighbour):
                    continue

                tentative_g = cost_so_far[current] + 1.0

                if tentative_g < cost_so_far.get(neighbour, float("inf")):
                    came_from[neighbour]  = current
                    cost_so_far[neighbour] = tentative_g
                    h = self._heuristic(neighbour, goal)
                    heapq.heappush(open_list, (tentative_g + h, neighbour))

        return None  # no path found

    def _in_bounds(self, coord: tuple) -> bool:
        """Return True if (row, col) is inside the grid."""
        row, col = coord
        return 0 <= row < self.map.grid_num and 0 <= col < self.map.grid_num

    @staticmethod
    def _heuristic(a: tuple, b: tuple) -> float:
        """Chebyshev distance — correct for 8-directional movement."""
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    # BUG FIX 2: was missing @staticmethod decorator and had a mismatched
    # signature that caused AttributeError when called as self._unwind().
    @staticmethod
    def _unwind(came_from: dict, current: tuple) -> list:
        """Reconstruct path by walking came_from back to the start."""
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(current)
        return path[::-1]

    # ------------------------------------------------------------------
    # World interaction — burn adjacent obstacles
    # ------------------------------------------------------------------

    def burn(self) -> None:
        """
        Set all obstacles adjacent to the wumpus's current grid cell
        to BURNING status.
        """
        cs  = self.map.cell_size
        row = int(self.map.wumpus_pose[0] / cs)
        col = int(self.map.wumpus_pose[1] / cs)

        cells_to_burn = []
        for dr, dc in self.directions:
            adjacent = (row + dr, col + dc)
            if adjacent in self.map.obstacle_set:
                cells_to_burn.append(adjacent)

        if cells_to_burn:
            self.map.set_status_on_obstacles(cells_to_burn, Status.BURNING)

    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------

    def find_closest_obstacle(self) -> Optional[tuple]:
        """
        Return the grid (row, col) of the obstacle closest to the wumpus,
        or None if obstacle_set is empty.
        """
        if not self.map.obstacle_set:
            return None

        cs  = self.map.cell_size
        wx  = self.map.wumpus_pose[0]
        wy  = self.map.wumpus_pose[1]

        closest_dist = math.inf
        closest      = None
        for obstacle in self.map.obstacle_set:
            if self.map.get_status(obstacle) == Status.INTACT:
                ox   = obstacle[0] * cs + cs / 2.0
                oy   = obstacle[1] * cs + cs / 2.0
                dist = math.hypot(wx - ox, wy - oy)
                if dist < closest_dist:
                    closest_dist = dist
                    closest      = obstacle

        return closest
