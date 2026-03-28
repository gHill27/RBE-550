"""
pathSimulator.py
================
PathSimulator — updated to work with the new PRM/Dubins planner output.

Key changes from original
--------------------------
  - World size derived from map.grid_num * map.cell_size (no hardcoded 36).
  - Obstacles drawn from map.obstacle_set instead of obstacle_coordinate_dict.
  - Accepts 3-tuple waypoints (x, y, theta_deg) from Dubins path output.
  - Multi-goal support: run_multi_goal() plans and simulates sequential goals.
  - Environment-only preview: show_environment() renders map with no path.
  - _interpolate() now smooths between Dubins waypoints (already dense),
    so it just linearly fills at the desired fps rather than assuming sparse nodes.
  - Heading interpolation uses shortest-angle arithmetic to avoid 359→1 flipping.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt


# Type alias matching your State convention
State = Tuple[float, float, float]   # (x, y, theta_deg)


class PathSimulator:
    def __init__(
        self,
        vehicle,                        # your Firetruck instance
        fps: int = 30,
    ):
        self.vehicle = vehicle
        self.fps     = fps

        # Derive world size from the map attached to the vehicle
        self._world_size = (
            vehicle.map.grid_num * vehicle.map.cell_size
        )
        self._cell_size = vehicle.map.cell_size

        self.fig: Optional[plt.Figure] = None
        self.ax:  Optional[plt.Axes]   = None
        self.ani: Optional[animation.FuncAnimation] = None

    # =======================================================================
    # Public API
    # =======================================================================

    def show_environment(
        self,
        goals: Optional[List[State]] = None,
        title: str = "Environment",
    ) -> None:
        """
        Render the map and optional goal markers with no path or lattice.
        Useful for sanity-checking obstacle layout before planning.

        Parameters
        ----------
        goals : list of (x, y, theta_deg) goal poses to mark on the map
        title : window title
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        self._setup_axes(ax, title)
        self._draw_obstacles(ax)

        if goals:
            colors = ["#1D9E75", "#EF9F27", "#534AB7", "#D85A30"]
            for i, g in enumerate(goals):
                color = colors[i % len(colors)]
                ax.plot(g[0], g[1], "o", color=color,
                        markersize=10, zorder=5, label=f"Goal {i+1}")
                self._draw_heading_arrow(ax, g, length=self._world_size * 0.03,
                                         color=color, lw=2.0)

        if goals:
            ax.legend(loc="upper right", fontsize=8)

        plt.tight_layout()
        plt.show()

    def run(
        self,
        path: List[State],
        title: str = "Vehicle simulation",
        trail_color: str = "#2ecc71",
    ) -> None:
        """
        Animate the vehicle following a single planned path.

        Parameters
        ----------
        path        : list of (x, y, theta_deg) from Firetruck.plan()
        title       : window title
        trail_color : colour of the ghost trail line
        """
        if not path:
            print("PathSimulator.run(): empty path, nothing to animate.")
            return

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self._setup_axes(self.ax, title)
        self._draw_obstacles(self.ax)

        # Ghost trail of the planned path
        px = [p[0] for p in path]
        py = [p[1] for p in path]
        self.ax.plot(px, py, "--", color=trail_color,
                     alpha=0.4, linewidth=1.0, label="Planned path")

        # Goal marker
        goal = path[-1]
        self.ax.plot(goal[0], goal[1], "*",
                     color="#EF9F27", markersize=14, zorder=6, label="Goal")
        self._draw_heading_arrow(self.ax, goal,
                                  length=self._world_size * 0.03,
                                  color="#EF9F27", lw=2.0)

        # Dense simulation path
        sim_path = self._interpolate(path)

        # Initial vehicle patch
        vehicle_patch = self._make_vehicle_patch(sim_path[0])
        for p in vehicle_patch:
            self.ax.add_patch(p)

        # Trace line (grows as vehicle moves)
        trace_x, trace_y = [sim_path[0][0]], [sim_path[0][1]]
        (trace_line,) = self.ax.plot(
            trace_x, trace_y,
            color=trail_color, linewidth=1.5, zorder=4,
        )

        self.ax.legend(loc="upper right", fontsize=8)

        def update(frame: int):
            state = sim_path[frame]
            new_patches = self._make_vehicle_patch(state)
            for patch, new_p in zip(vehicle_patch, new_patches):
                patch.set_xy(list(new_p.get_xy()))

            trace_x.append(state[0])
            trace_y.append(state[1])
            trace_line.set_data(trace_x, trace_y)
            return vehicle_patch + [trace_line]

        self.ani = animation.FuncAnimation(
            self.fig, update,
            frames=len(sim_path),
            interval=1000 // self.fps,
            blit=True,
        )
        plt.tight_layout()
        plt.show()

    def run_multi_goal(
        self,
        goals: List[State],
        colors: Optional[List[str]] = None,
        title: str = "Multi-goal simulation",
    ) -> None:
        """
        Plan and animate sequential legs from the truck's current pose
        through each goal in order.

        The truck's pose is updated after each leg so the next leg starts
        from where the previous one ended.

        Parameters
        ----------
        goals  : ordered list of (x, y, theta_deg) goal poses
        colors : per-leg trail colors (cycles if fewer than len(goals))
        title  : window title
        """
        if not goals:
            print("run_multi_goal(): no goals provided.")
            return

        default_colors = ["#2ecc71", "#3498db", "#9b59b6", "#e67e22", "#e74c3c"]
        if colors is None:
            colors = default_colors

        # ── Plan all legs first ─────────────────────────────────────────
        legs: List[List[State]] = []
        for i, goal in enumerate(goals):
            print(f"Planning leg {i+1}/{len(goals)}: → {goal}")
            path = self.vehicle.plan(goal)
            if path is None:
                print(f"  No path found for leg {i+1}. Stopping.")
                break
            legs.append(path)
            # Update truck pose so next leg starts from here
            self.vehicle.map.firetruck_pose = (goal[0], goal[1])

        if not legs:
            print("No legs could be planned.")
            return

        # ── Set up a single figure ──────────────────────────────────────
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self._setup_axes(self.ax, title)
        self._draw_obstacles(self.ax)

        # Draw all planned trails and goal markers up front
        for i, (leg, goal) in enumerate(zip(legs, goals)):
            color = colors[i % len(colors)]
            px = [p[0] for p in leg]
            py = [p[1] for p in leg]
            self.ax.plot(px, py, "--", color=color,
                         alpha=0.3, linewidth=1.0)
            self.ax.plot(goal[0], goal[1], "*",
                         color=color, markersize=12, zorder=6,
                         label=f"Goal {i+1}")
            self._draw_heading_arrow(
                self.ax, goal,
                length=self._world_size * 0.025,
                color=color, lw=1.5,
            )

        self.ax.legend(loc="upper right", fontsize=8)

        # ── Build one concatenated dense path ───────────────────────────
        full_sim: List[State] = []
        leg_boundaries: List[int] = []   # frame indices where legs start

        for leg in legs:
            dense = self._interpolate(leg)
            leg_boundaries.append(len(full_sim))
            full_sim.extend(dense)

        if not full_sim:
            return

        # ── Initial vehicle patch ───────────────────────────────────────
        vehicle_patch = self._make_vehicle_patch(full_sim[0])
        for p in vehicle_patch:
            self.ax.add_patch(p)

        # One trace line per leg, drawn progressively
        trace_lines = []
        for i, leg in enumerate(legs):
            color = colors[i % len(colors)]
            (line,) = self.ax.plot([], [], color=color, linewidth=1.5, zorder=4)
            trace_lines.append(line)

        # Frame → leg index lookup
        def leg_of(frame: int) -> int:
            idx = 0
            for b in leg_boundaries:
                if frame >= b:
                    idx = leg_boundaries.index(b)
            return idx

        trace_data: List[Tuple[List[float], List[float]]] = [
            ([], []) for _ in legs
        ]

        def update(frame: int):
            state   = full_sim[frame]
            leg_idx = leg_of(frame)

            # Update vehicle footprint
            new_patches = self._make_vehicle_patch(state)
            for patch, new_p in zip(vehicle_patch, new_patches):
                patch.set_xy(list(new_p.get_xy()))

            # Grow current leg's trace
            trace_data[leg_idx][0].append(state[0])
            trace_data[leg_idx][1].append(state[1])
            trace_lines[leg_idx].set_data(*trace_data[leg_idx])

            return vehicle_patch + trace_lines

        self.ani = animation.FuncAnimation(
            self.fig, update,
            frames=len(full_sim),
            interval=1000 // self.fps,
            blit=True,
        )
        plt.tight_layout()
        plt.show()

    # =======================================================================
    # Internal helpers
    # =======================================================================

    def _setup_axes(self, ax: plt.Axes, title: str) -> None:
        ax.set_xlim(0, self._world_size)
        ax.set_ylim(0, self._world_size)
        ax.set_aspect("equal")
        ax.grid(True, linestyle=":", alpha=0.3)
        ax.set_title(title)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

    def _draw_obstacles(self, ax: plt.Axes) -> None:
        """
        Draw obstacles from map.obstacle_set.
        FIX: original used obstacle_coordinate_dict which doesn't exist on Map.
        """
        for row, col in self.vehicle.map.obstacle_set:
            world_x = row * self._cell_size
            world_y = col * self._cell_size
            rect = patches.Rectangle(
                (world_x, world_y),
                self._cell_size, self._cell_size,
                color="dimgray", alpha=0.8, zorder=2,
            )
            ax.add_patch(rect)

    def _draw_heading_arrow(
        self,
        ax: plt.Axes,
        state: State,
        length: float,
        color: str,
        lw: float = 1.5,
    ) -> None:
        rad = math.radians(state[2])
        ax.annotate(
            "",
            xy=(state[0] + length * math.cos(rad),
                state[1] + length * math.sin(rad)),
            xytext=(state[0], state[1]),
            arrowprops=dict(arrowstyle="->", color=color, lw=lw),
            zorder=7,
        )

    def _interpolate(self, path: List[State]) -> List[State]:
        """
        The Dubins path from plan() is already dense (0.5m waypoints).
        This method linearly fills between consecutive waypoints at the
        fps rate so the animation runs at a consistent speed.

        FIX from original:
          - No hardcoded velocity assumption; step size from fps.
          - Heading interpolation uses shortest-angle arithmetic
            (fixes 359° → 1° flip that caused the truck to spin).
          - Works with 3-tuple (x, y, theta) — no trailer logic needed here.
        """
        if len(path) < 2:
            return list(path)

        dt          = 1.0 / self.fps
        speed       = self.vehicle.car.v_max          # m/s from CarModel
        step_dist   = speed * dt
        smooth: List[State] = []

        for i in range(len(path) - 1):
            p1, p2 = path[i], path[i + 1]
            dist   = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            steps  = max(1, int(dist / step_dist))

            for s in range(steps):
                ratio = s / steps
                ix    = p1[0] + (p2[0] - p1[0]) * ratio
                iy    = p1[1] + (p2[1] - p1[1]) * ratio

                # Shortest-angle heading interpolation
                diff  = (p2[2] - p1[2] + 180.0) % 360.0 - 180.0
                itheta = (p1[2] + diff * ratio) % 360.0

                smooth.append((ix, iy, itheta))

        smooth.append(path[-1])
        return smooth

    def _make_vehicle_patch(
        self, state: State
    ) -> List[patches.Polygon]:
        """
        Build matplotlib Polygon patches for the vehicle footprint.
        Calls car.footprint_at() from CarModel — returns a single Shapely Polygon.
        """
        car      = self.vehicle.car
        footprint = car.footprint_at(state[0], state[1], state[2])

        # footprint_at returns a single Shapely Polygon
        geoms = (
            [footprint]
            if footprint.geom_type == "Polygon"
            else list(footprint.geoms)
        )

        result = []
        colors     = ["cyan", "yellow"]
        edgecolors = ["blue", "orange"]
        for i, geom in enumerate(geoms):
            p = patches.Polygon(
                list(geom.exterior.coords),
                facecolor=colors[i % len(colors)],
                edgecolor=edgecolors[i % len(edgecolors)],
                alpha=0.9,
                zorder=10,
            )
            result.append(p)

        return result
