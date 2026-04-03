"""
pathVisualizer.py
=================
Real-time matplotlib display for the Wildfire simulation.

Consumed data (all read directly from the Map object each frame):
  map.obstacle_coordinate_dict  – {(row,col): {'status': Status, ...}}
  map.sim_time                  – float, seconds
  map.firetruck_pose            – (x_m, y_m, theta_deg)   world-metres
  map.wumpus_pose               – (x_m, y_m)              world-metres
  map.cell_size                 – metres per grid cell
  map.grid_num                  - number of cells per side

Paths are passed in per-frame:
  firetruck_path  - list of (x_m, y_m, theta_deg) States, or None
  wumpus_path     - list of (row, col) grid tuples, or None

Usage
-----
    from pathVisualizer import SimVisualizer

    viz = SimVisualizer(map_obj)          # create once
    viz.update(firetruck_path, wumpus_path)   # call every sim tick
    # At end:
    viz.close()

Call viz.update() as fast as your sim loop runs; matplotlib's
non-blocking draw (pause(0.001)) keeps the window responsive.

╔══════════════════════════════════════════════════════════════════════════╗
║                          AI USAGE DISCLOSURE                             ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Tool      : Claude (Anthropic) — claude-sonnet-4-6                      ║
║  Role      : Implementation and debugging partner                        ║
║  Scope     : Partially AI-assisted                                       ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Contributions                                                           ║
║  ─ Advised on matplotlib animation structure (FuncAnimation vs manual    ║
║    canvas flush) for smooth real-time updates without blocking.          ║
║  ─ Suggested PlannerVisualizer.plot_prm() signature                      ║
║    (map, graph, nodes, path=None) to allow the engine to trigger PRM     ║
║    debug rendering after build_tree() with a single call.                ║
║  ─ Recommended drawing PRM edges as LineCollection for O(edges) render   ║
║    instead of one plt.plot call per edge.                                ║
╠══════════════════════════════════════════════════════════════════════════╣
"""


from __future__ import annotations

import math
from typing import List, Optional, Tuple

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon, RegularPolygon
from matplotlib.lines import Line2D
import threading

# Use a non-blocking backend that works both in scripts and notebooks.
matplotlib.use("TkAgg")   # swap to "Qt5Agg" if TkAgg is unavailable

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
_C = {
    "bg":           "#0d0d0f",          # near-black canvas
    "grid_line":    "#1e2030",          # subtle grid
    "empty_cell":   "#12151e",          # open ground
    "intact":       "#2d5a1b",          # dark forest green
    "burning":      "#e05c00",          # deep orange
    "extinguished": "#3a6b8a",          # slate blue
    "burned":       "#5e5c5c",          # charcoal
    "truck_body":   "#d4a843",          # amber
    "truck_cabin":  "#f0c040",          # bright gold
    "truck_outline":"#ffffff",
    "wumpus":       "#e8003d",          # vivid red
    "wumpus_path":  "#ff6b6b",
    "truck_path":   "#43d4c8",          # cyan-teal
    "goal_marker":  "#ff3cac",          # hot pink
    "text_primary": "#e8eaf0",
    "text_dim":     "#6b7280",
    "panel_bg":     "#0a0c12",
    "hud_border":   "#2a2d3e",
}

_STATUS_COLOR = {
    "INTACT":       _C["intact"],
    "BURNING":      _C["burning"],
    "EXTINGUISHED": _C["extinguished"],
    "BURNED":       _C["burned"],
}

# Physical truck dimensions (metres) — must match CarModel in firetruck_prm.py
_TRUCK_LENGTH   = 4.9
_TRUCK_WIDTH    = 2.2
_TRUCK_WHEELBASE = 3.0
_TRUCK_REAR_OVERHANG = (_TRUCK_LENGTH - _TRUCK_WHEELBASE) / 2.0

# Wumpus star size in metres
_WUMPUS_RADIUS = 2.5


# ===========================================================================
# SimVisualizer
# ===========================================================================

class SimVisualizer:
    """
    Persistent matplotlib figure that redraws on every call to update().

    Parameters
    ----------
    map_obj : Map
        Live reference to the Map instance.  The visualizer reads its
        fields directly — nothing is copied at construction time.
    figsize : tuple
        Figure size in inches (width, height).
    """

    def __init__(self, map_obj, figsize: Tuple[int, int] = (14, 10)):
        self.map = map_obj

        cs  = map_obj.cell_size
        gn  = map_obj.grid_num
        self._world = gn * cs          # world size in metres
        self._cs    = cs
        self._gn    = gn

        # --- Figure layout: main map + right-side HUD panel ---------------
        self._fig = plt.figure(
            figsize=figsize,
            facecolor=_C["bg"],
            linewidth=0,
        )
        self._fig.canvas.manager.set_window_title("Wildfire Simulation")

        # GridSpec: map takes 80% width, HUD panel 20%
        gs = self._fig.add_gridspec(
            1, 2,
            width_ratios=[4, 1],
            left=0.02, right=0.98,
            top=0.97, bottom=0.03,
            wspace=0.04,
        )
        self._ax  = self._fig.add_subplot(gs[0])    # main map
        self._hud = self._fig.add_subplot(gs[1])    # info panel

        self._setup_axes()

        # --- Reusable artists (updated in-place each frame) ---------------
        # Obstacle patches stored in a dict keyed by (row,col)
        self._obstacle_patches: dict = {}

        # Path lines
        self._truck_path_line, = self._ax.plot(
            [], [], color=_C["truck_path"], lw=1.4,
            alpha=0.75, zorder=4, linestyle="--",
        )
        self._wumpus_path_line, = self._ax.plot(
            [], [], color=_C["wumpus_path"], lw=1.4,
            alpha=0.75, zorder=4, linestyle=":",
        )

        # Goal marker (crosshair)
        self._goal_scatter = self._ax.scatter(
            [], [], marker="X", s=180,
            color=_C["goal_marker"], zorder=8, linewidths=1.5,
            edgecolors=_C["bg"],
        )

        # Truck body (rotated rectangle drawn as a Polygon)
        self._truck_patch = mpatches.FancyBboxPatch(
            (0, 0), 1, 1,
            boxstyle="round,pad=0.05",
            facecolor=_C["truck_body"],
            edgecolor=_C["truck_outline"],
            linewidth=1.2,
            zorder=7,
        )
        self._ax.add_patch(self._truck_patch)

        # Heading arrow on truck
        self._truck_arrow = FancyArrowPatch(
            (0, 0), (1, 0),
            arrowstyle="-|>",
            color=_C["truck_outline"],
            mutation_scale=10,
            linewidth=1.2,
            zorder=8,
        )
        self._ax.add_patch(self._truck_arrow)

        # Wumpus star
        self._wumpus_star = RegularPolygon(
            (0, 0), numVertices=5,
            radius=_WUMPUS_RADIUS,
            orientation=math.pi / 2,
            facecolor=_C["wumpus"],
            edgecolor="#ffffff",
            linewidth=0.8,
            zorder=7,
        )
        self._ax.add_patch(self._wumpus_star)

        # Legend
        self._build_legend()

        plt.ion()
        self._fig.canvas.draw()
        plt.pause(0.001)

    # ------------------------------------------------------------------
    # Axis setup
    # ------------------------------------------------------------------

    def _setup_axes(self):
        ax = self._ax
        ax.set_facecolor(_C["empty_cell"])
        ax.set_xlim(0, self._world)
        ax.set_ylim(0, self._world)
        ax.set_aspect("equal")
        ax.tick_params(colors=_C["text_dim"], labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor(_C["hud_border"])

        # Draw grid lines once
        for i in range(self._gn + 1):
            v = i * self._cs
            ax.axhline(v, color=_C["grid_line"], lw=0.4, zorder=0)
            ax.axvline(v, color=_C["grid_line"], lw=0.4, zorder=0)

        ax.set_xlabel("X  (metres)", color=_C["text_dim"], fontsize=8)
        ax.set_ylabel("Y  (metres)", color=_C["text_dim"], fontsize=8)

        # HUD panel
        hud = self._hud
        hud.set_facecolor(_C["panel_bg"])
        hud.set_xticks([])
        hud.set_yticks([])
        for spine in hud.spines.values():
            spine.set_edgecolor(_C["hud_border"])

    def _build_legend(self):
        legend_items = [
            mpatches.Patch(facecolor=_C["intact"],       label="Intact"),
            mpatches.Patch(facecolor=_C["burning"],      label="Burning"),
            mpatches.Patch(facecolor=_C["extinguished"], label="Extinguished"),
            mpatches.Patch(facecolor=_C["burned"],       label="Burned"),
            Line2D([0], [0], color=_C["truck_path"],  lw=1.5, ls="--", label="Truck path"),
            Line2D([0], [0], color=_C["wumpus_path"], lw=1.5, ls=":",  label="Wumpus path"),
            mpatches.Patch(facecolor=_C["goal_marker"],  label="Goal"),
        ]
        self._ax.legend(
            handles=legend_items,
            loc="upper left",
            fontsize=7,
            framealpha=0.6,
            facecolor=_C["panel_bg"],
            edgecolor=_C["hud_border"],
            labelcolor=_C["text_primary"],
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        firetruck_path: Optional[List[Tuple]] = None,
        wumpus_path:    Optional[List[Tuple]] = None,
    ) -> None:
        """
        Redraw the entire scene.  Call once per simulation tick.

        Parameters
        ----------
        firetruck_path : list of (x_m, y_m, theta_deg) or None
        wumpus_path    : list of (row, col) grid tuples or None
        """
        if threading.current_thread() is not threading.main_thread():
            self._fig.canvas.get_tk_widget().after(
                0, lambda: self.update(firetruck_path, wumpus_path)
            )
            return

        self._draw_obstacles()
        self._draw_truck_path(firetruck_path)
        self._draw_wumpus_path(wumpus_path)
        self._draw_truck()
        self._draw_wumpus()
        self._draw_goal()
        self._draw_hud()

        self._fig.canvas.draw_idle()
        plt.pause(0.001)

    def close(self, reason: str = "time_limit") -> None:
        if threading.current_thread() is not threading.main_thread():
            # Schedule closure on main thread instead of calling directly
            self._fig.canvas.get_tk_widget().after(0, lambda: self.close(reason))
            return
        if reason == "wumpus_caught":
            self.show_end_screen(reason)
        else:
            plt.ioff()
            plt.show()

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_obstacles(self):
        """
        Sync obstacle patches with map.obstacle_coordinate_dict.

        Burned cells now remain in obstacle_coordinate_dict permanently
        (Map._delete_obstacle no longer removes them) so they stay
        visible as charcoal patches for the full simulation.

        Only patches whose coords have been fully removed from the dict
        (which should not happen in normal operation) are cleaned up.
        """
        ax = self._ax
        cs = self._cs
        current_coords = set(self.map.obstacle_coordinate_dict.keys())
        cached_coords  = set(self._obstacle_patches.keys())

        # Clean up any patches whose coord was removed from the dict
        # (defensive — under the fixed Map this should never trigger)
        for coord in cached_coords - current_coords:
            self._obstacle_patches[coord].remove()
            del self._obstacle_patches[coord]

        # Add new or update existing patches
        for coord, data in self.map.obstacle_coordinate_dict.items():
            status_name = data["status"].name   # "INTACT","BURNING","EXTINGUISHED","BURNED"
            color       = _STATUS_COLOR.get(status_name, _C["intact"])
            row, col    = coord
            x0 = row * cs
            y0 = col * cs

            if coord not in self._obstacle_patches:
                patch = mpatches.Rectangle(
                    (x0, y0), cs, cs,
                    facecolor=color,
                    edgecolor=_C["grid_line"],
                    linewidth=0.4,
                    zorder=2,
                )
                ax.add_patch(patch)
                self._obstacle_patches[coord] = patch

                if status_name == "BURNING":
                    self._add_fire_glow(ax, x0, y0, cs)
            else:
                patch = self._obstacle_patches[coord]
                patch.set_facecolor(color)

                if status_name == "BURNING":
                    # Pulse alpha for drama — faster pulse as fire intensifies
                    elapsed = (self.map.sim_time
                               - (data.get("burn_time") or self.map.sim_time))
                    rate  = 3.0 + min(elapsed * 0.2, 5.0)
                    pulse = 0.55 + 0.35 * math.sin(self.map.sim_time * rate)
                    patch.set_alpha(pulse)
                elif status_name == "BURNED":
                    # Charcoal with slight transparency so the grid shows through
                    patch.set_alpha(0.75)
                else:
                    patch.set_alpha(1.0)

    def _add_fire_glow(self, ax, x0, y0, cs):
        """One-time halo patch drawn behind a burning cell."""
        pad = cs * 0.25
        glow = mpatches.Rectangle(
            (x0 - pad, y0 - pad),
            cs + 2 * pad, cs + 2 * pad,
            facecolor="#ff6600",
            edgecolor="none",
            alpha=0.18,
            zorder=1,
        )
        ax.add_patch(glow)

    def _draw_truck_path(self, path):
        if not path:
            self._truck_path_line.set_data([], [])
            return
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        self._truck_path_line.set_data(xs, ys)

    def _draw_wumpus_path(self, path):
        if not path:
            self._wumpus_path_line.set_data([], [])
            return
        cs = self._cs
        # Wumpus path is in grid (row,col); convert to world centre coords
        xs = [p[0] * cs + cs / 2 for p in path]
        ys = [p[1] * cs + cs / 2 for p in path]
        self._wumpus_path_line.set_data(xs, ys)

    def _draw_truck(self):
        """
        Draw the firetruck as a rotated rectangle with a heading arrow.
        The reference point is the rear axle centre (firetruck_prm convention).
        """
        pose = self.map.firetruck_pose
        if pose is None:
            return

        x, y, theta_deg = float(pose[0]), float(pose[1]), float(pose[2])
        theta_rad = math.radians(theta_deg)

        # Remove old truck patch and re-add as a proper rotated polygon
        # Build footprint corners in local frame (rear axle = origin)
        hl = _TRUCK_WHEELBASE + _TRUCK_REAR_OVERHANG   # total forward extent
        hr = _TRUCK_REAR_OVERHANG                        # rear extent
        hw = _TRUCK_WIDTH / 2.0

        corners_local = np.array([
            [-hr,  -hw],
            [-hr,   hw],
            [ hl,   hw],
            [ hl,  -hw],
        ])

        # Rotate and translate
        c, s = math.cos(theta_rad), math.sin(theta_rad)
        R = np.array([[c, -s], [s, c]])
        corners_world = (R @ corners_local.T).T + np.array([x, y])

        # Update the truck patch as a polygon
        self._truck_patch.set_visible(False)   # hide the placeholder box
        if hasattr(self, "_truck_poly"):
            self._truck_poly.set_xy(corners_world)
            self._truck_poly.set_visible(True)
        else:
            self._truck_poly = plt.Polygon(
                corners_world,
                closed=True,
                facecolor=_C["truck_body"],
                edgecolor=_C["truck_outline"],
                linewidth=1.4,
                zorder=7,
            )
            self._ax.add_patch(self._truck_poly)

        # Heading arrow: rear axle → front axle
        nose_x = x + (hl) * c
        nose_y = y + (hl) * s
        self._truck_arrow.set_positions((x, y), (nose_x, nose_y))

        # Cabin highlight (front quarter of the truck, brighter colour)
        cabin_depth = _TRUCK_WIDTH * 0.6
        cabin_local = np.array([
            [hl - cabin_depth, -hw],
            [hl - cabin_depth,  hw],
            [hl,                hw],
            [hl,               -hw],
        ])
        cabin_world = (R @ cabin_local.T).T + np.array([x, y])
        if hasattr(self, "_truck_cabin"):
            self._truck_cabin.set_xy(cabin_world)
        else:
            self._truck_cabin = plt.Polygon(
                cabin_world,
                closed=True,
                facecolor=_C["truck_cabin"],
                edgecolor="none",
                alpha=0.7,
                zorder=8,
            )
            self._ax.add_patch(self._truck_cabin)

    def _draw_wumpus(self):
        """
        Draw the wumpus as a 5-pointed red star at wumpus_pose.
        wumpus_pose is stored in world metres on the Map.
        """
        pose = self.map.wumpus_pose
        if pose is None:
            return
        wx, wy = float(pose[0]), float(pose[1])
        self._wumpus_star.xy = (wx, wy)

    def _draw_goal(self):
        """
        Draw the current firetruck goal (from map.firetruck_goal) as a
        hot-pink ✕ marker.
        """
        
        goal = self.map.firetruck_goal
        if goal is None:
            self._goal_scatter.set_offsets(np.empty((0, 2)))
            return
        gx, gy = float(goal[0]), float(goal[1])
        self._goal_scatter.set_offsets([[gx, gy]])

    def _draw_hud(self):
        """Right-side panel: sim clock + live statistics."""
        hud = self._hud
        hud.cla()
        hud.set_facecolor(_C["panel_bg"])
        hud.set_xticks([])
        hud.set_yticks([])
        for spine in hud.spines.values():
            spine.set_edgecolor(_C["hud_border"])

        t = self.map.sim_time
        mins  = int(t) // 60
        secs  = int(t) % 60
        msecs = int((t % 1) * 10)

        # Count obstacle statuses
        counts = {"INTACT": 0, "BURNING": 0, "EXTINGUISHED": 0, "BURNED": 0}
        for data in self.map.obstacle_coordinate_dict.values():
            counts[data["status"].name] += 1

        # Active fires
        n_fires = len(self.map.active_fires)

        lines = [
            ("SIM TIME",         f"{mins:02d}:{secs:02d}.{msecs}", "#e8eaf0", 14, "bold"),
            ("",                 "",                                 "#444",    8,  "normal"),
            ("OBSTACLES",        "",                                 _C["text_dim"], 8, "bold"),
            ("  Intact",         str(counts["INTACT"]),             _C["intact"],   9, "normal"),
            ("  Burning",        str(counts["BURNING"]),            _C["burning"],  9, "normal"),
            ("  Extinguished",   str(counts["EXTINGUISHED"]),       _C["extinguished"], 9, "normal"),
            ("  Burned",         str(counts["BURNED"]),             _C["burned"] if counts["BURNED"] == 0 else "#666", 9, "normal"),
            ("",                 "",                                 "#444",    8,  "normal"),
            ("ACTIVE FIRES",     str(n_fires),                      "#e05c00" if n_fires > 0 else _C["text_dim"], 10, "bold"),
        ]

        # Add goal info
        goal = self.map.firetruck_goal
        if goal:
            lines += [
                ("",              "",                                 "#444", 8, "normal"),
                ("TRUCK GOAL",    f"({goal[0]:.1f}, {goal[1]:.1f})", _C["goal_marker"], 8, "normal"),
            ]

        y_pos = 0.97
        for label, value, color, size, weight in lines:
            if label == "":
                y_pos -= 0.025
                continue
            # Label on left
            hud.text(
                0.05, y_pos, label,
                transform=hud.transAxes,
                color=_C["text_dim"],
                fontsize=size - 1,
                fontweight=weight,
                va="top",
            )
            # Value on right (if present)
            if value:
                hud.text(
                    0.95, y_pos, value,
                    transform=hud.transAxes,
                    color=color,
                    fontsize=size,
                    fontweight=weight,
                    va="top",
                    ha="right",
                )
            y_pos -= 0.065

        # Truck pose at bottom
        pose = self.map.firetruck_pose
        if pose:
            hud.text(
                0.5, 0.08,
                f"Truck\n({pose[0]:.1f}, {pose[1]:.1f})\n{pose[2]:.1f}°",
                transform=hud.transAxes,
                color=_C["truck_body"],
                fontsize=7.5,
                ha="center", va="bottom",
            )

        wp = self.map.wumpus_pose
        if wp:
            hud.text(
                0.5, 0.02,
                f"Wumpus  ({wp[0]:.1f}, {wp[1]:.1f})",
                transform=hud.transAxes,
                color=_C["wumpus"],
                fontsize=7,
                ha="center", va="bottom",
            )
    
    def show_end_screen(self, reason: str = "wumpus_caught") -> None:
        """
        Overlay a full-canvas end screen, pause for 1 second, then close.
        Call this instead of close() when the sim ends with a notable result.
        """
        ax = self._ax

        # Semi-transparent dark overlay
        overlay = mpatches.Rectangle(
            (0, 0), self._world, self._world,
            facecolor="#0d0d0f",
            alpha=0.82,
            zorder=20,
            transform=ax.transData,
        )
        ax.add_patch(overlay)

        cx = self._world / 2
        cy = self._world / 2

        if reason == "wumpus_caught":
            headline = "WUMPUS CAUGHT!"
            subline  = "Firetruck wins this round"
            color    = _C["truck_body"]       # amber
        elif reason == "time_limit":
            headline = "TIME LIMIT REACHED"
            subline  = "Simulation complete"
            color    = _C["text_primary"]
        else:
            headline = "SIMULATION COMPLETE"
            subline  = reason
            color    = _C["text_primary"]

        ax.text(
            cx, cy + self._world * 0.07,
            headline,
            color=color,
            fontsize=22,
            fontweight="bold",
            ha="center", va="center",
            zorder=21,
        )
        ax.text(
            cx, cy - self._world * 0.04,
            subline,
            color=_C["text_dim"],
            fontsize=13,
            ha="center", va="center",
            zorder=21,
        )

        self._fig.canvas.draw()
        plt.pause(1.0)      # display for 1 second then continue
        plt.close(self._fig)


# ===========================================================================
# PlannerVisualizer  (kept for compatibility with firetruck_prm.py)
# ===========================================================================

class PlannerVisualizer:
    """
    Lightweight wrapper used by Firetruck during the build/debug phase
    (firetruck_prm.py passes (width, length) to the constructor).
    Plots the static PRM roadmap — not used during live simulation.
    """

    def __init__(self, car_dims: Tuple[float, float]):
        self.car_width, self.car_length = car_dims

    def plot_prm(self, map_obj, graph, nodes, path=None):
        cs = map_obj.cell_size
        gn = map_obj.grid_num
        world = gn * cs

        fig, ax = plt.subplots(figsize=(10, 10))
        fig.patch.set_facecolor(_C["bg"])
        ax.set_facecolor(_C["empty_cell"])
        ax.set_xlim(0, world)
        ax.set_ylim(0, world)
        ax.set_aspect("equal")
        ax.set_title("PRM Roadmap", color=_C["text_primary"], fontsize=12)

        # Grid
        for i in range(gn + 1):
            v = i * cs
            ax.axhline(v, color=_C["grid_line"], lw=0.3)
            ax.axvline(v, color=_C["grid_line"], lw=0.3)

        # Obstacles
        for (row, col), data in map_obj.obstacle_coordinate_dict.items():
            color = _STATUS_COLOR.get(data["status"].name, _C["intact"])
            ax.add_patch(mpatches.Rectangle(
                (row * cs, col * cs), cs, cs,
                facecolor=color, edgecolor=_C["grid_line"], lw=0.3, zorder=2,
            ))

        # PRM edges (thin, dimmed)
        for i, edges in graph.items():
            xi, yi, _ = nodes[i]
            for e in edges:
                j = e["to"] if isinstance(e, dict) else e.node_to
                xj, yj, _ = nodes[j]
                ax.plot([xi, xj], [yi, yj],
                        color=_C["text_dim"], lw=0.3, alpha=0.3, zorder=3)

        # Nodes
        xs = [n[0] for n in nodes]
        ys = [n[1] for n in nodes]
        ax.scatter(xs, ys, s=6, color=_C["truck_path"], zorder=4, alpha=0.7)

        # Solution path
        if path:
            px = [p[0] for p in path]
            py = [p[1] for p in path]
            ax.plot(px, py, color=_C["truck_body"], lw=2, zorder=6)

        plt.tight_layout()
        plt.show(block = True)

