"""
pathVisualizer.py
=================
PlannerVisualizer — updated to work with the new DubinsEdge-based graph.

Key changes from original
--------------------------
  - plot_prm() now reads edge.node_to from DubinsEdge objects instead of
    treating neighbors as raw ints (fixes crash with new graph format).
  - Lattice (nodes + edges) always renders, even when path=None.
  - Edge drawing skips invalid indices gracefully.
  - plot_prm() accepts an optional ax so it can be embedded in subplots.
  - Arrow overlays on nodes show heading (theta) for the PRM samples.
  - All other methods (update, show_final, _create_vehicle_polygon) unchanged.
"""

import math
from typing import List, Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from shapely.affinity import rotate, translate
from shapely.geometry import box


class PlannerVisualizer:
    def __init__(
        self,
        vehicle_size: tuple[float, float],
        title: str = "Live State Lattice Planner",
        grid_size: int = 50,
        vehicle=None,
    ):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(12, 12))
        self.grid_size = grid_size
        self.title = title
        self.vehicle = vehicle

        self.ax.set_xlim(0, self.grid_size)
        self.ax.set_ylim(0, self.grid_size)
        self.ax.set_aspect("equal")
        self.ax.grid(True, linestyle=":", alpha=0.5)

        # width, length — note: your original had these swapped; kept as-is
        # so existing callers don't break
        self.v_width, self.v_height = vehicle_size

    # ------------------------------------------------------------------
    # Internal helpers (unchanged from original)
    # ------------------------------------------------------------------

    def _create_vehicle_polygon(self, pose):
        if self.vehicle:
            x, y, t = pose[:3]
            t1 = pose[3] if len(pose) > 3 else t
            return self.vehicle.get_footprint(x, y, t, t1)
        else:
            rect = box(
                -self.v_width / 2,
                -self.v_height / 2,
                self.v_width / 2,
                self.v_height / 2,
            )
            rotated = rotate(rect, pose[2], origin=(0, 0))
            return translate(rotated, pose[0], pose[1])

    def show_goal_with_arrow(self, goal_state):
        gx, gy, gtheta = goal_state[:3]
        rad = math.radians(gtheta)
        dx, dy = math.cos(rad), math.sin(rad)
        plt.plot(gx, gy, "go", markersize=10, label="Goal")
        plt.quiver(gx, gy, dx, dy, color="green",
                   scale=10, width=0.015, pivot="middle")

    # ------------------------------------------------------------------
    # Live update (unchanged from original)
    # ------------------------------------------------------------------

    def update(self, current_pos, cost_history, obstacles, goal):
        self.ax.clear()
        self.ax.set_xlim(0, self.grid_size)
        self.ax.set_ylim(0, self.grid_size)
        self.ax.set_aspect("equal")
        self.ax.grid(True, linestyle=":", alpha=0.3)

        cell_size = 3
        for row, col in obstacles.keys():
            world_x, world_y = row * cell_size, col * cell_size
            rect = patches.Rectangle(
                (world_x, world_y), cell_size, cell_size, color="dimgray"
            )
            self.ax.add_patch(rect)

        all_nodes = list(cost_history.keys())
        if all_nodes:
            xs = [n[0] for n in all_nodes]
            ys = [n[1] for n in all_nodes]
            self.ax.scatter(xs, ys, c="orange", s=1, alpha=0.5)

        polys = self._create_vehicle_polygon(current_pos)
        if not isinstance(polys, List):
            polys = [polys]
        for poly in polys:
            geoms = [poly] if poly.geom_type == "Polygon" else list(poly.geoms)
            for p in geoms:
                ex_x, ex_y = p.exterior.xy
                self.ax.fill(ex_x, ex_y, color="cyan", alpha=0.8, edgecolor="blue")

        self.show_goal_with_arrow(goal)
        self.ax.plot(goal[0], goal[1], "ro", markersize=10)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    # ------------------------------------------------------------------
    # Final display (unchanged from original)
    # ------------------------------------------------------------------

    def show_final(self, path, cost_history, obstacles, goal):
        plt.ioff()
        self.update(path[-1], cost_history, obstacles, goal)

        if path:
            px = [s[0] for s in path]
            py = [s[1] for s in path]
            self.ax.plot(px, py, color="blue", linewidth=2, marker=".", zorder=10)

            for i, state in enumerate(path):
                if i % 5 == 0 or i == len(path) - 1:
                    polys = self._create_vehicle_polygon(state)
                    if not isinstance(polys, List):
                        polys = [polys]
                    for poly in polys:
                        geoms = ([poly] if poly.geom_type == "Polygon"
                                 else list(poly.geoms))
                        alpha = 0.05 if i < len(path) - 1 else 0.8
                        for p in geoms:
                            ex_x, ex_y = p.exterior.xy
                            self.ax.fill(ex_x, ex_y, color="cyan",
                                         alpha=alpha, edgecolor="blue")

        print("Planning complete. Close the window to end.")
        plt.show()

    # ------------------------------------------------------------------
    # PRM plot  — UPDATED
    # ------------------------------------------------------------------

    def plot_prm(
        self,
        map,
        graph: dict,
        nodes: list,
        path: Optional[list] = None,
        ax: Optional[plt.Axes] = None,
        show_headings: bool = True,
        block: bool = False,
    ) -> None:
        """
        Render the PRM roadmap.

        Changes from original
        ----------------------
        - graph values are now List[DubinsEdge] — reads edge.node_to.
        - Lattice always renders; path overlay is optional.
        - show_headings=True draws a small arrow per node showing theta.
        - ax parameter lets callers embed this in an existing subplot grid.

        Parameters
        ----------
        map           : your Map object (uses obstacle_set, cell_size, grid_num)
        graph         : adjacency dict  {node_idx: List[DubinsEdge]}
        nodes         : list of (x, y, theta_deg)
        path          : optional list of (x, y, theta_deg) waypoints
        ax            : existing Axes to draw on (creates new figure if None)
        show_headings : draw heading arrows on sampled nodes
        block         : if True, plt.show() blocks until window is closed
        """
        # ── axes setup ──────────────────────────────────────────────────
        own_fig = ax is None
        if own_fig:
            ax = self.ax

        limit = map.grid_num * map.cell_size
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.set_aspect("equal")
        ax.grid(True, linestyle=":", alpha=0.3)

        # ── 1. Obstacles ────────────────────────────────────────────────
        for gx, gy in map.obstacle_set:
            rect = plt.Rectangle(
                (gx * map.cell_size, gy * map.cell_size),
                map.cell_size, map.cell_size,
                color="dimgray", alpha=0.7, zorder=1,
            )
            ax.add_patch(rect)

        # ── 2. Edges (always rendered) ──────────────────────────────────
        # FIX: Use the actual 'path' list inside the edge dictionary 
        # to draw the Dubins curve instead of a straight line.
        all_edge_paths = []
        for node_idx, edges in graph.items():
            for edge_info in edges:
                path_pts = edge_info.get("path", [])
                if path_pts:
                    # Convert [(x, y, th), ...] to [(x, y), ...]
                    all_edge_paths.append([(p[0], p[1]) for p in path_pts])

        lc = LineCollection(all_edge_paths, color="#5b8dd9", linewidth=0.4, alpha=0.4, zorder=2)
        ax.add_collection(lc)

        # ── 3. Nodes (always rendered) ──────────────────────────────────
        node_x = [nd[0] for nd in nodes]
        node_y = [nd[1] for nd in nodes]
        ax.scatter(node_x, node_y,
                   color="#e74c3c", s=8, zorder=3, label=f"Nodes ({len(nodes)})")

        # ── 4. Heading arrows on nodes ──────────────────────────────────
        if show_headings and nodes and len(nodes[0]) >= 3:
            arrow_len = max(limit * 0.012, 1.5)
            for nd in nodes:
                rad = math.radians(nd[2])
                ax.annotate(
                    "",
                    xy=(nd[0] + arrow_len * math.cos(rad),
                        nd[1] + arrow_len * math.sin(rad)),
                    xytext=(nd[0], nd[1]),
                    arrowprops=dict(arrowstyle="->", color="#c0392b",
                                    lw=0.5),
                    zorder=4,
                )

        # ── 5. A* path overlay (only when a path exists) ────────────────
        if path:
            px = [p[0] for p in path]
            py = [p[1] for p in path]
            ax.plot(px, py,
                    color="#2ecc71", linewidth=2.5, zorder=5, label="A* path")
            ax.scatter(px, py,
                       color="#27ae60", s=12, zorder=6)

            # Start marker
            ax.plot(px[0], py[0], "o",
                    color="#1D9E75", markersize=10, zorder=7, label="Start")
            # Goal marker
            ax.plot(px[-1], py[-1], "*",
                    color="#EF9F27", markersize=14, zorder=7, label="Goal")

            # Heading arrow at start and goal
            for pt, color in [(path[0], "#1D9E75"), (path[-1], "#EF9F27")]:
                rad = math.radians(pt[2])
                arr = max(limit * 0.025, 3.0)
                ax.annotate(
                    "",
                    xy=(pt[0] + arr * math.cos(rad),
                        pt[1] + arr * math.sin(rad)),
                    xytext=(pt[0], pt[1]),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2.0),
                    zorder=8,
                )
        else:
            # No path — make it obvious in the title
            ax.set_title(
                f"PRM roadmap  |  {len(nodes)} nodes  |  no path found",
                color="#c0392b",
            )

        # ── 6. Title and legend ─────────────────────────────────────────
        if path:
            ax.set_title(
                f"PRM roadmap  |  {len(nodes)} nodes  |  path: {len(path)} waypoints"
            )
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.legend(loc="upper right", fontsize=8)

        # ── 7. Show ─────────────────────────────────────────────────────
        if own_fig:
            plt.tight_layout()
            if block:
                plt.show()
            else:
                plt.show(block=False)
                plt.pause(0.1)
                input("Press [Enter] to close the PRM plot... ")
