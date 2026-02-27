import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation


from shapely.geometry import box
from shapely.affinity import rotate, translate
import math
from typing import List



class PlannerVisualizer:
    def __init__(
        self,
        vechile_size: tuple[float, float],
        title="Live State Lattice Planner",
        grid_size=36,
    ):
        plt.ion()  # Turn on interactive mode
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.grid_size = grid_size
        self.title = title

        # Setup the static grid once
        self.ax.set_xlim(0, self.grid_size)
        self.ax.set_ylim(0, self.grid_size)
        #self.ax.invert_yaxis()
        #self.ax.xaxis.tick_top()
        self.ax.set_aspect("equal")
        self.ax.grid(True, linestyle=":", alpha=0.5)

        # Robot dimensions
        self.v_width, self.v_height = (
            vechile_size  # will be filled later by /change_vechile_size
        )

    def _create_vehicle_polygon(self, x, y, theta):
        rect = box(
            -self.v_width / 2, -self.v_height / 2, self.v_width / 2, self.v_height / 2
        )
        rotated = rotate(rect, theta, origin=(0, 0))
        return translate(rotated, x, y)

    def show_goal_with_arrow(self, goal_state):
        """
        Draws the goal position with an arrow indicating the required heading.
        goal_state: (x, y, theta_degrees)
        """
        gx, gy, gtheta = goal_state

        # Convert degrees to radians for math functions
        rad = math.radians(gtheta)
        dx = math.cos(rad)
        dy = math.sin(rad)

        # Draw the goal point
        plt.plot(gx, gy, "go", markersize=10, label="Goal")

        # Draw the heading arrow (Quiver)
        # pivot='middle' puts the center of the arrow on the coordinate
        plt.quiver(gx, gy, dx, dy, color="green", scale=10, width=0.015, pivot="middle")

    def update(self, current_pos, costHistory, obstacles, goal):
        """Refreshes the plot with current progress."""
        self.ax.clear()  # Clear for refresh (re-setup static elements)

        # Re-apply static settings after clear
        self.ax.set_xlim(0, self.grid_size)
        self.ax.set_ylim(0, self.grid_size)
        #self.ax.xaxis.tick_top()
        self.ax.set_aspect("equal")
        self.ax.grid(True, linestyle=":", alpha=0.3)

        # 1. Draw Obstacles
        cell_size = 3  # Scale factor
        for row, col in obstacles:
            # Scale grid index (row, col) to world meters
            world_x = row * cell_size
            world_y = col * cell_size

            # Draw a 3x3 square instead of a 1x1 square
            rect = patches.Rectangle(
                (world_x, world_y), cell_size, cell_size, color="dimgray"
            )
            self.ax.add_patch(rect)

        # 2. Draw Explored Nodes (Lattice)
        all_nodes = list(costHistory.keys())
        if all_nodes:
            xs = [n[0] for n in all_nodes]
            ys = [n[1] for n in all_nodes]
            self.ax.scatter(xs, ys, c="orange", s=1, alpha=0.5)

        # 3. Draw Current Vehicle Position
        cx, cy, ct = current_pos
        poly = self._create_vehicle_polygon(cx, cy, ct)

        self.show_goal_with_arrow(goal)
        ex_x, ex_y = poly.exterior.xy
        self.ax.fill(ex_x, ex_y, color="cyan", alpha=0.8, edgecolor="blue")

        # 4. Draw Goal
        self.ax.plot(goal[0], goal[1], "ro", markersize=10)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)  # Small pause to let the window update

    def show_final(self, path, costHistory, obstacles, goal):
        """Final draw that stops the script from closing."""
        plt.ioff()  # Turn off interactive mode
        self.update(path[-1], costHistory, obstacles, goal)  # Update to last state

        # Now draw the full completed blue path over the final frame
        if path:
            px, py, _ = zip(*path)
            self.ax.plot(px, py, color="blue", linewidth=2, marker=".", zorder=10)

            # Draw the footprints for the final path one last time for clarity
            for i, (x, y, theta) in enumerate(path):
                poly = self._create_vehicle_polygon(x, y, theta)
                ex_x, ex_y = poly.exterior.xy
                alpha = 0.05 if i < len(path) - 1 else 0.8
                self.ax.fill(ex_x, ex_y, color="cyan", alpha=alpha, edgecolor="blue")

        print("Planning Complete. Close the window to end the program.")
        plt.show()  # This is the "blocking" call that holds the grid open

    