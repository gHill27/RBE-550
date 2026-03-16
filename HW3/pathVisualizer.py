import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation


from shapely.geometry import box
from shapely.affinity import rotate, translate
from shapely.ops import unary_union
import math
from typing import List



class PlannerVisualizer:
    def __init__(
        self,
        vechile_size: tuple[float, float],
        title="Live State Lattice Planner",
        grid_size=36,
        vehicle = None
    ):
        plt.ion()  # Turn on interactive mode
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.grid_size = grid_size
        self.title = title
        self.vehicle = vehicle

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

    def _create_vehicle_polygon(self, Pose):
        if self.vehicle:
            # We pass the coordinates to the Truck's get_footprint method
            x,y,t, = Pose[:3]
            t1 = Pose[3] if len(Pose) > 3 else t
            return self.vehicle.get_footprint(x,y,t,t1)
        else:
            rect = box(
                -self.v_width / 2, -self.v_height / 2, self.v_width / 2, self.v_height / 2
            )
            rotated = rotate(rect, Pose[2], origin=(0, 0))
            return translate(rotated, Pose[0],Pose[1])

    def show_goal_with_arrow(self, goal_state):
        """
        Draws the goal position with an arrow indicating the required heading.
        goal_state: (x, y, theta_degrees)
        """
        gx, gy, gtheta = goal_state[:3]

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
        polys = self._create_vehicle_polygon(current_pos)
        if not isinstance(polys,List):
            polys = [polys]
        for poly in polys:
            # Handle both Polygon (single car) and MultiPolygon (Truck + Trailer)
            if poly.geom_type == 'Polygon':
                geoms = [poly]
            else:
                # This extracts the individual Polygons from the MultiPolygon
                geoms = list(poly.geoms)

            for p in geoms:
                ex_x, ex_y = p.exterior.xy
                self.ax.fill(ex_x, ex_y, color="cyan", alpha=0.8, edgecolor="blue")

        self.show_goal_with_arrow(goal)
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
            # Unpack x, y regardless of whether path is 3D or 4D
            px = [state[0] for state in path]
            py = [state[1] for state in path]
            self.ax.plot(px, py, color="blue", linewidth=2, marker=".", zorder=10)

            # Draw the footprints for the final path one last time for clarity
            for i, state in enumerate(path):
                # Draw every 5th footprint to keep it readable
                if i % 5 == 0 or i == len(path) - 1:
                    polys = self._create_vehicle_polygon(state)
                    if not isinstance(polys,List):
                        polys = [polys]
                    for poly in polys:
                        geoms = [poly] if poly.geom_type == 'Polygon' else poly.geoms
                        alpha = 0.05 if i < len(path) - 1 else 0.8
                        for p in geoms:
                            ex_x, ex_y = p.exterior.xy
                            self.ax.fill(ex_x, ex_y, color="cyan", alpha=alpha, edgecolor="blue")

        print("Planning Complete. Close the window to end the program.")
        plt.show()  # This is the "blocking" call that holds the grid open

    