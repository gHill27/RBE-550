import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import math
from typing import List, Tuple


class PathSimulator:
    def __init__(self, vehicle, path: List[Tuple[float, float, float]], fps: int = 30):
        self.vehicle = vehicle
        self.original_path = path
        self.fps = fps
        self.fig, self.ax = plt.subplots(figsize=(10, 10))

        # Simulation Settings
        self.grid_size = 36  # Matches your world box
        self.ani = None
        self.patches = []

    def _interpolate(self, velocity: float) -> List[Tuple[float, float, float]]:
        """Generates smooth sub-steps between sparse A* nodes."""
        dt = 1.0 / self.fps
        step_dist = velocity * dt
        smooth_path = []

        for i in range(len(self.original_path) - 1):
            p1, p2 = self.original_path[i], self.original_path[i + 1]
            dist = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

            # Determine how many frames this segment needs
            # Ensure p1 and p2 have t1 (trailer theta)
            # If path is 3D (car), t1 defaults to t0
            t1_start = p1[3] if len(p1) > 3 else p1[2]
            t1_end = p2[3] if len(p2) > 3 else p2[2]
            num_steps = max(1, int(dist / step_dist))

            for n in range(num_steps):
                ratio = n / num_steps
                # 1. Interpolate the Truck's hitch position (ix, iy)
                ix = p1[0] + (p2[0] - p1[0]) * ratio
                iy = p1[1] + (p2[1] - p1[1]) * ratio

                # 2. Interpolate Headings
                diff0 = (p2[2] - p1[2] + 180) % 360 - 180
                it0 = (p1[2] + diff0 * ratio) % 360

                diff1 = (t1_end - t1_start + 180) % 360 - 180
                it1 = (t1_start + diff1 * ratio) % 360

                # 3. GEOMETRY FIX: The trailer's (x, y) is NOT (ix, iy).
                # It is derived from the truck's hitch + trailer angle.
                # However, for drawing, your get_footprint(x, y, t0, t1)
                # likely expects (ix, iy) to be the pivot.
                # Ensure your footprint function uses (ix, iy) as the HITCH.

                smooth_path.append((ix, iy, it0, it1))

        final = self.original_path[-1]
        smooth_path.append(
            (final[0], final[1], final[2], final[3] if len(final) > 3 else final[2])
        )
        return smooth_path

    def run(self, velocity: float = 2.0):
        """Sets up and starts the animation loop."""
        self.ax.clear()
        self.ax.set_xlim(0, self.grid_size)
        self.ax.set_ylim(0, self.grid_size)
        self.ax.set_aspect("equal")
        self.ax.grid(True, linestyle=":", alpha=0.3)
        self.ax.set_title(f"Vehicle Simulation: {velocity} m/s")

        # 1. Draw Static Obstacles from the Vehicle's Map
        cell_size = 3
        for (
            row,
            col,
        ) in (
            self.vehicle.map.obstacle_coordinate_dict.keys()
        ):  ######TODO: Change color of obstacles that are burning to red
            world_x, world_y = row * cell_size, col * cell_size
            rect = patches.Rectangle(
                (world_x, world_y), cell_size, cell_size, color="dimgray", alpha=0.8
            )
            self.ax.add_patch(rect)

        # 2. Draw the ghost of the planned path
        px = [p[0] for p in self.original_path]
        py = [p[1] for p in self.original_path]
        self.ax.plot(px, py, "b--", alpha=0.3, label="Planned Path")

        # 3. Create the High-Res Path
        sim_path = self._interpolate(velocity)

        # 4. Initialize Vehicle Patch using Vehicle dimensions
        start_state = sim_path[0]
        # Use the vehicle's own footprint logic
        if len(start_state) > 3 and self.vehicle.__class__.__name__ == "Truck":
            initial_footprint = self.vehicle.get_footprint(*start_state)
        else:
            initial_footprint = self.vehicle.get_footprint(*start_state[:3])
        parts = initial_footprint

        self.patches = []
        colors = ["cyan", "yellow"]  # Cyan for Truck, Yellow for Trailer
        edges = ["blue", "orange"]
        for i, part in enumerate(parts):
            p = patches.Polygon(
                list(part.exterior.coords),
                facecolor=colors[i % len(colors)],
                edgecolor=edges[i % len(edges)],
                alpha=0.9,
                zorder=10,
            )
            self.ax.add_patch(p)
            self.patches.append(p)

        def update(frame):
            state = sim_path[frame]
            # Identify how many arguments the specific vehicle's footprint needs
            # (e.g., Car needs 3: x, y, t. Truck needs 4: x, y, t0, t1)
            arg_count = (
                self.vehicle.get_footprint.__code__.co_argcount - 1
            )  # -1 for 'self'
            footprint = self.vehicle.get_footprint(*state[:arg_count])

            # Update each polygon part (truck and trailer)
            current_geoms = footprint

            for patch, geom in zip(self.patches, current_geoms):
                patch.set_xy(list(geom.exterior.coords))

            return self.patches

        self.ani = animation.FuncAnimation(
            self.fig, update, frames=len(sim_path), interval=1000 // self.fps, blit=True
        )

        plt.legend()
        plt.show()
