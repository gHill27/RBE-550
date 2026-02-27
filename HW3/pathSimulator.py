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

    def _interpolate(self, velocity: float) -> List[Tuple[float, float, float]]:
        """Generates smooth sub-steps between sparse A* nodes."""
        dt = 1.0 / self.fps
        step_dist = velocity * dt
        smooth_path = []

        for i in range(len(self.original_path) - 1):
            p1, p2 = self.original_path[i], self.original_path[i+1]
            dist = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
            
            # Determine how many frames this segment needs
            num_steps = max(1, int(dist / step_dist))
            
            for n in range(num_steps):
                ratio = n / num_steps
                ix = p1[0] + (p2[0] - p1[0]) * ratio
                iy = p1[1] + (p2[1] - p1[1]) * ratio
                
                # Correctly handle 0/360 degree wrap-around during interpolation
                angle_diff = (p2[2] - p1[2] + 180) % 360 - 180
                itheta = (p1[2] + angle_diff * ratio) % 360
                smooth_path.append((ix, iy, itheta))
        
        smooth_path.append(self.original_path[-1]) # Ensure we hit the goal
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
        for row, col in self.vehicle.map.obstacle_coordinate_list:
            world_x, world_y = row * cell_size, col * cell_size
            rect = patches.Rectangle(
                (world_x, world_y), cell_size, cell_size, color="dimgray", alpha=0.8
            )
            self.ax.add_patch(rect)

        # 2. Draw the ghost of the planned path
        px, py, _ = zip(*self.original_path)
        self.ax.plot(px, py, 'b--', alpha=0.3, label="Planned Path")

        # 3. Create the High-Res Path
        sim_path = self._interpolate(velocity)

        # 4. Initialize Vehicle Patch using Vehicle dimensions
        start_state = sim_path[0]
        # Use the vehicle's own footprint logic
        footprint = self.vehicle.get_footprint(*start_state)
        vehicle_patch = patches.Polygon(
            list(footprint.exterior.coords), 
            facecolor='cyan', edgecolor='blue', alpha=0.9, zorder=10
        )
        self.ax.add_patch(vehicle_patch)

        def update(frame):
            state = sim_path[frame]
            # Get updated footprint for the current interpolated state
            new_footprint = self.vehicle.get_footprint(*state)
            vehicle_patch.set_xy(list(new_footprint.exterior.coords))
            return vehicle_patch,

        self.ani = animation.FuncAnimation(
            self.fig, update, frames=len(sim_path), interval=1000//self.fps, blit=True
        )
        
        plt.legend()
        plt.show()