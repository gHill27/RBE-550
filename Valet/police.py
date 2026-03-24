from Vehicles import Vehicle, Map, State
from typing import List, Callable, Tuple
import math
from pathSimulator import PathSimulator
from math import *


class Police(Vehicle):
    def __init__(self, startPose, map, plot=None, goalPose=None):
        super().__init__(
            width=5.2,
            height=1.8,
            startState=startPose,
            goalState=goalPose,
            map=map,
            plot=plot,
        )

    def calculate_motion_primitives(self, step_distance):
        mp = {}
        L = 2.8  # wheel base in meters

        # implementing 5 drive options: Hard left, slight left, Straight, slight right, Hard right
        steering_angles = [-30, -15, 0, 15, 30, 180]
        for phi in steering_angles:
            if abs(phi) == 0:
                mp[phi] = (step_distance, 0, 0)
            elif phi == 180:
                mp[phi] = (-step_distance, 0, 0)
            else:
                # simplified integration
                d_theta = round((step_distance / L) * math.tan(math.radians(phi)), 2)
                dx = round(step_distance * math.cos(d_theta / 2), 2)
                dy = round(step_distance * math.sin(d_theta / 2), 2)
                mp[phi] = (dx, dy, round(math.degrees(d_theta), 2))
        return mp

    def get_neighbors(
        self, current_state: State, motion_primatives: dict[float, tuple[float, float]]
    ) -> List[State]:
        """
        Generates possible neighboring states
        """
        cos = math.cos(math.radians(current_state[2]))
        sin = math.sin(math.radians(current_state[2]))

        neighbors = []
        for key, value in motion_primatives.items():
            dx, dy, dtheta = value[0], value[1], value[2]

            rotated_dx = dx * cos - dy * sin
            rotated_dy = dx * sin + dy * cos
            raw_neighbor = (
                current_state[0] + rotated_dx,
                current_state[1] + rotated_dy,
                (current_state[2] + dtheta) % 360,
            )
            neighbors.append(raw_neighbor)
        return neighbors

    def main_run(self):
        # Run the planner
        import time

        start = time.time()
        path = self.plan((self.goal_state), step_distance=1)
        end = time.time()
        print(f"Time taken: {end - start} seconds")
        if path:
            print("Starting Simulation...")
            print(f"Path found with {len(path)} nodes.")
            sim = PathSimulator(self, path)
            sim.run(velocity=6.0)  # Adjust speed here
        # path = self.plan((self.map.goal_pos[0],self.map.goal_pos[1],0),step_distance=1)
