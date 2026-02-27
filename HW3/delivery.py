from Vehicles import Vehicle,State,Map
from typing import Callable,List,Tuple
import math
from math import * 

class Delivery(Vehicle):
    """
    A specific implementation of a Delivery robot using A* State Lattice.
    """

    def __init__(self, startPose: State, goalPose: State = None, map: Map = None, plot=False):
        super().__init__(
            height=0.7,
            width=0.57,
            startState=startPose,
            goalState=goalPose,
            map=map,
            plot=plot,
        )
        # if a map doesn't exist create one

    def get_neighbors(
        self, current_state: State, motion_primatives: dict[float, tuple[float, float]]
    ) -> List[State]:
        """
        Generates possible neighboring states
        """
        neighbors = []
        for key, value in motion_primatives.items():
            dx, dy, dtheta = value[0], value[1], value[2]
            raw_neighbor = (
                current_state[0] + dx,
                current_state[1] + dy,
                dtheta,
            )
            neighbors.append(raw_neighbor)
        return neighbors

    def calculate_motion_primitives(
        self, step_distance: float, step_precision: int = 16
    ) -> dict[float, tuple[float, float]]:
        """
        Generates possible (dx, dy) moves for a set number of headings.
        """
        motion_primatives = {}
        for i in range(step_precision):
            angle = 0 + i * (360 / step_precision)
            coordinate = self.motion_primitive_equation(angle, step_distance)
            motion_primatives[angle] = State((coordinate[0], coordinate[1], angle))
        return motion_primatives

    def motion_primitive_equation(
        self, angle: float, distance_traveled: float
    ) -> tuple[float]:
        angle_rad = math.radians(angle)
        point = (
            round(distance_traveled * cos(angle_rad), 2),
            round(distance_traveled * sin(angle_rad), 2),
        )
        return point

    def main_run(self):
        # Run the planner
        path = self.plan(self.goal_state,step_distance=0.3)
        # path = self.plan((self.map.goal_pos[0],self.map.goal_pos[1],0),step_distance=0.3)
