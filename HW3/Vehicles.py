"""
Path Planning Project - HW3
Author: Gavin Hill
AI Collaboration Disclosure:
    - This project utilized Google Gemini (Free Tier) as a coding collaborator.
    - AI was used for: Refactoring logic for Shapely geometric checks, 
      implementing unit tests with pytest/mock, and optimizing state 
      discretization (binning).
    - Date of last interaction: February 2026
"""

import math
from math import *
import heapq

from Map_Generator import Map

import matplotlib.pyplot as plt
import matplotlib.patches as patches


from shapely.geometry import box, Polygon, Point
from shapely.affinity import rotate, translate
from shapely.ops import unary_union
from typing import TypeAlias, Optional, Tuple, List
from abc import ABC, abstractmethod

from pathVisualizer import PlannerVisualizer
import time

State: TypeAlias = tuple[float, float, float]  # (x, y, theta)
Point2D: TypeAlias = tuple[float, float]


class Vehicle(ABC):
    """
    Abstract Base Class for all autonomous vehicles.

    Attributes:
        height (float): The length of the vehicle (front-to-back) in meters.
        width (float): The width of the vehicle (side-to-side) in meters.
        base_footprint (Polygon): A Shapely box representing the vehicle at (0,0,0).
    """

    def __init__(
        self,
        height: float,
        width: float,
        startState: State,
        goalState: State,
        map: Map = None,
        plot=False,
    ):
        self.height = height
        self.width = width
        self.map = map if map else Map(12, 100, 0.1)
        self.full_obstacle_geometry: Optional[Polygon] = None
        self.exploredNodes: dict[State, State] = {}
        self.start_pos: State = startState  # (x,y,theta)
        self.goal_state: State = goalState
        self.prepare_obstacles(self.map.obstacle_coordinate_list)
        if plot:
            self.viz = PlannerVisualizer((width, height))
        else:
            self.viz = None
        # robot's dimentions for the minkoski sums
        self.base_footprint: Polygon = box(
            -width / 2, -height / 2, width / 2, height / 2
        )

    def plan(
        self, goal: State, step_size: int = 500, step_distance: float = 2
    ) -> Optional[List[State]]:
        """
        Performs an A* search on the state lattice.

        Args:
            goal: The target (x, y, theta) coordinates.
            step_size: Number of iterations between visualization updates.

        Returns:
            A list of States forming the path, or None if no path is found.
        """
        mp = self.calculate_motion_primitives(step_distance)
        count = 0
        start_node = self.start_pos
        # self.map.goal_pos = (1.3,2.2) #TODO fix this so it doesn't brick the code everytime

        # open_list stores: (f_score, current_coord)
        open_list = []
        heapq.heappush(
            open_list, (0 + self.calculate_heurisitic(start_node, goal), start_node)
        )

        # Track the best cost to reach a coordinate
        costHistory: dict[State, float] = {start_node: 0}
        # Track the path: {child_coord: parent_coord}
        # each coord should include a tuple of (x,y,theta)
        came_from: dict[State, State] = {}

        while open_list:
            # Get the node with the lowest f_score
            _, current_state = heapq.heappop(open_list)

            # Check if we are "close enough" to the goal (floating point friendly)
            if self.is_near_goal(current_state, goal):
                print("Goal Reached!")
                self.exploredNodes = came_from
                final_path = self.reconstruct_path(came_from, current_state)
                if self.viz:
                    self.viz.show_final(
                        final_path,
                        costHistory,
                        self.map.obstacle_coordinate_list,
                        goal,
                    )
                return final_path

            # Expand neighbors using motion primitives
            neighbors = self.get_neighbors(
                current_state=current_state, motion_primatives=mp
            )
            for raw_neighbor in neighbors:

                # creates bins for less repeated checks
                snapped_neighbor = self.snap_to_grid(raw_neighbor)

                # checks if the state is valid using raw neighbor
                if not self.is_state_valid(raw_neighbor):
                    continue

                # Assume constant cost of (step_distance) for each primitive
                turn_penalty = 0.15 if raw_neighbor[2] != current_state[2] else 0.0
                tentativeCostToCome = (
                    costHistory[current_state] + step_distance + turn_penalty
                )

                if (
                    snapped_neighbor not in costHistory
                    or tentativeCostToCome < costHistory[snapped_neighbor]
                ):
                    # This path to neighbor is better than any previous one
                    came_from[snapped_neighbor] = current_state
                    costHistory[snapped_neighbor] = tentativeCostToCome
                    estimatedCost = tentativeCostToCome + self.calculate_heurisitic(
                        snapped_neighbor, goal
                    )
                    heapq.heappush(open_list, (estimatedCost, snapped_neighbor))
            # LIVE UPDATE CALL
            if self.viz and count % step_size == 0:
                self.viz.update(
                    raw_neighbor,
                    costHistory,
                    self.map.obstacle_coordinate_list,
                    goal,
                )
            count += 1

        self.exploredNodes = came_from
        print("No path found.")
        return None

    def reconstruct_path(
        self, came_from: dict[State, State], current: State
    ) -> list[State]:
        """
        Walks backward from the goal to the start using the parent pointers.
        """
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]  # Return reversed path

    def snap_to_grid(
        self, state: tuple[float, float, float], res=0.1, angle_res=22.5
    ) -> tuple:
        """
        Discretizes a continuous state into a hashable 'bin' for the A* dictionaries.

        Note: Logic refined in collaboration with Gemini AI to handle
        floating-point precision issues in A* dictionaries.


        Args:
            state: (x, y, theta)
            res: Spatial resolution in meters (e.g., 0.1m)
            angle_res: Angular resolution in degrees
        """
        x, y, theta = state
        # Rounding to the nearest multiple of the resolution
        snapped_x = round(x / res) * res
        snapped_y = round(y / res) * res
        snapped_theta = round(theta / angle_res) * angle_res

        # We return a tuple of rounded values to use as a dictionary key
        return (round(snapped_x, 2), round(snapped_y, 2), round(snapped_theta, 2))

    def is_near_goal(
        self, state: State, goal: State, pos_threshold=0.2, angle_threshold=10
    ):
        """
        Checks if the robot is close enough to the goal in both position and heading.
        """
        curr_x, curr_y, curr_theta = state
        goal_x, goal_y, goal_theta = goal

        # 1. Position Check (Euclidean)
        dist = math.sqrt((curr_x - goal_x) ** 2 + (curr_y - goal_y) ** 2)

        # 2. Orientation Check
        # We use a modular difference to handle 0/360 degree wrap-around
        angle_diff = abs((curr_theta - goal_theta + 180) % 360 - 180)

        return dist <= pos_threshold and angle_diff <= angle_threshold

    def calculate_heurisitic(
        self, pose: State, goal: State, weight: float = 2, heading_weight: float = 1.5
    ) -> float:
        """
        Estimates the cost to reach the goal using Euclidean distance.

        Args:
            pose: Current (x, y, theta) state.
            goal: The (x, y) target coordinates.
        """
        cost = round(
            sqrt((goal[0] - pose[0]) ** 2 + (goal[1] - pose[1]) ** 2),
            2,
        )

        # If we are close to the goal, start caring about the angle
        if cost < 1.0:
            angle_diff = abs((pose[2] - goal[2] + 180) % 360 - 180)
            # Normalize angle diff so it doesn't overpower the distance
            # (e.g., 180 degrees = 1.0 units of distance)
            return (cost + (angle_diff / 180.0)) * heading_weight

        return cost * weight

    @abstractmethod
    def get_neighbors(
        self, current_state: State, motion_primatives: dict[float, tuple[float, float]]
    ) -> List[State]:
        pass

    def get_footprint(self, x: float, y: float, theta: float) -> Polygon:
        """
        Calculates the physical space the vehicle occupies at a specific state.

        Args:
            x: The X coordinate of the vehicle center.
            y: The Y coordinate of the vehicle center.
            theta: Heading in degrees.

        Returns:
            A Shapely Polygon representing the transformed footprint.
        """
        rotated = rotate(self.base_footprint, theta, origin=(0, 0))
        return translate(rotated, xoff=x, yoff=y)

    @abstractmethod
    def calculate_motion_primitives(
        self, step_distance: float, step_precision: int = 16
    ) -> dict[float, tuple[float, float]]:
        pass

    def is_collision(self, state: State):
        """
        Checks if the vehicle's footprint overlaps with any map obstacles.
        """
        if self.full_obstacle_geometry is None:
            return False

        # Generate footprint based on the current state
        footprint = self.get_footprint(*state)

        # Check if the robot geometry touches the obstacle geometry
        return footprint.intersects(self.full_obstacle_geometry)

    def is_state_valid(self, state: State) -> bool:
        """The 'Master' check for boundary and collisions."""
        # 1. Generate footprint
        footprint = self.get_footprint(*state)

        # 2. Boundary Check (Entire shell must be inside 0-35.99m)
        world_box = box(0.01, 0.01, 35.99, 35.99)
        if not footprint.within(world_box):
            return False

        # 3. Obstacle Check
        if self.full_obstacle_geometry and footprint.intersects(
            self.full_obstacle_geometry
        ):
            return False

        return True

    def prepare_obstacles(self, obstacle_List: List[Tuple[int, int]], cell_size=3):
        """
        Flattens individual grid obstacles into a single geometric 'map'.

        Args:
            obstacle_list: List of (row, col) integer coordinates from the map.
        """

        polys = []
        for row, col in obstacle_List:
            # Create a 1x1 square for each obstacle coordinate
            # (x_min, y_min, x_max, y_max)
            x_min, y_min = row * cell_size, col * cell_size
            x_max, y_max = x_min + cell_size, y_min + cell_size
            polys.append(box(x_min, y_min, x_max, y_max))

        # Combine all boxes into one complex shape (highly optimized for checking)
        self.full_obstacle_geometry = unary_union(polys)


class Delivery(Vehicle):
    """
    A specific implementation of a Delivery robot using A* State Lattice.
    """

    def __init__(self, startPose: State, goalPose: State, map: Map = None, plot=False):
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
            motion_primatives[angle] = State(coordinate[0], coordinate[1], angle)
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
        path = self.plan((self.goal_state))


class Police(Vehicle):
    def __init__(self, startPose, goalPose, map, plot):
        super().__init__(
            width=5.2,
            height=1.8,
            startState=startPose,
            goalState=goalPose,
            map=map,
            plot=plot,
        )

    def calculate_motion_primitives(self, step_distance, step_precision=5):
        mp = {}
        L = 2.7  # wheel base in meters TODO: CHANGE TO MATCH ACTUAL VALUE

        # implementing 5 drive options: Hard left, slight left, Straight, slight right, Hard right
        steering_angles = [-30, -15, 0, 15, 30]
        for phi in steering_angles:
            if abs(phi) == 0:
                mp[phi] = (step_distance, 0, 0)
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
        # prepreforming expensive cos/sin calculation
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

        path = self.plan((self.goal_state))


class Truck(Vehicle):
    def __init__(self, startState):
        super().__init__(height=5.4, width=2.0, startState=startState)


if __name__ == "__main__":
    # Add this to your __main__ to debug
    police_car = Police(
        startPose=(5, 5, 0), goalPose=(25, 25, 45), map=Map(12, 3, 0), plot=True
    )
    print(f"Is Start Valid? {police_car.is_state_valid(police_car.start_pos)}")
    police_car.main_run()
