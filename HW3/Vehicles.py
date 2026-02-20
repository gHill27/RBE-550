import math
from math import *
import heapq

from Map_Generator import Map

import matplotlib.pyplot as plt
import matplotlib.patches as patches


from shapely.geometry import box, Polygon
from shapely.affinity import rotate, translate
from shapely.ops import unary_union
from typing import TypeAlias,Optional,Tuple,List
from abc import ABC,abstractmethod

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
    def __init__(self, height: float, width: float, plot = False):
        self.height = height
        self.width = width
        if plot:
            self.viz = PlannerVisualizer()
        else:
            self.viz = None
        #robot's dimentions for the minkoski sums
        self.base_footprint : Polygon = box(-width/2, -height/2, width/2, height/2)
        
    
    @abstractmethod 
    def plan(self) -> Optional[list[State]]:
        """Calculates a path from start to goal. Must be implemented by subclasses."""
        # should be based off the type of vechile
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


class Delivery(Vehicle):
    """
    A specific implementation of a Delivery robot using A* State Lattice.
    """
    def __init__(self, startPose : State = (0.4,0.4,0.0),map = None, plot = False):
        super().__init__(height=0.7, width=0.57,plot=plot)
        #if a map doesn't exist create one
        self.map = map if map else Map(12,100,0.1)
        self.full_obstacle_geometry: Optional[Polygon] = None
        self.exploredNodes: dict[State,State] = {}
        self.start_pos : State = startPose  # (x,y,theta)
        self.prepare_obstacles(self.map.obstacle_coordinate_list)
        pass

    def prepare_obstacles(self,obstacle_List: List[Tuple[int,int]]):
        """
        Flattens individual grid obstacles into a single geometric 'map'.
        
        Args:
            obstacle_list: List of (row, col) integer coordinates from the map.
        """
        polys = []
        for row, col in obstacle_List:
            # Create a 1x1 square for each obstacle coordinate
            # (x_min, y_min, x_max, y_max)
            polys.append(box(row, col, row + 1, col + 1))

        # Combine all boxes into one complex shape (highly optimized for checking)
        self.full_obstacle_geometry = unary_union(polys)

    def plan(self, goal: State , step_size: int = 50) -> Optional[List[State]]:
        """
        Performs an A* search on the state lattice.
        
        Args:
            goal: The target (x, y, theta) coordinates.
            step_size: Number of iterations between visualization updates.
            
        Returns:
            A list of States forming the path, or None if no path is found.
        """
        mp = self.calculate_motion_primitives()
        count = 0
        start_node = self.start_pos
        # self.map.goal_pos = (1.3,2.2) #TODO fix this so it doesn't brick the code everytime

        # open_list stores: (f_score, current_coord)
        open_list = []
        heapq.heappush(
            open_list, (0 + self.calculate_heurisitic(start_node,goal), start_node)
        )

        # Track the best cost to reach a coordinate
        costHistory : dict[State,float] = {start_node: 0}
        # Track the path: {child_coord: parent_coord}
        # each coord should include a tuple of (x,y,theta)
        came_from : dict[State,State]= {}

        while open_list:
            # Get the node with the lowest f_score
            _, current_state = heapq.heappop(open_list)

            # Check if we are "close enough" to the goal (floating point friendly)
            if self.calculate_heurisitic(current_state,goal) < 0.1:
                print("Goal Reached!")
                self.exploredNodes = came_from
                final_path = self.reconstruct_path(came_from, current_state)
                if self.viz:
                    self.viz.show_final(
                        final_path,
                        costHistory,
                        self.map.obstacle_coordinate_list,
                        self.map.goal_pos,
                    )
                return final_path

            # Expand neighbors using motion primitives
            for key, value in mp.items():
                dx, dy, dtheta = value[0], value[1], key
                thetaexpect = round(dtheta, 1)
                raw_neighbor = (current_state[0] + dx, current_state[1] + dy, thetaexpect)
                # Rounding for consistency in the dictionary keys
                raw_neighbor = (round(raw_neighbor[0], 2), round(raw_neighbor[1], 2), raw_neighbor[2])

                # if (math.floor(neighbor[0]),math.floor(neighbor[1])) in self.map.obstacle_coordinate_list and 0 < neighbor[0] < 12 and 0 < neighbor[1] < 12:
                #     continue
                if not self.is_state_valid(raw_neighbor):
                    continue  # Skip: The robot hits a wall at this specific angle

                snapped_neighbor = self.snap_to_grid(raw_neighbor)
                # Assume constant cost of 0.5 (step size) for each primitive
                tentativeCostToCome = costHistory[current_state] + 0.5

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
                    self.map.goal_pos,
                )
            count += 1

        self.exploredNodes = came_from
        print("No path found.")
        return None

    def reconstruct_path(self, came_from: dict[State,State], current:State) -> list[State]:
        """
        Walks backward from the goal to the start using the parent pointers.
        """
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]  # Return reversed path

    def calculate_heurisitic(self, pose: State, goal : State) -> float:
        """
        Estimates the cost to reach the goal using Euclidean distance.
        
        Args:
            pose: Current (x, y, theta) state.
            goal: The (x, y) target coordinates.
        """
        cost = round(
            sqrt(
                (goal[0] - pose[0]) ** 2
                + (goal[1] - pose[1]) ** 2
            ),
            2,
        )
        return cost

    def calculate_motion_primitives(
        self, step_precision: int = 16
    ) -> dict[float, tuple[float, float]]:
        """
        Generates possible (dx, dy) moves for a set number of headings.
        """
        motion_primatives = {}
        for i in range(step_precision):
            angle = 0 + i * (360 / step_precision)
            coordinate = self.motion_primitive_equation(angle)
            motion_primatives[angle] = coordinate
        return motion_primatives

    def motion_primitive_equation(
        self, angle: float, distance_traveled=0.5
    ) -> tuple[float]:
        angle_rad = math.radians(angle)
        point = (
            round(distance_traveled * cos(angle_rad), 2),
            round(distance_traveled * sin(angle_rad), 2),
        )
        return point
    def snap_to_grid(self, state: tuple[float, float, float], res=0.1, angle_res=22.5) -> tuple:
        """
        Discretizes a continuous state into a hashable 'bin' for the A* dictionaries.
        
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
    def is_collision(self, state : State):
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
        
        # 2. Boundary Check (Entire shell must be inside 0-12)
        world_box = box(0.01, 0.01, 11.99, 11.99)
        if not footprint.within(world_box):
            return False
            
        # 3. Obstacle Check
        if self.full_obstacle_geometry and footprint.intersects(self.full_obstacle_geometry):
            return False
                
        return True

    def main_run(self, plot_explored=True):
        # Run the planner
        path = self.plan((self.map.goal_pos[0],self.map.goal_pos[1],0))
        print(path)
        match plot_explored:
            case True:
                if path:
                    print(f"Path found with {len(path)} steps!")
                    # self.viz.draw(path,self.exploredNodes)
                else:
                    print("No path found, but plotting explored area...")
                    # self.viz.draw(None)
            case False:
                if path:
                    print(f"Path found with {len(path)} steps!")
                    # self.plot_path(path)
                else:
                    print("No path found, but plotting explored area...")
                    # self.plot_path(None)


class Police(Vehicle):
    def __init__(self):
        super().__init__(height=5.2, width=1.8)


class Truck(Vehicle):
    def __init__(self):
        super().__init__(height=5.4, width=2.0)


if __name__ == "__main__":
    deliver = Delivery(plot=True)
    deliver.main_run()