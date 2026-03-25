import heapq 
import math
import random
from scipy.spatial import KDTree

from shapely.geometry import box, Polygon, Point, LineString
from shapely.affinity import rotate, translate
from shapely.ops import unary_union

from typing import TypeAlias, Optional, Tuple, List

from Map_Generator import Map, Status
from pathSimulator import PathSimulator
from pathVisualizer import PlannerVisualizer

PRM_RANDOM = random.Random()

State: TypeAlias = tuple[float, float, float]  # (x, y, theta)
Point2D: TypeAlias = tuple[float, float]

class Firetruck():
    def __init__(self, map:Map, plot=None):
        self.width = 4.9
        self.height = 2.2
        self.map:Map = map
        plot = plot


        self.MIN_TURNING_RADIUS = 13
        self.MAX_VEL = 10
        self.nodes = []
        self.graph = {}  #index : neighbor_index  
        self.full_obstacle_geometry: Optional[Polygon] = None   
        self.prepare_obstacles(self.map.obstacle_set)
        if plot:
            self.viz = PlannerVisualizer((self.width, self.height))
        else:
            self.viz = None
        # robot's dimentions for the minkoski sums
        self.base_footprint: Polygon = box(
            -self.width / 2, -self.height / 2, self.width / 2, self.height / 2
        )  
            
    def plan(
        self, goal: State, step_size: int = 10, step_distance: float = 0.5
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
                        self.map.obstacle_set,
                        goal,
                    )
                return final_path

            # Expand neighbors using motion primitives
            neighbors = self.get_neighbors(
                current_state=current_state, motion_primatives=mp
            )
            for raw_neighbor in neighbors:

                # creates bins for less repeated checks
                snapped_neighbor = self.snap_to_grid(
                    raw_neighbor, res=(step_distance * 0.4), angle_res=5
                )
                curr_snapped = self.snap_to_grid(current_state, res=(step_distance*0.4),angle_res=5)

                # checks if the state is valid using raw neighbor
                if not self.is_state_valid(raw_neighbor):
                    continue

                # Assume constant cost of (step_distance) for each primitive
                # Penalize the magnitude of the turn
                angle_diff = abs((raw_neighbor[2] - current_state[2] + 180) % 360 - 180)
                turn_penalty = (angle_diff / 15.0) * 0.2  # Scaled penalty
                tentativeCostToCome = (
                    costHistory[current_state] + step_distance + turn_penalty
                )

                if (
                    snapped_neighbor not in costHistory
                    or tentativeCostToCome < costHistory[snapped_neighbor]
                ):
                    # This path to neighbor is better than any previous one
                    came_from[snapped_neighbor] = curr_snapped
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
                    self.map.obstacle_set,
                    goal,
                )
            count += 1

        self.exploredNodes = came_from
        print("No path found.")
        return None

    def build_tree(self):
        """Main function to generate the roadmap."""
        # 1. Sample valid points
        self._sample_points()
        
        # 2. Connect nodes
        self._connect_nodes()
        print(f"PRM Built: {len(self.nodes)} nodes, {sum(len(v) for v in self.graph.values())//2} edges")

    def _sample_points(self,samples:int = 500):    
        # Calculate world bounds (50 * 5 = 250)
        limit = self.map.grid_num * self.map.cell_size
        self.nodes = []
        self.graph = {}
        
        # We use the index of the list as the key to ensure no overlap
        for _ in range(samples):
            # Generate x and y COMPLETELY separately
            tx = PRM_RANDOM.uniform(5.0, limit - 5.0)
            ty = PRM_RANDOM.uniform(5.0, limit - 5.0)            

            # Pass as a explicit 3-tuple to match your State TypeAlias
            if self._is_point_free((tx, ty, 0.0)):    
                new_node = (tx, ty)
                self.nodes.append(new_node)
                
                # The key in the graph must match the current index of self.nodes
                current_index = len(self.nodes) - 1
                self.graph[current_index] = []

    def _is_point_free(self, pos):
        # Convert world pos to grid coord for quick lookup
        grid_x = int(pos[0] / self.map.cell_size)
        grid_y = int(pos[1] / self.map.cell_size)
        return (grid_x, grid_y) not in self.map.obstacle_set

    def _connect_nodes(self, k_neighbors=10, max_radius=50):
        """Finds the 10 closest neighbors for every node using a K-D Tree."""
        if not self.nodes:
            return

        # 1. Build the tree (O(N log N))
        tree = KDTree(self.nodes)
        
        # 2. Query the tree for the 'k+1' nearest neighbors
        # We ask for k+1 because the closest node to node[i] is always node[i] itself.
        distances, indices = tree.query(self.nodes, k=k_neighbors + 1)

        for i in range(len(self.nodes)):
            # indices[i] is a list of the 11 closest indices
            for dist, neighbor_idx in zip(distances[i], indices[i]):
                # Skip if it's the node itself or too far away
                if i == neighbor_idx or dist > max_radius:
                    continue
                
                # 3. Collision Check (The expensive part)
                if self._is_path_clear(self.nodes[i], self.nodes[neighbor_idx]):
                    # Add to adjacency list (ensure no duplicates)
                    if neighbor_idx not in self.graph[i]:
                        self.graph[i].append(neighbor_idx)
                    if i not in self.graph[neighbor_idx]:
                        self.graph[neighbor_idx].append(i)

    def _is_path_clear(self, start, end):
        """Collision check for the edge (straight line)."""
        line = LineString([start, end])
        
        # Simple implementation: Check if line intersects any obstacle box
        # Optimized tip: Only check obstacles in the bounding box of the line
        if self.full_obstacle_geometry and line.intersects(self.full_obstacle_geometry):
            return False
        return True
    
    def prepare_obstacles(
        self, obstacle_set: set[Tuple[int, int] : Status], cell_size=5
    ):
        """
        Flattens individual grid obstacles into a single geometric 'map'.

        Args:
            obstacle_list: List of (row, col) integer coordinates from the map.
        """

        polys = []
        for row, col in obstacle_set:
            # Create a 1x1 square for each obstacle coordinate
            # (x_min, y_min, x_max, y_max)
            x_min, y_min = row * cell_size, col * cell_size
            x_max, y_max = x_min + cell_size, y_min + cell_size
            polys.append(box(x_min, y_min, x_max, y_max))

        # Combine all boxes into one complex shape (highly optimized for checking)
        self.full_obstacle_geometry = unary_union(polys)


    def main_run(self):
        # Run the planner
        import time

        start = time.time()
        path = self.plan((self.map.find_firetruck_goal()), step_distance=1)
        end = time.time()
        print(f"Time taken: {end - start} seconds")
        if path:
            print("Starting Simulation...")
            print(f"Path found with {len(path)} nodes.")
            sim = PathSimulator(self, path)
            sim.run(velocity=6.0)  # Adjust speed here
        # path = self.plan((self.map.goal_pos[0],self.map.goal_pos[1],0),step_distance=1)

##########################################################
#HW3 code to refrence:
 # def calculate_motion_primitives(self, step_distance):
    #     mp = {}
    #     L = 3  # wheel base in meters

    #     # implementing 5 drive options: Hard left, slight left, Straight, slight right, Hard right
    #     steering_angles = [-30, -15, 0, 15, 30, 180]
    #     for phi in steering_angles:
    #         if abs(phi) == 0:
    #             mp[phi] = (step_distance, 0, 0)
    #         elif phi == 180:
    #             mp[phi] = (-step_distance, 0, 0)
    #         else:
    #             # simplified integration
    #             d_theta = round((step_distance / L) * math.tan(math.radians(phi)), 2)
    #             dx = round(step_distance * math.cos(d_theta / 2), 2)
    #             dy = round(step_distance * math.sin(d_theta / 2), 2)
    #             mp[phi] = (dx, dy, round(math.degrees(d_theta), 2))
    #     return mp

    # def plan(self):
    #     # A star plan along the current grid discovered through PRM
    #     pass



    # def get_neighbors(
    #     self, current_state: State, motion_primatives: dict[float, tuple[float, float]]
    # ) -> List[State]:
    #     """
    #     Generates possible neighboring states
    #     """
    #     cos = math.cos(math.radians(current_state[2]))
    #     sin = math.sin(math.radians(current_state[2]))

    #     neighbors = []
    #     for key, value in motion_primatives.items():
    #         dx, dy, dtheta = value[0], value[1], value[2]

    #         rotated_dx = dx * cos - dy * sin
    #         rotated_dy = dx * sin + dy * cos
    #         raw_neighbor = (
    #             current_state[0] + rotated_dx,
    #             current_state[1] + rotated_dy,
    #             (current_state[2] + dtheta) % 360,
    #         )
    #         neighbors.append(raw_neighbor)
    #     return neighbors
    
    # def reconstruct_path(
    #     self, came_from: dict[State, State], current: State
    # ) -> list[State]:
    #     """
    #     Walks backward from the goal to the start using the parent pointers.
    #     """
    #     path = [current]
    #     while current in came_from:
    #         current = came_from[current]
    #         path.append(current)
    #     return path[::-1]  # Return reversed path

    # def snap_to_grid(
    #     self, state: tuple[float, float, float], res, angle_res=15.0
    # ) -> tuple:
    #     """
    #     Discretizes a continuous state into a hashable 'bin' for the A* dictionaries.

    #     Note: Logic refined in collaboration with Gemini AI to handle
    #     floating-point precision issues in A* dictionaries.


    #     Args:
    #         state: (x, y, theta)
    #         res: Spatial resolution in meters (e.g., 0.1m)
    #         angle_res: Angular resolution in degrees
    #     """
    #     x, y, theta = state
    #     # Rounding to the nearest multiple of the resolution
    #     snapped_x = round(x / res) * res
    #     snapped_y = round(y / res) * res
    #     snapped_theta = (round(theta / angle_res) * angle_res) % 360

    #     # We return a tuple of rounded values to use as a dictionary key
    #     return (round(snapped_x, 1), round(snapped_y, 1), round(snapped_theta, 1))

    # def is_near_goal(
    #     self, state: State, goal: State, pos_threshold=0.4, angle_threshold=15
    # ):
    #     """
    #     Checks if the robot is close enough to the goal in both position and heading.
    #     """
    #     curr_x, curr_y, curr_theta = state
    #     goal_x, goal_y, goal_theta = goal

    #     # 1. Position Check (Euclidean)
    #     dist = math.sqrt((curr_x - goal_x) ** 2 + (curr_y - goal_y) ** 2)

    #     # 2. Orientation Check
    #     # We use a modular difference to handle 0/360 degree wrap-around
    #     angle_diff = abs((curr_theta - goal_theta + 180) % 360 - 180)

    #     return dist <= pos_threshold and angle_diff <= angle_threshold

    # def calculate_heurisitic(
    #     self, pose: State, goal: State, weight: float = 1.2, heading_weight: float = 0.5
    # ) -> float:
    #     """
    #     Estimates the cost to reach the goal using Euclidean distance.

    #     Args:
    #         pose: Current (x, y, theta) state.
    #         goal: The (x, y) target coordinates.
    #     """
    #     cost = round(
    #         sqrt((goal[0] - pose[0]) ** 2 + (goal[1] - pose[1]) ** 2),
    #         2,
    #     )

    #     # If we are close to the goal, start caring about the angle
    #     if cost < 1.0:
    #         angle_diff = abs((pose[2] - goal[2] + 180) % 360 - 180)
    #         # Normalize angle diff so it doesn't overpower the distance
    #         # (e.g., 180 degrees = 1.0 units of distance)
    #         return (cost + (angle_diff / 180.0)) * heading_weight

    #     return cost * weight

    
    # def get_neighbors(
    #     self, current_state: State, motion_primatives: dict[float, tuple[float, float]]
    # ) -> List[State]:
    #     pass

    # def get_footprint(self, x: float, y: float, theta: float) -> Polygon:
    #     """
    #     Calculates the physical space the vehicle occupies at a specific state.

    #     Args:
    #         x: The X coordinate of the vehicle center.
    #         y: The Y coordinate of the vehicle center.
    #         theta: Heading in degrees.

    #     Returns:
    #         A Shapely Polygon representing the transformed footprint.
    #     """
    #     rotated = rotate(self.base_footprint, theta, origin=(0, 0))
    #     return [translate(rotated, xoff=x, yoff=y)]

    
    # def calculate_motion_primitives(
    #     self, step_distance: float, step_precision: int = 16
    # ) -> dict[float, tuple[float, float]]:
    #     pass

    # def is_state_valid(self, state: State) -> bool:
    #     """The 'Master' check for boundary and collisions."""
    #     # 1. Generate footprint
    #     footprints = self.get_footprint(*state)
    #     # 2. Boundary Check (Entire shell must be inside 0-35.99m)
    #     world_box = box(0.01, 0.01, self.map.cell_size*self.map.grid_num , self.map.cell_size*self.map.grid_num)
    #     for footprint in footprints:
    #         if not footprint.within(world_box):
    #             return False

    #         # 3. Obstacle Check
    #         if self.full_obstacle_geometry and footprint.intersects(
    #             self.full_obstacle_geometry
    #         ):
    #             return False

    #     return True

    
