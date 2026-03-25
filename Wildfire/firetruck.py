from Vehicles import Vehicle, Map, State
from typing import List, Callable, Tuple
import math
from pathSimulator import PathSimulator
import math
import random
from shapely.geometry import LineString, Point

PRM_RANDOM = random.Random()

class Firetruck(Vehicle):
    def __init__(self, startPose, map, plot=None):
        super().__init__(
            width=4.9,
            height=2.2,
            startState=startPose,
            goalState=None,
            map=map,
            plot=plot,
        )
        self.minimum_turn_radius = 13
        self.max_velocity = 10
        self.nodes = []
        self.graph = {}  #index : neighbor_index       
            

    def calculate_motion_primitives(self, step_distance):
        mp = {}
        L = 3  # wheel base in meters

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

    def plan(self):
        # A star plan along the current grid discovered through PRM
        pass

    def build_tree(self):
        """Main function to generate the roadmap."""
        # 1. Sample valid points
        self._sample_points()
        
        # 2. Connect nodes
        self._connect_nodes()
        print(f"PRM Built: {len(self.nodes)} nodes, {sum(len(v) for v in self.graph.values())//2} edges")

    def _sample_points(self,samples:int = 100):    
        # Calculate world bounds (50 * 5 = 250)
        limit = self.map.grid_num * self.map.cell_size
        self.nodes = []
        self.graph = {}
        
        # We use the index of the list as the key to ensure no overlap
        for _ in range(samples):
            # Generate x and y COMPLETELY separately
            tx = PRM_RANDOM.uniform(5.0, limit - 5.0)
            ty = PRM_RANDOM.uniform(5.0, limit - 5.0)
            print(tx)
            print(ty)
            

            # Pass as a explicit 3-tuple to match your State TypeAlias
            if self._is_point_free((tx, ty, 0.0)):    
                new_node = (tx, ty)
                print(new_node)
                self.nodes.append(new_node)
                
                # The key in the graph must match the current index of self.nodes
                current_index = len(self.nodes) - 1
                self.graph[current_index] = []


    def _is_point_free(self, pos):
        # Convert world pos to grid coord for quick lookup
        grid_x = int(pos[0] / self.map.cell_size)
        grid_y = int(pos[1] / self.map.cell_size)
        return (grid_x, grid_y) not in self.map.obstacle_set

    def _connect_nodes(self,radius = 30):
        """Connects each node to its nearest neighbors within a radius."""
        
        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):
                dist = math.dist(self.nodes[i], self.nodes[j])
                
                if dist <= radius:
                    if self._is_path_clear(self.nodes[i], self.nodes[j]):
                        self.graph[i].append(j)
                        self.graph[j].append(i)

    def _is_path_clear(self, start, end):
        """Collision check for the edge (straight line)."""
        line = LineString([start, end])
        
        # Simple implementation: Check if line intersects any obstacle box
        # Optimized tip: Only check obstacles in the bounding box of the line
        if self.full_obstacle_geometry and line.intersects(self.full_obstacle_geometry):
            return False
        return True










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
        path = self.plan((self.map.find_firetruck_goal()), step_distance=1)
        end = time.time()
        print(f"Time taken: {end - start} seconds")
        if path:
            print("Starting Simulation...")
            print(f"Path found with {len(path)} nodes.")
            sim = PathSimulator(self, path)
            sim.run(velocity=6.0)  # Adjust speed here
        # path = self.plan((self.map.goal_pos[0],self.map.goal_pos[1],0),step_distance=1)
