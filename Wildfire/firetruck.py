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
        self.map:Map = map
        plot = plot
        #Known constraints
        self.width = 4.9
        self.height = 2.2
        self.MIN_TURNING_RADIUS = 13
        self.MAX_VEL = 10
        #variables used in funcitons
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
  #######################################################################################################
  # Building the PRM Tree  
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
    
#########################################################################################################
# Traversal through the tree
    def plan(self, goal_state: Tuple[float, float, float]):
    # 1. Connect current truck pos to the web
        start_idx = self.get_nearest_node((self.map.firetruck_pose[0], self.map.firetruck_pose[1]))
        
        # 2. Connect the fire (goal) to the web
        goal_idx = self.get_nearest_node((goal_state[0], goal_state[1]))

        if start_idx is None or goal_idx is None:
            print("Could not connect start or goal to the PRM!")
            return None

        # 3. Search the web
        path = self.a_star_prm(start_idx, goal_idx)
        
        # 4. Final step: add the actual goal_state to the end of the path
        if path:
            path.append(goal_state)
            
        return path
    
    def a_star_prm(self, start_idx: int, goal_idx: int) -> Optional[List[int]]:
        # 1. Setup Priority Queue: (f_score, current_index)
        open_list = []
        # h(start) = distance from start_node to goal_node
        h_start = math.dist(self.nodes[start_idx], self.nodes[goal_idx])
        heapq.heappush(open_list, (h_start, start_idx))

        # 2. Tracking Dictionaries
        came_from = {} # To reconstruct the path: {child_idx: parent_idx}
        g_score = {i: float('inf') for i in range(len(self.nodes))}
        g_score[start_idx] = 0

        while open_list:
            _, current = heapq.heappop(open_list)

            # Check if we reached the goal index
            if current == goal_idx:
                return self._reconstruct_index_path(came_from, current)

            # Look at neighbors in the PRM graph
            for neighbor in self.graph.get(current, []):
                # Distance between current node and neighbor node
                edge_weight = math.dist(self.nodes[current], self.nodes[neighbor])
                tentative_g_score = g_score[current] + edge_weight

                if tentative_g_score < g_score[neighbor]:
                    # This path is better, record it.
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    
                    # f = g + h
                    h = math.dist(self.nodes[neighbor], self.nodes[goal_idx])
                    f_score = tentative_g_score + h
                    heapq.heappush(open_list, (f_score, neighbor))

        return None # No path found through the web

    def get_nearest_node(self, pos: Tuple[float, float]) -> Optional[int]:
        """Finds the closest PRM node index with a clear line-of-sight."""
        best_idx = None
        min_dist = float('inf')

        for i, node_pos in enumerate(self.nodes):
            d = math.dist(pos, node_pos)
            if d < min_dist:
                # Only connect if there are no obstacles in the way!
                if self._is_path_clear(pos, node_pos):
                    min_dist = d
                    best_idx = i
        return best_idx

    def _reconstruct_index_path(self, came_from, current):
        """Converts the parent-pointers into a list of (x, y, theta) for the simulator."""
        path = []
        while current in came_from:
            node_pos = self.nodes[current]
            # We add 0.0 for theta as a placeholder; 
            # the truck's controller usually handles the heading.
            path.append((node_pos[0], node_pos[1], 0.0))
            current = came_from[current]
        
        # Add the final start node and reverse
        start_node = self.nodes[current]
        path.append((start_node[0], start_node[1], 0.0))
        return path[::-1]
    
    

########################################################################################################
# Main Loop
    def main_run(self):
        # Run the planner
        import time

        start = time.time()
        path = self.plan((100,100,0))
        end = time.time()
        print(f"Time taken: {end - start} seconds")
        if path:
            print("Starting Simulation...")
            print(f"Path found with {len(path)} nodes.")
            # sim = PathSimulator(self, path)
            # sim.run(velocity=6.0)  # Adjust speed here

