from Vehicles import Vehicle, State
from shapely import Polygon
from shapely.affinity import rotate, translate
from shapely.ops import unary_union
from shapely.geometry import box
import math
import numpy as np
import heapq


class Truck(Vehicle):
    def __init__(self, startPose, map, goalPose, plot = None):
        super().__init__(
            width=5.4,
            height=2.0,
            startState=startPose,
            goalState=goalPose,
            map=map,
            plot=plot,
        )
        self.trailerWidth = 4.5
        self.trailerHeight = 2.0
        self.lut = TruckTrailerLUT()
        
    def plan(self, goal: tuple, step_size: int = 100, step_distance: float = 1.5):
        """
        4D A* Search for Truck and Trailer.
        Goal is (x, y, theta0, theta1)
        """
        # 1. Initialize LUT and Search structures
        self.lut = TruckTrailerLUT(step_dist=step_distance, L=3.5, d1=5.4)
        start_node = self.start_pos # (x, y, t0, t1)
        
        # open_list: (f_score, state)
        open_list = []
        heapq.heappush(open_list, (0 + self.calculate_heuristic(start_node, goal), start_node))

        # Track costs and parents
        cost_history = {self.snap_to_grid(start_node, step_distance*0.4): 0}
        came_from = {}
        
        count = 0
        while open_list:
            _, current_state = heapq.heappop(open_list)
            curr_snapped = self.snap_to_grid(current_state, step_distance*0.4)

            # 2. Goal Check
            if self.is_near_goal(current_state, goal):
                print(f"Path found! Nodes explored: {count}")
                return self.reconstruct_path(came_from, curr_snapped)

            # 3. Expansion
            # We pass current_state to get_neighbors which uses our LUT
            for neighbor in self.get_neighbors(current_state, self.lut):
                
                # Collision & Boundary Check
                if not self.is_state_valid(neighbor):
                    continue

                # 4. Custom Cost Calculation
                # Base cost is distance
                move_cost = step_distance
                
                # Penalty for reversing (make it 3x more expensive than forward)
                if self.is_reversing(current_state, neighbor):
                    move_cost *= 3.0
                
                # Penalty for sharp articulation (prevents unnecessary wiggliness)
                articulation_penalty = abs(neighbor[2] - neighbor[3]) * 0.5
                
                tentative_g_score = cost_history[curr_snapped] + move_cost + articulation_penalty
                
                neighbor_snapped = self.snap_to_grid(neighbor, step_distance*0.4)

                if neighbor_snapped not in cost_history or tentative_g_score < cost_history[neighbor_snapped]:
                    cost_history[neighbor_snapped] = tentative_g_score
                    came_from[neighbor_snapped] = current_state
                    
                    f_score = tentative_g_score + self.calculate_heuristic(neighbor, goal)
                    heapq.heappush(open_list, (f_score, neighbor))

            count += 1
            if count % step_size == 0 and self.viz:
                self.viz.update(current_state, cost_history, self.map.obstacle_coordinate_list, goal)

        print("No path found.")
        return None
    
    def is_reversing(self, current, next_state):
        """Helper to check if the move is backward."""
        # Calculate dot product of truck heading and movement vector
        dx = next_state[0] - current[0]
        dy = next_state[1] - current[1]
        heading_vector = (math.cos(current[2]), math.sin(current[2]))
        move_vector = (dx, dy)
        
        dot_product = heading_vector[0] * move_vector[0] + heading_vector[1] * move_vector[1]
        return dot_product < 0 # Negative dot product means moving against heading

    def calculate_heuristic(self, state, goal):
        """4D Heuristic: Distance + Truck Heading + Trailer Heading."""
        dist = math.sqrt((state[0]-goal[0])**2 + (state[1]-goal[1])**2)
        
        # Truck heading alignment
        t0_diff = abs(self.normalize_angle(state[2] - goal[2]))
        
        # Trailer alignment (we want the trailer straight relative to truck)
        t1_diff = abs(self.normalize_angle(state[2] - state[3]))
        
        return dist + (t0_diff * 1.5) + (t1_diff * 0.5)
    
    def calculate_motion_primitives(self, step_distance):
        return self.lut.table

    def get_neighbors(self, current_state):
        x, y, t0, t1 = current_state
        psi = t0 - t1 
        neighbors = []

        # We add reversing logic here as well
        for direction in [1, -1]:
            for phi in self.lut.steer_options:
                res = self.lut.get_primitive(psi, phi, direction)
                if res:
                    dx, dy, dt0, dt1 = res
                    # Standard 2D rotation for the Truck's relative movement
                    world_dx = dx * math.cos(t0) - dy * math.sin(t0)
                    world_dy = dx * math.sin(t0) + dy * math.cos(t0)

                    new_state = (
                        x + world_dx,
                        y + world_dy,
                        self.normalize_angle(t0 + dt0),
                        self.normalize_angle(t1 + dt1)
                    )
                    neighbors.append(new_state)
        return neighbors

    def get_footprint(self, x, y, t0, t1) -> Polygon:
        """Calculates both the Truck and Trailer polygons combined."""
        # 1. Truck Footprint (centered at x, y)
        truck_base = box(-self.height/2, -self.width/2, self.height/2, self.width/2)
        truck_poly = translate(rotate(truck_base, t0, use_radians=True), x, y)

        # 2. Trailer Footprint
        # The trailer axle center is d1 meters behind the hitch (x, y)
        trailer_axle_x = x - self.d1 * math.cos(t1)
        trailer_axle_y = y - self.d1 * math.sin(t1)
        
        # Center the trailer box on its axle
        trailer_base = box(-self.height/2, -self.width/2, 
                           self.height/2, self.width/2)
        trailer_poly = translate(rotate(trailer_base, t1, use_radians=True), 
                                 trailer_axle_x, trailer_axle_y)

        # 3. Combine them
        return unary_union([truck_poly, trailer_poly])

    def snap_to_grid(self, state, res, angle_res=math.radians(15)):
        """Overridden to handle 4D state (x, y, t0, t1)"""
        x, y, t0, t1 = state
        snapped_x = round(x / res) * res
        snapped_y = round(y / res) * res
        snapped_t0 = round(t0 / angle_res) * angle_res
        snapped_t1 = round(t1 / angle_res) * angle_res
        
        return (round(snapped_x, 2), round(snapped_y, 2), 
                round(snapped_t0, 3), round(snapped_t1, 3))
    
    def main_run(self):
        self.plan(self.goal_state)

    def normalize_angle(self, angle):
        """
        Wraps the angle to stay within [-pi, pi] for radians 
        or [-180, 180] for degrees.
        """
        # If using Radians (recommended for the LUT math):
        return (angle + math.pi) % (2 * math.pi) - math.pi

        # IF you prefer Degrees (make sure to be consistent!):
        # return (angle + 180) % 360 - 180

def calculate_step(state, phi, step_dist, L=3.5, d1=5.0):
    """
    Calculates the next state for a truck-trailer system.
    
    Args:
        state: Tuple (x, y, theta0, theta1) in radians
        phi: Steering angle in radians
        step_dist: Distance to move (ds)
        L: Truck wheelbase (m)
        d1: Trailer length from hitch to axle (m) - user specified 5m
    """
    x, y, theta0, theta1 = state
    
    # We use a few sub-steps for better numerical stability during turns
    sub_steps = 5
    ds = step_dist / sub_steps
    
    for _ in range(sub_steps):
        # Update Truck
        x += ds * math.cos(theta0)
        y += ds * math.sin(theta0)
        theta0 += (ds / L) * math.tan(phi)
        
        # Update Trailer
        # The change in trailer angle is driven by the difference 
        # between the truck heading and trailer heading.
        theta1 += (ds / d1) * math.sin(theta0 - theta1)
        
    # Normalize angles to [-pi, pi]
    theta0 = (theta0 + math.pi) % (2 * math.pi) - math.pi
    theta1 = (theta1 + math.pi) % (2 * math.pi) - math.pi
    
    return (x, y, theta0, theta1)



def generate_relative_primitive(start_articulation, phi, step_dist, L=3.5, d1=5.0):
    """
    Calculates a RELATIVE offset starting from (0, 0, 0, start_articulation).
    You run this ONCE at the start of your program to build a library.
    """
    # Start at origin with truck heading 0
    state = (0.0, 0.0, 0.0, start_articulation) 
    
    # Run the math once...
    new_state = calculate_step(state, phi, step_dist, L, d1)
    
    # Return only the CHANGES (the offsets)
    dx, dy, d_theta0, d_theta1 = new_state
    return (dx, dy, d_theta0, d_theta1)




class TruckTrailerLUT:
    def __init__(self, step_dist=1.5, L=3.5, d1=5.0):
        self.step_dist = step_dist
        self.L = L
        self.d1 = d1
        
        # 1. Define our discrete "Bins"
        # Articulation: from -70 to 70 degrees (avoiding 90 deg jackknife)
        self.articulation_bins = np.linspace(math.radians(-70), math.radians(70), 15)
        # Steering: standard -30, -15, 0, 15, 30
        self.steer_options = [math.radians(a) for a in [-30, -15, 0, 15, 30]]
        
        # 2. The Table: {(articulation_bin, steer): (dx, dy, d_theta0, d_theta1)}
        self.table = {}
        self._generate_table()

    def _generate_table(self):
        """Populates the table with pre-computed relative motions."""
        for psi_start in self.articulation_bins:
            for phi in self.steer_options:
                # We start at (0,0,0) with the trailer at psi_start
                # theta1 = theta0 - psi => 0 - psi_start
                state = (0.0, 0.0, 0.0, -psi_start)
                
                # Run the simulation (Euler integration)
                new_state = self._simulate_step(state, phi)
                
                # Store the delta (change)
                # (dx, dy, d_theta0, d_theta1)
                self.table[(psi_start, phi)] = (
                    new_state[0], new_state[1], 
                    new_state[2], new_state[3]
                )

    def _simulate_step(self, state, phi):
        """The core kinematic simulation loop (run only once per bin)."""
        x, y, t0, t1 = state
        sub_steps = 10
        ds = self.step_dist / sub_steps
        
        for _ in range(sub_steps):
            x += ds * math.cos(t0)
            y += ds * math.sin(t0)
            t0 += (ds / self.L) * math.tan(phi)
            t1 += (ds / self.d1) * math.sin(t0 - t1)
            
        return (x, y, t0, t1)

    def get_primitive(self, current_psi, phi):
        """Finds the closest pre-computed relative move."""
        # Find the nearest articulation bin
        closest_psi = min(self.articulation_bins, key=lambda x: abs(x - current_psi))
        return self.table.get((closest_psi, phi))