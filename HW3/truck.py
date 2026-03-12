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
        startPose = (*startPose, startPose[2])
        goalPose = (*goalPose, goalPose[2])
        super().__init__(
            width=5.4,
            height=2.0,
            startState=startPose,
            goalState=goalPose,
            map=map,
            plot=plot,
        )
        
        self.trailerWidth = 4.5
        self.L = 3.5
        self.d1 = 5.4
        self.trailerHeight = 2.0
        self.lut = TruckTrailerLUT()
        self.viz.vechile = self
        
    def plan(self, goal: tuple, step_size: int = 100, step_distance: float = 1.5):
        """
        4D A* Search for Truck and Trailer.
        Goal is (x, y, theta0, theta1)
        """
        start_node = self.start_pos # (x, y, t0, t1)
        
        # open_list: (f_score, state)
        open_list = []
        heapq.heappush(open_list, (0 + self.calculate_heuristic(start_node, goal), start_node))

        # Track costs and parents
        bin_res = step_distance*0.4
        cost_history = {self.snap_to_grid(start_node,bin_res ): 0}
        came_from = {}
        
        count = 0
        while open_list:
            _, current_state = heapq.heappop(open_list)
            curr_snapped = self.snap_to_grid(current_state, bin_res)

            # 2. Goal Check
            if self.is_near_goal(current_state, goal):
                print(f"Path found! Nodes explored: {count}")
                return self.reconstruct_path(came_from, curr_snapped)

            # 3. Expansion
            # We pass current_state to get_neighbors which uses our LUT
            for neighbor in self.get_neighbors(current_state):
                
                # Collision & Boundary Check
                if not self.is_state_valid(neighbor):
                    continue

                # 4. Custom Cost Calculation
                # Base cost is distance
                move_cost = step_distance

                # Penalty for sharp articulation (prevents unnecessary wiggliness)
                articulation_penalty = abs(neighbor[2] - neighbor[3]) * 0.1
                
                tentative_g_score = cost_history[curr_snapped] + move_cost + articulation_penalty
                
                neighbor_snapped = self.snap_to_grid(neighbor, bin_res)

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
    

    def calculate_heuristic(self, state, goal):
        """4D Heuristic: Distance + Truck Heading + Trailer Heading."""
        dist = math.sqrt((state[0]-goal[0])**2 + (state[1]-goal[1])**2)
        
        # Truck heading alignment
        t0_diff = abs(self.normalize_angle(state[2] - goal[2]))
        
        # Trailer alignment (we want the trailer straight relative to truck)
        t1_diff = abs(self.normalize_angle(state[2] - state[3]))
        
        return dist + (t0_diff * 0.15) + (t1_diff * 0.1)
    
    def calculate_motion_primitives(self, step_distance):
        pass

    def get_neighbors(self, current_state):
        x, y, t0, t1 = current_state
        psi = t0 - t1 
        neighbors = []
        t0_rad,t1_rad,psi_rad = math.radians(t0),math.radians(t1), math.radians(psi)
        for phi in self.lut.steer_options:
            res = self.lut.get_primitive(psi, phi)
            if res:
                dx, dy, lut_t0, lut_t1 = res # lut_t0 and lut_t1 are local headings
                
                # 1. Rotate the displacement into the world frame
                world_dx = dx * math.cos(t0_rad) - dy * math.sin(t0_rad)
                world_dy = dx * math.sin(t0_rad) + dy * math.cos(t0_rad)
                
                # 2. Calculate the new heading of the truck
                new_t0 = self.normalize_angle(t0 + lut_t0)
                
                # 3. Derive the new trailer heading from the LUT's final articulation
                # Articulation (psi) = Truck_Heading - Trailer_Heading
                final_psi = lut_t0 - lut_t1
                new_t1 = self.normalize_angle(new_t0 - final_psi)

                neighbors.append((x + world_dx, y + world_dy, new_t0, new_t1))
        return neighbors

    def get_footprint(self, x, y, t0, t1) -> Polygon:
        """Calculates both the Truck and Trailer polygons combined."""
        # 1. Truck Footprint (centered at x, y)
        if t1 is None:
            t1 = t0
        truck_base = box(-self.height/2, -self.width/2, self.height/2, self.width/2)
        truck_poly = translate(rotate(truck_base, t0, use_radians=False), x, y)

        # 2. Trailer Footprint
        # The trailer axle center is d1 meters behind the hitch (x, y)
        t1_rad = math.radians(t1)
        trailer_axle_x = x - self.d1 * math.cos(t1_rad)
        trailer_axle_y = y - self.d1 * math.sin(t1_rad)
        
        # Center the trailer box on its axle
        trailer_base = box(-self.trailerHeight/2, -self.trailerWidth/2, 
                           self.trailerHeight/2, self.trailerWidth/2)
        trailer_poly = translate(rotate(trailer_base, t1, use_radians=False), 
                                 trailer_axle_x, trailer_axle_y)

        # 3. Combine them
        return unary_union([truck_poly, trailer_poly])

    def snap_to_grid(self, state, res, angle_res=15):
        """Overridden to handle 4D state (x, y, t0, t1)"""
        x, y, t0, t1 = state
        snapped_x = round(x / res) * res
        snapped_y = round(y / res) * res
        snapped_t0 = round(t0 / angle_res) * angle_res
        snapped_t1 = round(t1 / angle_res) * angle_res
        
        return (round(snapped_x, 2), round(snapped_y, 2), 
                round(snapped_t0, 3), round(snapped_t1, 3))
    
    def normalize_angle(self, angle):
        """Wraps angle to [-180, 180]."""
        return (angle + 180) % 360 - 180

    def is_near_goal(self, state, goal, pos_threshold=1.5, angle_threshold=15):
        curr_x, curr_y, t0, t1 = state
        gx, gy, gt0, gt1 = goal
        
        dist = math.sqrt((curr_x - gx)**2 + (curr_y - gy)**2)
        t0_diff = abs(self.normalize_angle(t0 - gt0))
        
        # Optional: Only reach goal if truck AND trailer are aligned
        return dist < pos_threshold and t0_diff < angle_threshold
    
    def main_run(self):
        self.plan(self.goal_state)

    




class TruckTrailerLUT:
    def __init__(self, step_dist=1.5, L=3.5, d1=5.4):
        self.step_dist = step_dist
        self.L = L
        self.d1 = d1
        
        # 1. Define our discrete "Bins"
        # Articulation: from -70 to 70 degrees (avoiding 90 deg jackknife)
        self.articulation_bins = np.linspace(-70, 70, 15)
        # Steering: standard -30, -15, 0, 15, 30
        self.steer_options =  [-30, -15, 0, 15, 30]
        
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

    def _simulate_step(self, state, phi_deg):
        """The core kinematic simulation loop (run only once per bin)."""
        x, y, t0_deg, t1_deg = state
        sub_steps = 10
        ds = self.step_dist / sub_steps
        
        for _ in range(sub_steps):
            # Convert to radians for math.sin/cos/tan
            t0_rad = math.radians(t0_deg)
            t1_rad = math.radians(t1_deg)
            phi_rad = math.radians(phi_deg)

            # Update Position
            x += ds * math.cos(t0_rad)
            y += ds * math.sin(t0_rad)

            # Update Headings (Convert the angular change from Rad to Deg)
            dt0_deg = math.degrees((ds / self.L) * math.tan(phi_rad))
            dt1_deg = math.degrees((ds / self.d1) * math.sin(t0_rad - t1_rad))

            t0_deg += dt0_deg
            t1_deg += dt1_deg
            
        return (x, y, t0_deg, t1_deg)

    def get_primitive(self, current_psi, phi):
        """Finds the closest pre-computed relative move."""
        # Find the nearest articulation bin
        closest_psi = min(self.articulation_bins, key=lambda x: abs(x - current_psi))
        return self.table.get((closest_psi, phi))