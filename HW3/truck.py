from Vehicles import Vehicle, State
from shapely import Polygon
from shapely.affinity import rotate, translate
from shapely.ops import unary_union
from shapely.geometry import box
from pathSimulator import PathSimulator
import math
import numpy as np
import heapq


class Truck(Vehicle):
    def __init__(self, startPose, map, goalPose, plot = None):
        startPose = (*startPose, startPose[2])
        goalPose = (*goalPose, goalPose[2])
        super().__init__(
            width=2,
            height=5.4,
            startState=startPose,
            goalState=goalPose,
            map=map,
            plot=plot,
        )
        
        self.trailerWidth = 2
        self.L = 3.5
        self.d1 = 5
        self.trailerHeight = 4.5
        self.lut = TruckTrailerLUT()
        self.viz.vehicle = self    

    def calculate_heuristic(self, state, goal):
        WEIGHT = 1.5
        """4D Heuristic: Distance + Truck Heading + Trailer Heading."""
        dist = math.sqrt((state[0]-goal[0])**2 + (state[1]-goal[1])**2)
        if dist < 2:
            # Truck heading alignment
            t0_diff = abs(self.normalize_angle(state[2] - goal[2]))
            
            # Trailer alignment (we want the trailer straight relative to truck)
            t1_diff = abs(self.normalize_angle(state[2] - state[3]))
        
            return dist*WEIGHT + (t0_diff * 0.15) + (t1_diff * 0)
        return dist * WEIGHT
    
    def calculate_motion_primitives(self, step_distance):
        return self.lut

    def get_neighbors(self, current_state, motion_primatives):
        x, y, t0, t1 = current_state
        psi = t0 - t1 
        neighbors = []
        t0_rad,t1_rad,psi_rad = math.radians(t0),math.radians(t1), math.radians(psi)
        for phi in self.lut.steer_options:
            res = self.lut.get_primitive(current_state=current_state, phi = phi)
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
        """
        (x, y) is the HITCH point.
        t0 is Truck heading, t1 is Trailer heading (both in degrees).
        """
        # 1. TRUCK: Hitch is at the rear. 
        # Box goes from 0 to +Length along the X-axis.
        truck_base = box(0, -self.width/2, self.height, self.width/2)
        # Rotate around the HITCH (0,0), then move to world (x,y)
        truck_poly = translate(rotate(truck_base, t0, origin=(0, 0), use_radians=False), x, y)

        # 2. TRAILER: Hitch is at the front.
        # Box goes from -d1 to 0 along the X-axis.
        trailer_base = box(-self.d1, -self.trailerWidth/2, -1, self.trailerWidth/2)
        # Rotate around the HITCH (0,0), then move to world (x,y)
        trailer_poly = translate(rotate(trailer_base, t1, origin=(0, 0), use_radians=False), x, y)

        # Return as a list to maintain color indexing [Truck, Trailer]
        return [truck_poly, trailer_poly]

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
        path = self.plan(self.goal_state, step_distance=1.5)
        if path:
            print("Starting Simulation...")
            print(f"Path found with {len(path)} nodes.") # Check this number!
            sim = PathSimulator(self, path)
            sim.run(velocity=6.0)  # Adjust speed here

    




class TruckTrailerLUT:
    def __init__(self, step_dist=1.5, L=3.5, d1=5.0):
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
        x, y, t0_deg, t1_deg = state
        sub_steps = 10
        ds = self.step_dist / sub_steps
        
        phi_rad = math.radians(phi_deg)

        for _ in range(sub_steps):
            # 1. Calculate the rate of change for headings (in radians)
            # Tractor change: bicycle model
            dt0_rad = (ds / self.L) * math.tan(phi_rad)
            # Trailer change: standard kinematic follow-rule
            t0_rad = math.radians(t0_deg)
            t1_rad = math.radians(t1_deg)
            dt1_rad = (ds / self.d1) * math.sin(t0_rad - t1_rad)

            # 2. Update Position using the MIDPOINT heading
            # This ensures the displacement vector is aligned with the arc, not the tangent
            mid_t0_rad = t0_rad + (dt0_rad / 2.0)
            x += ds * math.cos(mid_t0_rad)
            y += ds * math.sin(mid_t0_rad)

            # 3. Update the state headings for the next sub-step
            t0_deg += math.degrees(dt0_rad)
            t1_deg += math.degrees(dt1_rad)
            
        return (x, y, t0_deg, t1_deg)

    def get_primitive(self, current_state, phi):
        x, y, t0, t1 = current_state
        current_psi = t0 - t1 # Articulation angle
        
        # 1. Find the nearest pre-computed bin
        closest_psi = min(self.articulation_bins, key=lambda x: abs(x - current_psi))
        dx_rel, dy_rel, dt0, dt1 = self.table.get((closest_psi, phi))
        
        # 2. ROTATE the relative displacement into the world frame
        # This is the step that stops the sideways sliding!
        t0_rad = math.radians(t0)
        dx_world = dx_rel * math.cos(t0_rad) - dy_rel * math.sin(t0_rad)
        dy_world = dx_rel * math.sin(t0_rad) + dy_rel * math.cos(t0_rad)
        
        # 3. Apply the rotated displacement
        new_x = x + dx_world
        new_y = y + dy_world
        new_t0 = (t0 + dt0) % 360
        new_t1 = (t1 + dt1) % 360
        
        return (new_x, new_y, new_t0, new_t1)