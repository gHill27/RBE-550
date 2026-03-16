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
        
            return dist*WEIGHT + (t0_diff * 0.15) + (t1_diff * 0.1)
        return dist * WEIGHT
    
    def calculate_motion_primitives(self, step_distance):
        return self.lut

    def get_neighbors(self, current_state, motion_primatives):
        neighbors = []
        # Check both Forward and Reverse for every steering option
        for direction in [1, -1]:
            for phi in motion_primatives.steer_options:
                res = motion_primatives.get_primitive(
                    current_state=current_state, 
                    phi=phi, 
                    direction=direction
                )
                if res:
                    neighbors.append(res)
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
        trailer_base = box(-self.d1, -self.trailerWidth/2, 0, self.trailerWidth/2)
        # Rotate around the HITCH (0,0), then move to world (x,y)
        trailer_poly = translate(rotate(trailer_base, t1, origin=(0, 0), use_radians=False), x, y)

        # Return as a list to maintain color indexing [Truck, Trailer]
        return [truck_poly, trailer_poly]

    def snap_to_grid(self, state, res, angle_res=3):
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
        # Check that the trailer is straight relative to the truck (psi approx 0)
        psi = abs(self.normalize_angle(t0 - t1))
        
        return dist < pos_threshold and t0_diff < angle_threshold and psi < 10
    
    def main_run(self):
        step_distance  = 1.5
        self.lut = TruckTrailerLUT(step_dist= step_distance, L=self.L, d1 = self.d1)
        path = self.plan(self.goal_state, step_distance=step_distance)
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
        self.articulation_bins = np.linspace(-70, 70, 71)
        # Steering: standard -30, -15, 0, 15, 30
        self.steer_options =  [-30, -15, 0, 15, 30]
        
        # 2. The Table: {(articulation_bin, steer): (dx, dy, d_theta0, d_theta1)}
        self.table = {}
        self._generate_table()

    def _generate_table(self):
        for psi_start in self.articulation_bins:
            for phi in self.steer_options:
                for direction in [1, -1]: # Generate both Forward and Reverse
                    state = (0.0, 0.0, 0.0, -psi_start)
                    new_state = self._simulate_step(state, phi, direction)
                    
                    # Key is now (psi, phi, direction)
                    self.table[(psi_start, phi, direction)] = (
                        new_state[0], new_state[1], 
                        new_state[2], new_state[3]
                    )

    def _simulate_step(self, state, phi_deg, direction=1):
        """
        direction=1 for Forward, direction=-1 for Reverse.
        """
        x, y, t0_deg, t1_deg = state
        sub_steps = 15 # Increased sub-steps for reverse stability
        # direction * step_dist
        ds = (self.step_dist * direction) / sub_steps
        
        phi_rad = math.radians(phi_deg)

        for _ in range(sub_steps):
            # Tractor change: bicycle model
            dt0_rad = (ds / self.L) * math.tan(phi_rad)
            
            t0_rad = math.radians(t0_deg)
            t1_rad = math.radians(t1_deg)
            # Trailer change: kinematic follow-rule
            dt1_rad = (ds / self.d1) * math.sin(t0_rad - t1_rad)

            # Update Position
            mid_t0_rad = t0_rad + (dt0_rad / 2.0)
            x += ds * math.cos(mid_t0_rad)
            y += ds * math.sin(mid_t0_rad)

            t0_deg += math.degrees(dt0_rad)
            t1_deg += math.degrees(dt1_rad)
            
        return (x, y, t0_deg, t1_deg)

    def get_primitive(self, current_state, phi,direction):
        x, y, t0, t1 = current_state
        current_psi = self.normalize_angle(t0 - t1)
        
        # 1. Find nearest bin
        closest_psi = min(self.articulation_bins, key=lambda b: abs(b - current_psi))
        
        # These are absolute values from a simulation starting at (0,0,0)
        # with trailer at -closest_psi
        res = self.table.get((closest_psi, phi,direction))
        if not res: return None
        
        lx, ly, lt0, lt1 = res # 'l' for local/lut
        
        # 2. Rotate the relative displacement into world frame
        t0_rad = math.radians(t0)
        dx_world = lx * math.cos(t0_rad) - ly * math.sin(t0_rad)
        dy_world = lx * math.sin(t0_rad) + ly * math.cos(t0_rad)
        
        # 3. Calculate absolute new state
        new_x = x + dx_world
        new_y = y + dy_world
        
        # Headings: lt0 is the change in truck heading
        # lt1 is the new absolute trailer heading if truck started at 0
        new_t0 = self.normalize_angle(t0 + lt0)
        
        # The articulation (psi) at the end of the LUT step is (lt0 - lt1)
        final_psi = lt0 - lt1
        if abs(final_psi) > 75: # Safety limit slightly above your 70 bin
            return None
        new_t1 = self.normalize_angle(new_t0 - final_psi)
        
        return (new_x, new_y, new_t0, new_t1)

    def normalize_angle(self, angle):
        return (angle + 180) % 360 - 180