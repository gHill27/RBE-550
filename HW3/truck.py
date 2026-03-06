from Vehicles import Vehicle, State

import math
import numpy as np


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

    def calculate_motion_primitives(self, step_distance):
        pass

    def get_neighbors(self, current_state):
        x, y, t0, t1 = current_state
        psi = t0 - t1 # Current bend
        neighbors = []

        for phi in self.lut.steer_options:
            # 1. Grab the 'cheat sheet' values
            dx, dy, dt0, dt1 = self.lut.get_primitive(psi, phi)

            # 2. Rotate the relative DX, DY to match the truck's heading (t0)
            # Using the standard 2D rotation matrix
            world_dx = dx * math.cos(t0) - dy * math.sin(t0)
            world_dy = dx * math.sin(t0) + dy * math.cos(t0)

            # 3. Add to current position
            new_state = (
                x + world_dx,
                y + world_dy,
                self.normalize_angle(t0 + dt0),
                self.normalize_angle(t1 + dt1)
            )
            
            # 4. Jackknife check
            if abs(new_state[2] - new_state[3]) < math.radians(85):
                neighbors.append(new_state)
                
        return neighbors

    def main_run(self):
        pass



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