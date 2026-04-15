#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from planner import RRTPlanner3D
import numpy as np

# ---------------------------------------------------------------------------
# Geometry constants (mm)
# ---------------------------------------------------------------------------
CASE_THICKNESS  = 25
CASE_HEIGHT     = 300
BEARING_OFFSET  = 215
BEARING_Z       = BEARING_OFFSET + (CASE_THICKNESS / 2)   
CS_BEARING_Z    = 100 + CASE_THICKNESS / 2   
PRIMARY_LENGTH = 330
SECONDARY_LENGTH = 330 # case_length (280) + 2*case_thickness (50)

# Apply the half-length offset to the START and GOAL
# This centers the shaft mesh on the coordinate
START = np.array([-350.0 + (PRIMARY_LENGTH / 2)+ 40, 0.0, BEARING_Z])
GOAL  = np.array([0.0 + (PRIMARY_LENGTH / 2),    0.0, BEARING_Z])


def main():
    # 1. Define Search Space Bounds
    # x: [-400, 100], y: [-100, 100], z:
    bounds = [(-400, 100), (-100, 100), (0, 300)]
    
    # 2. Initialize Planner
    planner = RRTPlanner3D(bounds=bounds, models_folder='models')

    # 3. Add Obstacles (Fixed items in the CollisionManager)
    # The transmission case has the bearing holes
    print("Loading obstacles...")
    planner.checker.add_from_scad(
        'transmission_case.scad', 
        name='TransmissionCase', 
        position=(0, 0, 0),
        parameters={'part': 'case'}
    )
    
    # Add the countershaft if it's already installed
    planner.checker.add_from_scad(
        'secondary_shaft.scad', 
        name='CounterShaft', 
        position=(SECONDARY_LENGTH/2 + 20, 0, CS_BEARING_Z),
        parameters={'part': 'countershaft'}
    )

    # 4. Set the Robot (The Primary Shaft)
    # This now registers the mesh as "robot" in the CollisionManager
    print("Loading primary shaft (robot)...")
    planner.set_robot(
        scad_file='primary_shaft.scad',
        start_position=START
    )
    

    print("\n--- Diagnostic Visualization ---")
    planner.checker.update_position("robot", START)

    # return_names=True returns a set of frozensets/tuples
    collisions = planner.checker.manager.in_collision_internal(return_names=True)

    if collisions:
        print(f"🚨 START STATE COLLISION DETECTED")
        # Check if 'collisions' is a collection of names or just a True/False
        if isinstance(collisions, (set, list, frozenset)):
            for pair in collisions:
                pair_list = list(pair)
                if len(pair_list) == 2:
                    n1, n2 = pair_list
                    dist = planner.checker.manager.min_distance_other(n1, n2)
                    print(f"   -> {n1} ↔ {n2} | Distance: {dist:.4f}mm")
        else:
            # It's just a boolean True
            print("   -> Collision exists, but trimesh didn't return specific names.")
    else:
        print("✅ Start state is clear.")

    planner.checker.visualize()
    # 5. Plan Path
    # The validity checker now uses manager.update_position("robot", state)
    print(f"\nPlanning from {START} to {GOAL}...")
    waypoints = planner.plan_path(
        start=START,
        goal=GOAL,
        planner_type='rrt_connect',
        max_time=60.0,
        goal_tolerance=5.0
    )

    # 6. Result Handling
    if waypoints:
        print(f"✅ Success! Path found with {len(waypoints)} waypoints.")
        planner.save_path(waypoints, 'planned_path.npy')
        
        if input("\nVisualize result? (y/n): ").lower() == 'y':
            planner.visualize_path(waypoints)
    else:
        print("❌ No path found. Check if the start/goal are in collision.")

# Add this to main.py
def run_diagnostic(planner, start_pos):
    print("\n--- Diagnostic Visualization ---")
    print(f"Testing Start Position: {start_pos}")
    
    # Update manager to check this specific state
    planner.checker.update_position("robot", start_pos)
    
    # Check for collisions
    is_colliding = planner.checker.manager.in_collision_internal()
    if is_colliding:
        colliding_pairs = planner.checker.manager.in_collision_internal(return_names=True)
        print(f"🚨 COLLISION DETECTED at start: {colliding_pairs}")
    else:
        print("✅ Start position is VALID and CLEAR.")

    print("Opening 3D visualizer... (Close window to proceed)")
    planner.checker.visualize()


if __name__ == "__main__":
    main()