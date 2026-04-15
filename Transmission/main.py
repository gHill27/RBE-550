#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from planner import RRTPlanner3D
import numpy as np
import trimesh
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
START = np.array([-248 + (PRIMARY_LENGTH / 2), 0.0, BEARING_Z+1])
GOAL  = np.array([0.0 + (PRIMARY_LENGTH / 2),    0.0, BEARING_Z])


def main():
    # 1. Define Search Space Bounds
    # x: [-400, 100], y: [-100, 100], z:
    bounds = [(-400, 400), (-400, 400), (-300, 300)]
    
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
    case_mesh = planner.checker.added_meshes['TransmissionCase']
    case_mesh.visual.face_colors = (200,200,200,60)
    
    # Add the countershaft if it's already installed
    planner.checker.add_from_scad(
        'secondary_shaft.scad', 
        name='CounterShaft', 
        position=(SECONDARY_LENGTH/2 + 1, 0, CS_BEARING_Z),
        parameters={'part': 'countershaft'}
    )

    # 4. Set the Robot (The Primary Shaft)
    # This now registers the mesh as "robot" in the CollisionManager
    print("Loading primary shaft (robot)...")
    planner.set_robot(
        scad_file='primary_shaft.scad',
        start_position=START
    )


    pos = tuple(START)
    planner.checker.update_position("robot", pos)
    robot_mesh = planner.checker.added_meshes["robot"]
    robot_transform = planner.checker.current_poses["robot"]

    for name, mesh in planner.checker.added_meshes.items():
        if name == "robot":
            continue
        temp = trimesh.collision.CollisionManager()
        temp.add_object(name, mesh, planner.checker.current_poses[name])
        hit = temp.in_collision_single(robot_mesh, robot_transform)
        print(f"  robot vs {name}: {'COLLISION' if hit else 'clear'}")
    

    print("\n--- Diagnostic Visualization ---")
    planner.checker.update_position("robot", START)

    # Use the helper method from your CollisionChecker3D class
    
    collisions = planner.checker.check_all_collisions()
    if collisions:
        print(f"🚨 Collisions at start: {collisions}")
    else:
        print("✅ Start position clear.")
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