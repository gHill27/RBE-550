#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from planner import RRTPlanner3D
import numpy as np
import trimesh
import scipy.spatial.transform as sst
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
START = np.array([83, 0.0, BEARING_Z+1])
GOAL  = np.array([-200 - (PRIMARY_LENGTH / 2),    0.0, BEARING_Z+1])

def euler_to_quat(roll, pitch, yaw, degrees=True):
    """Returns (qw, qx, qy, qz)."""
    r = sst.Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=degrees)
    qx, qy, qz, qw = r.as_quat()  # scipy returns (x,y,z,w)
    return (qw, qx, qy, qz)

def main():
    # 1. Define Search Space Bounds
    # x: [-400, 100], y: [-100, 100], z:
    bounds = [(-400, 300), (-200, 200), (0, 400)]
    
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
    print("Robot mesh extents:", planner.robot_mesh.extents)
    print("Robot bounds:", planner.robot_mesh.bounds)

    print("\n--- Diagnostic Visualization ---")
    # planner.checker.update_position("robot", START)

    robot_transform = np.eye(4)
    robot_transform[:3, 3] = START


    # planner.checker.visualize(robot_mesh=planner.robot_mesh, robot_transform= robot_transform)

    # 5. Plan Path
    # The validity checker now uses manager.update_position("robot", state)
    # In main():
    start_quat = euler_to_quat(0, 0, 0)       # identity — shaft along X at start
    goal_quat  = euler_to_quat(0, 90, 0)      # 90° about Y — adjust to match your geometry

    print(f"\nPlanning from {START} to {GOAL}...")
    
    waypoints = planner.plan_path(
        start=START,
        goal=GOAL,
        planner_type='rrt',
        start_orientation=start_quat,
        goal_orientation=goal_quat,
        max_time = 500.0,
        goal_tolerance=2.0
    )

    # 6. Result Handling
    if waypoints:
        planner.save_path(waypoints, 'planned_path.npy')

        # 2D tree overview — try all three projections
        planner.plot_tree_matplotlib(waypoints, projection='xz')  # side view
        planner.plot_tree_matplotlib(waypoints, projection='xy')  # top view

        # 3D static snapshot
        planner.visualize_polyscope(waypoints)

        # Animated traversal — 10 interpolation steps between waypoints
        planner.animate_path_polyscope(waypoints, steps_between=10)

    else:
        print("❌ No path found. Check if the start/goal are in collision.")
    
    if waypoints is None:
        print("Checking start validity directly:")
        start_transform = np.eye(4)
        start_transform[:3,3] = START
        print("Start valid?", not planner.checker.check_mesh_against_manager(planner.robot_mesh, start_transform))

        print("Checking goal validity directly:")
        goal_transform = np.eye(4)
        goal_transform[:3,3] = GOAL
        print("Goal valid?", not planner.checker.check_mesh_against_manager(planner.robot_mesh, goal_transform))
        print("Number of obstacles in manager:", len(planner.checker.manager._names))
        print("Robot mesh vertices:", len(planner.robot_mesh.vertices))

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
    # planner.checker.visualize()


if __name__ == "__main__":
    main()