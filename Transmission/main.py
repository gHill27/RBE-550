from planner import RRTPlanner3D
import numpy as np

# Define your transmission's physical bounds
# From your parameters: case_length=280, case_width=210, case_height=300
bounds = [
    (-200, 200),   # X: enough to go through case
    (-150, 150),   # Y: width of case
    (0, 350)       # Z: from bottom to above bearing_offset_height (215)
]

# Create planner
planner = RRTPlanner3D(bounds)

# Add obstacles
planner.add_obstacle('transmission_case.scad', 'Case', position=(0, 0, 0))
planner.add_obstacle('secondary_shaft.scad', 'Countershaft', 
                    position=(165, 0, 112.5))  # From your translation

# Set mainshaft as robot
# Starting outside the case (left side)
planner.set_robot(
    scad_file='primary_shaft.scad',
    start_position=(-150, 0, 127.5)  # X: outside left, Z: aligned with bearing hole
)

# Plan path through the bearing hole
waypoints = planner.plan_path(
    start=(-150, 0, 127.5),   # Start outside left side
    goal=(150, 0, 127.5),     # Goal fully inserted through hole
    planner_type='rrt_connect',
    max_time=10.0,
    goal_tolerance=5.0
)

if waypoints:
    print(f"✓ Found insertion path with {len(waypoints)} waypoints")
    planner.visualize_path(waypoints)