# import pytest
# import math
# import time
# import os
# import sys
# import psutil
# from unittest.mock import MagicMock, patch
# from shapely import Point, box

# # Get the directory of the current script (HW3/tests/)
# current_dir = os.path.dirname(os.path.abspath(__file__))
# # Get the parent directory (HW3/)
# parent_dir = os.path.dirname(current_dir)
# # Add the parent directory to sys.path so we can find Vehicles.py
# sys.path.insert(0, parent_dir)

# from Map_Generator import Map
# from firetruck import Firetruck

# @pytest.fixture
# def basic_setup():
#     """Sets up a standard 50x50 map (250m x 250m) and a Firetruck."""
#     grid_size = 50
#     cell_size = 5
#     m = Map(Grid_num=grid_size, cell_size=cell_size, fill_percent=0.1)
    
#     start_pose = (5.0, 5.0, 0.0)
#     goal_pose = (100.0, 100.0, 0.0)
    
#     truck = Firetruck(start_pose, m, goal_pose)
#     return truck, m

# # --- TEST 1: BOUNDARY LIMITS ---
# def test_world_boundary_limits(basic_setup):
#     """Checks if the 250m hardcoded limit is breaking the 250m map."""
    
#     # Test a point far beyond the 250m hardcoded box
#     truck = Firetruck((10,10,0),Map(50,5,0.0),(100,100,0))
#     far_point = (200.0, 200.0, 0.0)
    
#     is_valid = truck.is_state_valid(far_point)
#     assert is_valid, (
#         f"State at {far_point} is invalid. Check if is_state_valid() "
#         "is still hardcoded to a 250m box while the map is 640m."
#     )

# # --- TEST 2: SNAP VS. THRESHOLD ---
# def test_discretization_vs_goal_threshold(basic_setup):
#     """Checks if the snapping resolution is too coarse for the goal threshold."""
#     truck, _ = basic_setup
    
#     # Your current settings: snap_res = 22.5, angle_threshold = 15
#     # If we snap to 22.5, we can NEVER satisfy a 15-degree tolerance 
#     # unless the goal is exactly on a 22.5 multiple.
    
#     angle_snap = 15 # from snap_to_grid
#     angle_threshold = 15 # from is_near_goal
    
#     assert angle_snap <= angle_threshold, (
#         f"Logic Error: Angle snap ({angle_snap}°) is larger than goal threshold ({angle_threshold}°). "
#         "The vehicle will 'jitter' around the goal but never land in the valid zone."
#     )

# # --- TEST 3: START STATE VALIDITY ---
# def test_start_state_not_blocked(basic_setup):
#     """Ensures the truck isn't 'born' inside an obstacle."""
#     truck, m = basic_setup
    
#     # We test the start_pos directly
    
#     is_valid = truck.is_state_valid(truck.start_pos)
#     assert is_valid, (
#         "The start_pos itself is invalid! This usually means the safety buffer "
#         "in Map.generate_safe_map is too small for the truck's 4.9m width."
#     )

# # --- TEST 4: MOTION PRIMITIVE MOVEMENT ---
# def test_all_primitives_move(basic_setup):
#     truck, _ = basic_setup
#     mp = truck.calculate_motion_primitives(step_distance=1.0)
    
#     for phi, (dx, dy, dtheta) in mp.items():
#         # Ensure the vehicle actually displaced somewhere
#         # We use math.hypot for the total distance moved
#         distance = math.sqrt(dx**2 + dy**2)
#         assert distance > 0, f"Primitive for steering angle {phi} resulted in no movement!"

# # --- TEST 5: NEIGHBOR GENERATION ---
# def test_neighbor_validity_flow(basic_setup):
#     """Tests if a single step generates at least one valid neighbor."""
#     truck, _ = basic_setup
#     mp = truck.calculate_motion_primitives(step_distance=2.0)
#     neighbors = truck.get_neighbors(truck.start_pos, mp)
    
#     valid_neighbors = [n for n in neighbors if truck.is_state_valid(n)]
    
#     assert len(valid_neighbors) > 0, (
#         "No valid neighbors found from start! Even in an empty map, "
#         "this means neighbors are either going out of bounds or colliding immediately."
#     )
# # -- TEST6: 
# def test_start_has_valid_neighbors(basic_setup):
#     truck, m = basic_setup
#     mp = truck.calculate_motion_primitives(step_distance=2.0)
#     neighbors = truck.get_neighbors(truck.start_pos, mp)
    
#     # Track why they fail
#     valid_count = 0
#     reasons = {"boundary": 0, "obstacle": 0}
    
#     for n in neighbors:
#         if truck.is_state_valid(n):
#             valid_count += 1
#         else:
#             # Check why
#             footprint = truck.get_footprint(*n)
#             limit = m.grid_num * m.cell_size
#             world_box = box(0.1, 0.1, limit - 0.1, limit - 0.1)
#             if not footprint.within(world_box):
#                 reasons["boundary"] += 1
#             else:
#                 reasons["obstacle"] += 1
                
#     assert valid_count > 0, (
#         f"Start state is 'trapped'. Total neighbors: {len(neighbors)}. "
#         f"Fails: {reasons['boundary']} out of bounds, {reasons['obstacle']} collisions."
#     )
# def test_clearance_after_move(basic_setup):
#     truck, m = basic_setup
#     # Move the truck 1 unit forward
#     x,y,theta =truck.start_pos
#     next_state = (x+1,y,theta)
    
#     # If the truck is 4.9m wide, a 1m move shouldn't necessarily clear an obstacle 
#     # but it SHOULD be a valid state if the start was valid.
#     if truck.is_state_valid(truck.start_pos):
#         assert truck.is_state_valid(next_state), (
#             "Vehicle became invalid after a tiny forward move in empty space. "
#             "Check if the world boundary box is too restrictive."
#         )

# def test_goal_is_reachable_after_snapping(basic_setup):
#     truck, _ = basic_setup
#     goal = truck.goal_state # (100, 100, 0)
    
#     # Create a state that is 0.2m away from the goal
#     x,y,theta = goal
#     near_state = (x - 0.2, y, theta)
    
#     # 1. Check raw state
#     assert truck.is_near_goal(near_state, goal), "Raw state 0.2m away should be 'near goal'"
    
#     # 2. Check snapped state (This is what A* actually uses)
#     snapped = truck.snap_to_grid(near_state, res=0.5)
#     assert truck.is_near_goal(snapped, goal), (
#         f"Snapped state {snapped} is no longer 'near' the goal {goal}. "
#         "Your grid resolution is pushing the vehicle outside the goal's acceptance window!"
#     )
# def test_narrow_passage_feasibility(basic_setup):
#     truck, m = basic_setup
#     # Create a 'gate' of two obstacles with a 5.1m gap
#     # Since the truck is 4.9m, it SHOULD fit.
#     m.obstacle_set = {(5, 5), (5, 7)} # Assuming 5m cells, this is a 5m gap
    
#     # Try to move through the center of the gap (row 5, col 6)
#     gap_center = (5 * 5, 6 * 5, 0) 
    
#     assert truck.is_state_valid(gap_center), (
#         f"Truck (width {truck.width}m) cannot fit through a 5m gap! "
#         "This is why your planner finds no path in a crowded map."
#     )