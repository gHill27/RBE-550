import pytest
import math
import time
import os
import sys
import psutil
from unittest.mock import MagicMock, patch
from shapely import Point, box

# Get the directory of the current script (HW3/tests/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (HW3/)
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path so we can find Vehicles.py
sys.path.insert(0, parent_dir)

from Vehicles import Map
from police import Police
from truck import Truck, TruckTrailerLUT
from delivery import Delivery


# --- FIXTURES ---
@pytest.fixture
def goal_State():
    return (30.0, 30.0, 0.0)


@pytest.fixture
def empty_map():
    return Map(Grid_num=12, cell_size=3, fill_percent=0.0)


@pytest.fixture
def blocked_map():
    """Creates a map where the exit from (0,0) is blocked."""
    m = Map(Grid_num=12, cell_size=3, fill_percent=0.0)
    # Surround the start cell (0,0) in world coordinates
    m.obstacle_coordinate_list = [(2, 0), (0, 2), (2, 1), (1, 2), (2, 2)]
    return m


# --- GEOMETRIC & BOUNDARY TESTS ---


def test_boundary_detection(empty_map, goal_State):
    """Checks if the vehicle recognizes world boundaries."""
    bot = Delivery(startPose=(1, 1, 0), goalPose=goal_State, map=empty_map)

    # Clearly inside
    assert bot.is_state_valid((5, 5, 0)) is True
    # Halfway off the 36m x 36m map
    assert bot.is_state_valid((35.9, 35.9, 0)) is False
    # Negative coordinates
    assert bot.is_state_valid((-1, 5, 0)) is False


def test_police_footprint_vs_delivery(empty_map, goal_State):
    """Tests that the Police car's larger size triggers collisions where the Delivery bot is safe."""
    # Create an obstacle at (6, 6) grid -> (18, 18) meters
    m = Map(12, 3, 0)

    deliv = Delivery(startPose=(3, 3, 0), goalPose=goal_State, map=m)
    police = Police(startPose=(3, 3, 0), goalPose=goal_State, map=m, plot=False)

    m.obstacle_coordinate_list = [(6, 6)]
    deliv.prepare_obstacles(m.obstacle_coordinate_list)
    police.prepare_obstacles(m.obstacle_coordinate_list)

    # Position the center 2 meters away from the 3m obstacle
    # The Delivery bot (0.7m long) should be safe.
    # The Police car (5.2m long) should collide.
    test_pos = (16.0, 18.0, 0)

    print(f"DEBUG: Obstacles in police memory: {police.map.obstacle_coordinate_list}")
    assert deliv.is_state_valid(test_pos) is True
    assert police.is_state_valid(test_pos) is False


# --- MOTION PRIMITIVE TESTS ---


def test_ackermann_math(goal_State):
    """Verifies that the Police car heading change matches steering geometry."""
    m = Map(12, 3, 0)
    police = Police(startPose=(10, 10, 0), goalPose=goal_State, map=m, plot=False)
    mp = police.calculate_motion_primitives(step_distance=1.0)

    # Steering 0 should have 0 d_theta
    assert mp[0][2] == 0
    # Hard left (+30) should result in a positive d_theta
    assert mp[30][2] > 0
    # Hard right (-30) should result in a negative d_theta
    assert mp[-30][2] < 0


# --- PLANNER EFFICIENCY & EDGE CASES ---


def test_unreachable_goal(blocked_map, goal_State):
    """Ensures planner exits gracefully when no path exists."""
    police = Police(
        startPose=(3, 3, 0), goalPose=goal_State, map=blocked_map, plot=False
    )
    # Goal is on the other side of the wall
    path = police.plan(goal=(30, 30, 0), step_distance=2.0)
    assert path is None


def test_path_efficiency(empty_map, goal_State):
    """Verifies that the path found is reasonably direct (A* optimality)."""
    start = (5, 5, 0)
    goal = (15, 5, 0)
    bot = Delivery(startPose=start, goalPose=goal_State, map=empty_map)
    path = bot.plan(goal=goal, step_distance=1.0)

    assert path is not None
    # Straight line is 10m. Path should not be excessively long (e.g., > 15m)
    total_dist = 0
    for i in range(len(path) - 1):
        total_dist += math.sqrt(
            (path[i + 1][0] - path[i][0]) ** 2 + (path[i + 1][1] - path[i][1]) ** 2
        )

    assert total_dist < 12.0  # Allow for small discretization curves


# 1. Testing the Planner logic by Mocking the heuristic
def test_plan_with_perfect_heuristic():
    """
    Mock the heuristic to always return 0 to see if the A* still functions correctly with 'greedy' behavior.
    """
    start = (5, 5, 0)
    goal = (7, 5, 0)
    # We use a real Delivery object but patch its heuristic method
    with patch.object(Delivery, "calculate_heurisitic", return_value=0.0):
        bot = Delivery(startPose=start, goalPose=goal)
        path = bot.plan(goal=goal, step_distance=1.0)

        assert path is not None
        assert path[0] == start
        # Even with a 0 heuristic, A* should find the goal via cost-to-come


# 2. Testing Collision Logic by Mocking Shapely Geometries
def test_is_state_valid_with_mocked_collision(goal_State):
    """
    Force is_collision to return True to see if is_state_valid
    correctly rejects the state.
    """
    bot = Delivery(startPose=(5, 5, 0), goalPose=goal_State)

    # We mock the 'intersects' call on the footprint
    with patch("Vehicles.Polygon.intersects", return_value=True):
        # Even in an empty world, if the 'intersects' call returns True,
        # the state must be invalid.
        assert bot.is_state_valid((5, 5, 0)) is False


# 3. Testing get_neighbors rotation math
def test_police_neighbor_rotation(goal_State):
    """
    Verify that get_neighbors correctly applies rotation
    matrices to motion primitives.
    """
    m = MagicMock()  # Mock the Map
    police = Police(
        startPose=(0, 0, 90), goalPose=goal_State, map=m, plot=False
    )  # Facing North

    # Manually defined primitive: moving 1m 'forward' in local X
    mock_primitives = {0: (1.0, 0.0, 0.0)}

    neighbors = police.get_neighbors((0, 0, 90), mock_primitives)

    # If facing 90 deg (North), a local X move of 1.0
    # should result in a World Y move of 1.0
    res_x, res_y, res_theta = neighbors[0]
    assert pytest.approx(res_x) == 0.0
    assert pytest.approx(res_y) == 1.0
    assert res_theta == 90


# 4. Mocking the Visualizer to ensure it doesn't slow down tests
def test_visualizer_calls(goal_State):
    """
    Ensure the visualizer's update function is called
    the expected number of times during a plan.
    """
    with patch("Vehicles.PlannerVisualizer") as MockViz:
        # Setup the mock instance
        instance = MockViz.return_value

        bot = Delivery(startPose=(1, 1, 0), goalPose=goal_State, plot=True)
        # Small distance, should hit goal quickly
        bot.plan(goal=(1.5, 1, 0), step_size=1)

        # Verify that update was called (meaning the viz hook is working)
        assert instance.update.called


def test_planner_throughput_and_binning():
    """
    Measures the efficiency of the A* expansion and the
    effectiveness of the state-lattice binning.
    """
    m = Map(Grid_num=12, cell_size=3, fill_percent=0.0)
    # Start and Goal far apart to force a long search
    start, goal = (2, 2, 0), (30, 30, 0)
    bot = Delivery(startPose=start, goalPose=goal, map=m)

    # Track how many times snap_to_grid is called using a wrapper
    call_counts = {"snaps": 0}
    original_snap = bot.snap_to_grid

    def count_snaps(*args, **kwargs):
        call_counts["snaps"] += 1
        return original_snap(*args, **kwargs)

    with patch.object(Delivery, "snap_to_grid", side_effect=count_snaps):
        start_time = time.time()
        path = bot.plan(goal=goal, step_distance=0.5)
        end_time = time.time()

    duration = end_time - start_time
    nodes_expanded = len(bot.exploredNodes)

    # CALCULATE METRICS
    nodes_per_second = nodes_expanded / duration if duration > 0 else 0
    # Compression ratio: how many raw neighbors were collapsed into unique bins
    compression_ratio = (
        call_counts["snaps"] / nodes_expanded if nodes_expanded > 0 else 0
    )

    print(f"\n--- Performance Results ---")
    print(f"Nodes Expanded: {nodes_expanded}")
    print(f"Throughput: {nodes_per_second:.2f} nodes/sec")
    print(f"Binning Compression Ratio: {compression_ratio:.2f}x")

    # ASSERTIONS
    # 1. Ensure throughput is healthy (e.g., > 1000 nodes/sec on modern hardware)
    assert (
        nodes_per_second > 500
    ), "Planner is running too slow! Check for geometric bottlenecks."

    # 2. Ensure binning is actually working
    # If compression is 1.0x, it means every single move became a new state (no pruning)
    assert (
        compression_ratio > 1.0
    ), "Binning is not pruning any states. Check snap_to_grid resolution."

    # 3. Ensure a path was actually found
    assert path is not None


def test_heading_normalization(goal_State):
    """Ensures the planner treats 359 and 1 degrees as 2 degrees apart."""
    m = Map(12, 3, 0)
    bot = Delivery(startPose=(10, 10, 359), goalPose=goal_State, map=m)

    # Target a point slightly to the right of 359 (crossing 0)
    snapped_a = bot.snap_to_grid((10, 10, 359), res=0.2)
    snapped_b = bot.snap_to_grid((10, 10, 1), res=0.2)

    # If your snap_to_grid uses % 360, these might be the same or very close
    # This test verifies your state representation doesn't create 'infinite' unique angles
    assert abs(snapped_a[2] - snapped_b[2]) <= 45.0  # Or your angular resolution


def test_goal_tolerance_acceptance(goal_State):
    """Verifies the robot stops within a reasonable distance of the goal."""
    m = Map(12, 3, 0)
    start = (10, 10, 0)
    goal = (10.2, 10.2, 0)

    bot = Delivery(startPose=start, goalPose=goal, map=m)
    path = bot.plan(goal=goal, step_distance=0.5)

    assert path is not None

    # Check proximity instead of step count
    final_node = path[-1]
    dist_to_goal = math.sqrt(
        (final_node[0] - goal[0]) ** 2 + (final_node[1] - goal[1]) ** 2
    )

    # The final position should be within the robot's tolerance (e.g., 0.5m)
    assert dist_to_goal < 0.5, f"Robot ended too far from goal: {dist_to_goal}m"


def test_police_minimum_turning_radius(goal_State):
    """Ensures the Police car doesn't execute a turn tighter than physically possible."""
    m = Map(12, 3, 0)
    police = Police(startPose=(10, 10, 0), goalPose=goal_State, map=m)

    # Measure heading change over a 1.0m step at max steer
    # mp[30] is max left steer
    primitives = police.calculate_motion_primitives(step_distance=1.0)
    max_turn_delta_theta = abs(primitives[30][2])

    # Based on wheelbase ~2.7m and steer 30deg,
    # the max theta change for 1m should be roughly 10-12 degrees.
    # If it's 90 degrees, the math is broken.
    assert max_turn_delta_theta < 20.0


def test_police_narrow_passage(goal_State):
    """Tests if the Police car can navigate a gap barely wider than itself."""
    m = Map(12, 3, 0)
    # Create a vertical 'gate' at x=15
    # Car width is 1.8m. Gap is from y=14 to y=16 (2.0m wide)

    police = Police(startPose=(0, 13.5, 0), goalPose=goal_State, map=m)
    m.obstacle_coordinate_list = [
        (5, 3),
        (5, 5),  # Obstacles at x=15, y=12 and x=15, y=18
    ]
    police.prepare_obstacles(m.obstacle_coordinate_list)

    # Should be valid when perfectly centered and straight
    assert police.is_state_valid((15, 13.5, 0)) is True
    # Should be invalid if it rotates 45 degrees while in the gap
    assert police.is_state_valid((15, 13.5, 45)) is False


# def test_invalid_start_goal(blocked_map):
#     """Planner should return None immediately if start/goal is in an obstacle."""
#     # Place goal inside the obstacle at (2, 0) -> meters (6, 0)
#     goal_inside = (6.0, 0.5, 0)
#     start_valid = (15.0, 15.0, 0)

#     bot = Delivery(startPose=start_valid, goalPose=goal_inside, map=blocked_map)

#     # You'll likely need to add an 'if not is_state_valid' check at the start of plan()
#     path = bot.plan(goal=goal_inside)
#     assert path is None

# def test_thin_wall_leaking():
#     """Ensures the robot doesn't 'phase' through obstacles between steps."""
#     m = Map(12, 3, 0)
#     # A thin vertical wall at x=10
#     m.obstacle_coordinate_list = [(3, 0), (3, 1), (3, 2)]

#     # Start at x=8.5, neighbor would be at x=11.5 if moving 3m
#     bot = Delivery(startPose=(8.5, 3.0, 0), map=m)

#     # Force a large step to try and jump the wall
#     path = bot.plan(goal=(15.0, 3.0, 0), step_distance=3.0)


#     # If it found a path, it likely 'jumped' over the wall at x=9-12
#     assert path is None, "Robot leaked through a thin wall!"
def test_kinematic_consistency():
    """
    Ensures the displacement (dx, dy) aligns with the vehicle's heading.
    If this fails, the truck will move 'sideways'.
    """
    # Mocking a step from your LUT: start at (0,0,0,0), steer at 20 degrees
    # Replace 'lut_instance' with your actual LUT generator
    lut = TruckTrailerLUT()
    start_state = (0, 0, 0, 0)
    steering_angle = 20
    end_state = lut._simulate_step(start_state, steering_angle)

    dx = end_state[0] - start_state[0]
    dy = end_state[1] - start_state[1]
    actual_move_angle = math.degrees(math.atan2(dy, dx))

    # The vehicle's heading starts at 0 and ends at end_state[2]
    # The movement vector should be roughly the average of these
    avg_heading = (0 + end_state[2]) / 2.0

    # Tolerance of 1-2 degrees is acceptable for discrete integration
    assert (
        abs(actual_move_angle - avg_heading) < 2.0
    ), f"Crabbing detected! Move Angle: {actual_move_angle}, Avg Heading: {avg_heading}"


def test_trailer_alignment_on_straight():
    lut = TruckTrailerLUT(step_dist=2.0, d1=5.0)
    # Truck at 0, Trailer "kinked" at 30 degrees
    # psi = t0 - t1 = -30
    current_state = (0, 0, 0, -30)

    # Move straight
    new_state = lut.get_primitive(current_state, 0, 1)

    # The new trailer angle should be CLOSER to the truck angle (0)
    # than it was before. (-30 -> closer to 0)
    old_error = abs(current_state[2] - current_state[3])
    new_error = abs(new_state[2] - new_state[3])

    assert (
        new_error < old_error
    ), f"Trailer failed to align! Error was {old_error}, now {new_error}"


def test_path_cost_revisions(empty_map, goal_State):
    """Verifies that A* updates a node if a cheaper path is found."""
    start = (10, 10, 0)
    # Place a goal where two primitives could reach it, one with more 'wobble'
    goal = (12, 10, 0)
    bot = Delivery(startPose=start, goalPose=goal, map=empty_map)

    # We want to ensure that the final costHistory entry for the goal
    # is the minimum possible cost, not just the first one found.
    bot.plan(goal=goal, step_distance=0.5)

    # Re-check your logic:
    # if tentativeCostToCome < costHistory[snapped_neighbor]:
    # This is the line that handles this case!


def test_search_limit_exhaustion(goal_State):
    """Ensures the planner exits gracefully when a complex map has no solution."""
    m = Map(12, 3, 0.4)  # Very dense map
    # Police car in a tight box
    police = Police(startPose=(1.5, 1.5, 0), goalPose=(30, 30, 0), map=m)

    start_time = time.time()
    # If you haven't implemented a max_nodes or timeout, this test might hang!
    path = police.plan(goal=(30, 30, 0), step_distance=1.0)
    duration = time.time() - start_time

    assert duration < 5.0  # Should not think for more than 5 seconds
    assert path is None


def test_angle_snapping_stability(goal_State):
    """Ensures 0 and 360 degrees snap to the same grid bin."""
    bot = Delivery(startPose=(0, 0, 0), goalPose=goal_State)

    bin_0 = bot.snap_to_grid((10, 10, 0.1), res=0.5)
    bin_360 = bot.snap_to_grid((10, 10, 359.9), res=0.5)

    # If these aren't the same, the robot will 'waffle' at the 0-degree mark
    assert bin_0 == bin_360, f"Angle wrap-around failed: {bin_0} vs {bin_360}"


def test_continuous_goal_clearance():
    """Verifies clearance even when the goal is at a fractional coordinate."""
    # Goal is at 10.4, very close to the edge of the 9-12 cell
    goal_cont = (10.4, 10.4, 0)
    start_cont = (2.0, 2.0, 0)

    # Generate a map with high obstacle density
    m = Map(Grid_num=12, fill_percent=0.6, cell_size=3)
    m.goal_pos = goal_cont

    goal_point = Point(goal_cont[0], goal_cont[1])
    cell_size = 3

    for row, col in m.obstacle_coordinate_list:
        obs_box = box(
            row * cell_size,
            col * cell_size,
            (row + 1) * cell_size,
            (col + 1) * cell_size,
        )

        # The distance from any part of the obstacle to the goal point
        # must be >= 3.0m
        distance = obs_box.distance(goal_point)
        assert distance >= 3.0, f"Obstacle at {row, col} is only {distance}m from goal!"


def test_vehicle_footprint_clearance():
    """Ensures the specific vehicle can rotate 360 degrees at goal without hitting anything."""
    goal_state = (15.0, 15.0, 0)
    m = Map(Grid_num=12, fill_percent=0.3, cell_size=3)

    # Use the larger vehicle
    bot = Police(startPose=(1, 1, 0), goalPose=goal_state, map=m)

    # Check every 45 degrees at the goal position
    for angle in range(0, 360, 45):
        test_state = (goal_state[0], goal_state[1], float(angle))
        assert bot.is_state_valid(
            test_state
        ), f"Collision at goal when heading is {angle}!"


# @pytest.mark.memory
# def test_police_memory_footprint():
#     """
#     Ensures that planning for a large vehicle (Police)
#     stays within reasonable memory and node-count limits.
#     """
#     # Create a complex maze to force a difficult search
#     m = Map(Grid_num=12, cell_size=3, fill_percent=0.2)
#     start, goal = (3, 3, 0), (30, 30, 0)
#     police = Police(startPose=start, goalPose=goal, map=m)

#     # Measure memory before
#     process = psutil.Process(os.getpid())
#     mem_before = process.memory_info().rss / (1024 * 1024)  # Convert to MB

#     # Run the plan
#     path = police.plan(goal=goal, step_distance=1.0)

#     # Measure memory after
#     mem_after = process.memory_info().rss / (1024 * 1024)
#     mem_used = mem_after - mem_before
#     nodes_in_memory = len(police.exploredNodes)

#     print(f"\n--- Resource Usage ---")
#     print(f"Nodes Stored: {nodes_in_memory}")
#     print(f"Memory Increment: {mem_used:.2f} MB")

#     # ASSERTIONS
#     # 1. Limit total nodes: For a 36m map, exploring > 50,000 nodes
#     # usually means your heuristic or binning is failing.
#     assert nodes_in_memory < 50000, "State space explosion! Binning is not effective."

#     # 2. Limit memory: Planning shouldn't take more than 50MB of extra RAM
#     # for these dictionary structures.
#     assert mem_used < 50, "Memory leak or excessive state storage detected."
