import unittest
from unittest.mock import MagicMock, patch
import math
import pytest
from shapely.geometry import Point

# Assuming your class is in delivery_planner.py
import sys
import os

# Get the path of the current file's directory (tests/)
current_dir = os.path.dirname(__file__)
# Go up one level to the parent directory (HW3/)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
# Add it to the system path
sys.path.insert(0, parent_dir)


from Vehicles import Delivery


# We use a fixture to create a robot instance for every test
@pytest.fixture
def mock_delivery():
    # We patch the Map class WHERE IT IS USED (inside Vehicles.py)
    with patch("Vehicles.Map") as MockMap:
        # Create a fake map instance
        fake_map = MockMap.return_value
        fake_map.grid_size = 12
        fake_map.goal_pos = (10, 10)
        # Set a controlled list of obstacles
        fake_map.obstacle_coordinate_list = [(2, 2), (2, 3)]

        # Instantiate the robot; it will now use our fake_map
        robot = Delivery()
        # Ensure the robot processes our fake obstacles
        robot.prepare_obstacles(fake_map.obstacle_coordinate_list)
        return robot


# --- The Tests ---


def test_coordinate_binning(mock_delivery):
    """Verifies that two points 1mm apart are treated as the same state."""
    state_a = (1.0001, 2.0000, 90.0)
    state_b = (0.9999, 2.0000, 90.0)

    bin_a = mock_delivery.snap_to_grid(state_a, res=0.1)
    bin_b = mock_delivery.snap_to_grid(state_b, res=0.1)

    # These should be identical keys now
    assert bin_a == bin_b
    assert bin_a == (1.0, 2.0, 90.0)


def test_heuristic_math(mock_delivery):
    """Test that our Euclidean distance is correct."""
    # Goal is (10, 10)
    dist = mock_delivery.calculate_heurisitic((7, 6, 0), (10, 10))
    # distance = sqrt(3^2 + 4^2) = 5
    assert dist == 5.0


def test_collision_detection(mock_delivery):
    """Verify the robot detects our 'fake' obstacles."""
    # We put an obstacle at (2, 2). Robot center at (2.5, 2.5) should hit it.
    assert mock_delivery.is_collision((2.5, 2.5, 0)) is True

    # Far away point should be safe
    assert mock_delivery.is_collision((8, 8, 0)) is False


def test_reconstruct_path(mock_delivery):
    """Check if the breadcrumb trail is built correctly."""
    came_from = {(1, 1, 0): (0, 0, 0), (2, 2, 0): (1, 1, 0)}
    path = mock_delivery.reconstruct_path(came_from, (2, 2, 0))
    assert path == [(0, 0, 0), (1, 1, 0), (2, 2, 0)]


@pytest.mark.parametrize(
    "state, expected_safe",
    [
        ((6.0, 6.0, 0.0), True),  # Perfectly in the middle
        ((0.1, 6.0, 0.0), False),  # Left edge clips X=0 (-0.25m)
        ((11.9, 6.0, 0.0), False),  # Right edge clips X=12 (+0.25m)
        ((6.0, 0.1, 90.0), False),  # Top edge clips Y=0 when rotated 90 deg
        ((15.0, 5.0, 0.0), False),  # Completely outside
    ],
)
def test_grid_boundaries(mock_delivery, state, expected_safe):
    """Ensures the robot cannot move or clip its body outside the 12x12 grid."""
    is_safe = mock_delivery.is_state_valid(state)
    assert is_safe == expected_safe, f"Boundary check failed for state {state}"


@pytest.mark.parametrize(
    "state, description",
    [
        (
            (0.2, 5.0, 0.0),
            "Left side poke-out",
        ),  # Center is at 0.2, but width/2 is 0.35 -> -0.15 (Out!)
        ((11.8, 5.0, 0.0), "Right side poke-out"),  # 11.8 + 0.35 = 12.15 (Out!)
        (
            (5.0, 0.2, 90.0),
            "Top side poke-out",
        ),  # Rotated 90 deg, length becomes height
        ((5.0, 11.8, 90.0), "Bottom side poke-out"),
    ],
)
def test_strict_boundaries(mock_delivery, state, description):
    """
    This test fails if any part of the robot's shell leaves the 0-12 range.
    """
    # This should return FALSE (not safe) because it's poking out
    is_safe = mock_delivery.is_state_valid(state)
    assert (
        is_safe is False
    ), f"Cheating detected! {description} at {state} should be invalid."


import time

# def test_planner_performance(mock_delivery):
#     start_time = time.time()
#     # Run a complex but possible plan
#     mock_delivery.plan(goal=(10, 10))
#     duration = time.time() - start_time

#     print(f"Planning took {duration:.4f} seconds")
#     # Assert that planning stays under a reasonable limit (e.g., 2 seconds)
#     assert duration < 2.0


def test_collision_logic_integrity(mock_delivery):
    # Setup: Wall at X=5, from Y=0 to Y=10
    wall = [(y, 5) for y in range(11)]
    mock_delivery.prepare_obstacles(wall)

    # Case A: Center is safe, but right side clips the wall
    # Center at 4.7 + width(0.35) = 5.05 (Collision!)
    assert mock_delivery.is_state_valid((4.7, 5.0, 0.0)) is False

    # Case B: Completely inside the wall
    assert mock_delivery.is_state_valid((5.5, 5.0, 0.0)) is False

    # Case C: Safe distance from wall
    assert mock_delivery.is_state_valid((4.0, 5.0, 0.0)) is True


@pytest.mark.parametrize(
    "state, expected, reason",
    [
        # --- BOUNDARY CHECKS ---
        ((6.0, 6.0, 0.0), True, "Safe center position"),
        ((0.2, 6.0, 0.0), False, "Left edge poking out (0.2 - 0.35 = -0.15)"),
        ((11.8, 6.0, 0.0), False, "Right edge poking out (11.8 + 0.35 = 12.15)"),
        ((6.0, 0.2, 90.0), False, "Top edge poking out after 90deg rotation"),
        # --- OBSTACLE CHECKS (Assumes obstacle at 2,2) ---
        ((2.5, 2.5, 0.0), False, "Direct hit: Center is inside obstacle box"),
        ((1.7, 2.5, 0.0), False, "Partial hit: Right side of robot clips the box"),
        ((1.5, 2.5, 0.0), True, "Near miss: Robot is close but not touching"),
    ],
)
def test_safety_logic(mock_delivery, state, expected, reason):
    """
    Test individual states against the boundary and a fixed obstacle.
    Obstacle is mocked at (2, 2) which spans x:[2,3], y:[2,3].
    """
    # mock_delivery fixture (from previous step) sets up obstacle at (2,2)
    result = mock_delivery.is_state_valid(state)
    assert result == expected, f"Failed: {reason}. State: {state}"


def test_empty_map_behavior(mock_delivery):
    """Ensures that if there are no obstacles, we don't crash and only check boundaries."""
    mock_delivery.full_obstacle_geometry = None
    # This is physically in bounds but would have hit the (2,2) obstacle
    assert mock_delivery.is_state_valid((2.5, 2.5, 0)) is True


def test_gap_clearance(mock_delivery):
    """Checks if the robot can physically fit in a 1.0m wide gap."""
    # Create two pillars with a 1.0m gap between them at x=5
    gap_obstacles = [(4, 5), (6, 5)]
    mock_delivery.prepare_obstacles(gap_obstacles)

    # A 0.57m wide robot should easily fit through a 1.0m gap at y=5
    # Center at (5.5, 5.0)
    assert mock_delivery.is_state_valid((5.5, 5.0, 0.0)) is True


def test_impossible_maze_fails_correctly(mock_delivery):
    """
    Ensures the planner returns None when a solid wall blocks the goal
    and boundary checking prevents 'cheating' around the outside.
    """
    # 1. Define a solid horizontal wall at Y=5 from X=0 to X=12
    # This leaves no physical gap for the robot.
    wall_obstacles = [(5, x) for x in range(13)]

    with patch("Vehicles.Map") as MockMap:
        fake_map = MockMap.return_value
        fake_map.goal_pos = (6.0, 10.0)  # Goal is behind the wall
        fake_map.obstacle_coordinate_list = wall_obstacles
        fake_map.grid_size = 12

        # 2. Setup the Robot
        robot = Delivery()
        robot.start_pos = (6.0, 1.0, 0.0)  # Start is in front of the wall
        robot.prepare_obstacles(wall_obstacles)

        # 3. RUN THE PLANNER
        # If is_state_valid is working, the robot cannot go around [0,12]
        # and cannot go through the wall.
        path = robot.plan(goal=fake_map.goal_pos, step_size=100000)

        # 4. ASSERTION: Path must be None
        assert (
            path is None
        ), "Planner 'cheated' or found a ghost path through a solid wall!"
