import pytest
import numpy as np
import trimesh
from collision import CollisionChecker3D

@pytest.fixture
def checker():
    """Fixture to initialize a fresh checker for each test."""
    cc = CollisionChecker3D(models_folder='models')
    yield cc
    cc.clear()

def test_identical_mesh_collision(checker):
    """Edge Case: Two identical boxes at the exact same coordinates."""
    box = trimesh.creation.box(extents=(10, 10, 10))
    checker.add_mesh(box, name="box1", position=(0, 0, 0))
    checker.add_mesh(box, name="box2", position=(0, 0, 0))
    
    collisions = checker.check_all_collisions()
    # Should detect a collision between box1 and box2
    assert any(set(['box1', 'box2']).issubset(set(c)) for c in collisions)

def test_flush_surfaces(checker):
    """Edge Case: Two boxes that are perfectly touching but not overlapping."""
    box = trimesh.creation.box(extents=(10, 10, 10))
    # Box 1 is at 0,0,0 (extending from -5 to 5)
    checker.add_mesh(box, name="box1", position=(0, 0, 0))
    # Box 2 is at 10,0,0 (extending from 5 to 15) - surfaces touch at X=5
    checker.add_mesh(box, name="box2", position=(10, 0, 0))
    
    details = checker.get_collision_details("box1", "box2")
    # Distance should be effectively 0
    assert details['distance'] == pytest.approx(0.0, abs=1e-7)

def test_extreme_distance(checker):
    """Edge Case: Objects placed very far apart."""
    box = trimesh.creation.box(extents=(1, 1, 1))
    checker.add_mesh(box, name="near", position=(0, 0, 0))
    checker.add_mesh(box, name="far", position=(10000, 10000, 10000))
    
    collisions = checker.check_all_collisions()
    assert len(collisions) == 0
    
    details = checker.get_collision_details("near", "far")
    assert details['distance'] > 10000

def test_robot_self_exclusion(checker):
    """Logic Check: Ensure the 'robot' doesn't collide with itself in the manager."""
    box = trimesh.creation.box(extents=(5, 5, 5))
    checker.add_mesh(box, name="robot", position=(0, 0, 0))
    
    # Internal collision check of the manager itself
    collisions = checker.check_all_collisions()
    # A single object should never be in collision with itself
    assert len(collisions) == 0

def test_dynamic_update_collision(checker):
    """Logic Check: Moving a mesh into a collision state."""
    box = trimesh.creation.box(extents=(10, 10, 10))
    checker.add_mesh(box, name="obstacle", position=(0, 0, 0))
    checker.add_mesh(box, name="robot", position=(50, 50, 50)) # Starts clear
    
    assert len(checker.check_all_collisions()) == 0
    
    # Move robot to collide with obstacle
    checker.update_position("robot", (2, 2, 2))
    collisions = checker.check_all_collisions()
    assert len(collisions) > 0