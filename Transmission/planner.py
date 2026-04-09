#!/usr/bin/env python3
"""
ompl_planner.py - 3D RRT path planning with OpenSCAD obstacles
"""

import numpy as np
import trimesh
from typing import List, Tuple, Optional
from pathlib import Path

try:
    from ompl import base as ob
    from ompl import geometric as og
    OMPL_AVAILABLE = True
except ImportError:
    OMPL_AVAILABLE = False
    print("Warning: OMPL not installed. Run: pip install ompl")

from mesh_gen import MeshGenerator
from collision import CollisionChecker3D

class RRTPlanner3D:
    """3D RRT path planner with mesh-based collision checking"""
    
    def __init__(self, bounds: List[Tuple[float, float]], models_folder: str = 'models'):
        """
        Initialize 3D planner.
        
        Args:
            bounds: [(x_min, x_max), (y_min, y_max), (z_min, z_max)]
            models_folder: Folder containing OpenSCAD models
        """
        if not OMPL_AVAILABLE:
            raise ImportError("OMPL not installed. Run: pip install ompl")
        
        self.bounds = bounds
        self.models_folder = models_folder  # Store this
        self.checker = CollisionChecker3D(models_folder=models_folder)
        self.robot_mesh = None
        self.robot_radius = None
        self.robot_start = None
        
        # Setup OMPL state space (3D)
        self.space = ob.RealVectorStateSpace(3)
        bounds_obj = ob.RealVectorBounds(3)
        for i, (low, high) in enumerate(bounds):
            bounds_obj.setLow(i, low)
            bounds_obj.setHigh(i, high)
        self.space.setBounds(bounds_obj)
        
        self.si = ob.SpaceInformation(self.space)
    
    def add_obstacle(self, scad_file: str, name: str, 
                     position: Tuple[float, float, float] = (0, 0, 0),
                     parameters: dict = None):
        """Add obstacle from OpenSCAD file in models folder"""
        self.checker.add_from_scad(scad_file, name, position, parameters)
        print(f"  Obstacle: {name} at {position}")
    
    def set_robot(self, scad_file: str = None, radius: float = None,
                  start_position: Tuple[float, float, float] = (0, 0, 0)):
        """
        Set robot as either mesh or sphere.
        
        Args:
            scad_file: OpenSCAD file for robot mesh (in models folder)
            radius: Use sphere robot with given radius
            start_position: Starting position
        """
        if scad_file:
            # Use the same models_folder as the checker
            generator = MeshGenerator(models_folder=self.models_folder)
            self.robot_mesh = generator.from_scad(scad_file)
            self.robot_radius = None
            print(f"✓ Robot loaded from {self.models_folder}/{scad_file}")
        elif radius:
            self.robot_mesh = trimesh.primitives.Sphere(radius=radius)
            self.robot_radius = radius
            print(f"✓ Robot set as sphere (radius={radius})")
        else:
            raise ValueError("Must provide either scad_file or radius")
        
        self.robot_start = start_position
        
        # Setup collision checking
        def is_state_valid(state):
            x, y, z = state[0], state[1], state[2]
            
            # Check bounds
            if (x < self.bounds[0][0] or x > self.bounds[0][1] or
                y < self.bounds[1][0] or y > self.bounds[1][1] or
                z < self.bounds[2][0] or z > self.bounds[2][1]):
                return False
            
            # Create robot at this position
            robot_at_pos = self.robot_mesh.copy()
            robot_at_pos.apply_translation([x, y, z])
            
            # Check collision with all obstacles
            for i, obstacle in enumerate(self.checker.meshes):
                obstacle_at_pos = obstacle.copy()
                obstacle_at_pos.apply_translation(self.checker.positions[i])
                
                if robot_at_pos.intersects_mesh(obstacle_at_pos):
                    return False
            
            return True
        
        # Set the state validity checker
        self.si.setStateValidityChecker(ob.StateValidityChecker(is_state_valid))
        self.si.setup()
    
    def plan_path(self, 
                  start: Tuple[float, float, float],
                  goal: Tuple[float, float, float],
                  planner_type: str = 'rrt_connect',
                  max_time: float = 5.0,
                  goal_tolerance: float = 0.1) -> Optional[List[np.ndarray]]:
        """
        Plan 3D path using RRT.
        
        Args:
            start: (x, y, z) start position
            goal: (x, y, z) goal position
            planner_type: 'rrt', 'rrt_connect', or 'rrt_star'
            max_time: Maximum planning time in seconds
            goal_tolerance: Distance tolerance for goal
        
        Returns:
            List of waypoints or None if planning fails
        """
        # Create problem definition
        pdef = ob.ProblemDefinition(self.si)
        
        # Set start and goal states
        start_state = ob.State(self.space)
        start_state[0], start_state[1], start_state[2] = start
        goal_state = ob.State(self.space)
        goal_state[0], goal_state[1], goal_state[2] = goal
        
        pdef.setStartAndGoalStates(start_state, goal_state, goal_tolerance)
        
        # Select planner
        if planner_type == 'rrt':
            planner = og.RRT(self.si)
        elif planner_type == 'rrt_connect':
            planner = og.RRTConnect(self.si)
        elif planner_type == 'rrt_star':
            planner = og.RRTstar(self.si)
        else:
            planner = og.RRTConnect(self.si)
        
        planner.setProblemDefinition(pdef)
        
        # Plan
        print(f"\n🚀 Planning with {planner_type.upper()}...")
        print(f"   Start: {start}")
        print(f"   Goal: {goal}")
        print(f"   Time limit: {max_time}s")
        
        solved = planner.solve(max_time)
        
        if solved:
            print("✓ Path found!")
            path = pdef.getSolutionPath()
            
            # Simplify path
            simplifier = og.PathSimplifier(self.si)
            simplifier.simplify(path)
            
            # Extract waypoints
            waypoints = []
            for i in range(path.getStateCount()):
                state = path.getState(i)
                waypoints.append(np.array([state[0], state[1], state[2]]))
            
            # Calculate path length
            length = 0
            for i in range(1, len(waypoints)):
                length += np.linalg.norm(waypoints[i] - waypoints[i-1])
            
            print(f"   Waypoints: {len(waypoints)}")
            print(f"   Path length: {length:.2f}")
            
            return waypoints
        else:
            print("✗ No path found!")
            return None
    
    def visualize_path(self, waypoints: List[np.ndarray]):
        """Visualize obstacles and planned path"""
        scene = trimesh.Scene()
        
        # Add obstacles
        for i, mesh in enumerate(self.checker.meshes):
            mesh_copy = mesh.copy()
            mesh_copy.apply_translation(self.checker.positions[i])
            mesh_copy.visual.face_colors = [100, 100, 255, 128]
            scene.add_geometry(mesh_copy, node_name=f"obstacle_{i}")
        
        # Add path
        if len(waypoints) > 1:
            points = np.array(waypoints)
            for i in range(len(points) - 1):
                # Create line segment
                direction = points[i+1] - points[i]
                length = np.linalg.norm(direction)
                if length > 0:
                    # Create cylinder transform
                    transform = np.eye(4)
                    transform[:3, 3] = points[i]
                    # Align cylinder with direction
                    z_axis = np.array([0, 0, 1])
                    rot_axis = np.cross(z_axis, direction / length)
                    angle = np.arccos(np.dot(z_axis, direction / length))
                    if angle > 0:
                        transform[:3, :3] = trimesh.transformations.rotation_matrix(angle, rot_axis)[:3, :3]
                    
                    cylinder = trimesh.creation.cylinder(radius=0.05, height=length, transform=transform)
                    cylinder.visual.face_colors = [255, 0, 0, 255]
                    scene.add_geometry(cylinder)
            
            # Add start marker
            start_sphere = trimesh.primitives.Sphere(radius=0.1, center=points[0])
            start_sphere.visual.face_colors = [0, 255, 0, 255]
            scene.add_geometry(start_sphere)
            
            # Add goal marker
            goal_sphere = trimesh.primitives.Sphere(radius=0.1, center=points[-1])
            goal_sphere.visual.face_colors = [0, 0, 255, 255]
            scene.add_geometry(goal_sphere)
        
        scene.show()
    
    def save_path(self, waypoints: List[np.ndarray], filename: str = 'planned_path.npy'):
        """Save path to file"""
        np.save(filename, np.array(waypoints))
        print(f"✓ Path saved to {filename}")
    
    def load_path(self, filename: str) -> List[np.ndarray]:
        """Load path from file"""
        return list(np.load(filename))