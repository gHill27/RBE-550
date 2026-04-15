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
        self.models_folder = models_folder
        self.checker = CollisionChecker3D(models_folder=models_folder)
        self.robot_mesh = None
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
        # print(f"  Obstacle: {name} at {position}")
    
    def set_robot(self, scad_file: str = None, radius: float = None,
                  start_position: Tuple[float, float, float] = (0, 0, 0)):
        if scad_file:
            generator = MeshGenerator(models_folder=self.models_folder)
            self.robot_mesh = generator.from_scad(scad_file)

            # self.robot_mesh.apply_translation(-self.robot_mesh.center_mass)
        elif radius:
            self.robot_mesh = trimesh.primitives.Sphere(radius=radius)
        else:
            raise ValueError("Must provide either scad_file or radius")
        
        self.robot_start = start_position
        
        # --- NEW: Register the robot with the collision manager ---
        # We give it a fixed name "robot" so we can move it later
        self.checker.add_mesh(self.robot_mesh, name="robot", position=start_position)
        self.si.setStateValidityCheckingResolution(0.01)
        
        class ValidityChecker(ob.StateValidityChecker):
            def __init__(self, si, planner):
                super().__init__(si)
                self.planner = planner
            def isValid(self, state):
                return self.planner._is_state_valid(state)
        
        self.validity_checker = ValidityChecker(self.si, self)
        self.si.setStateValidityChecker(self.validity_checker)
        self.si.setup()
    
    def _is_state_valid(self, state) -> bool:
        pos = (float(state[0]), float(state[1]), float(state[2]))
        self.checker.update_position("robot", pos)
        
        robot_mesh = self.checker.added_meshes["robot"]
        robot_transform = self.checker.current_poses["robot"]
        
        for name, mesh in self.checker.added_meshes.items():
            if name == "robot":
                continue
            transform = self.checker.current_poses[name]
            temp = trimesh.collision.CollisionManager()
            temp.add_object(name, mesh, transform)
            if temp.in_collision_single(robot_mesh, robot_transform):
                return False
        return True
        
    
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
        # Setup problem definition
        pdef = ob.ProblemDefinition(self.si)
        
        # Set start state using allocState
        start_state = self.space.allocState()
        start_state[0] = start[0]
        start_state[1] = start[1]
        start_state[2] = start[2]
        
        # Set goal state using allocState
        goal_state = self.space.allocState()
        goal_state[0] = goal[0]
        goal_state[1] = goal[1]
        goal_state[2] = goal[2]
        
        pdef.setStartAndGoalStates(start_state, goal_state, goal_tolerance)
        
        # Select planner
        if planner_type == 'rrt':
            planner = og.RRT(self.si)
        elif planner_type == 'rrt_connect':
            planner = og.RRTConnect(self.si)
        elif planner_type == 'rrt_star':
            planner = og.RRTstar(self.si)
        else:
            print(f"Unknown planner type '{planner_type}', using RRTConnect")
            planner = og.RRTConnect(self.si)
        
        planner.setProblemDefinition(pdef)
        
        # Plan
        print(f"\n🚀 Planning with {planner_type.upper()}...")
        print(f"   Start: {start}")
        print(f"   Goal: {goal}")
        print(f"   Time limit: {max_time}s")
        
        # Solve
        solved = planner.solve(max_time)
        
        if solved:
            print("✓ Path found!")
            
            # Get the path
            path = pdef.getSolutionPath()
            
            
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
            print("\nPossible issues:")
            print("  - Start or goal position may be in collision")
            print("  - Path may be blocked by obstacles")
            print("  - Try increasing max_time")
            print("  - Try planner_type='rrt_star' for better exploration")
            
            
            return None
    
    # Drop this method into RRTPlanner3D in ompl_planner.py,
# replacing the existing visualize_path method entirely.

    def visualize_path(self, waypoints):
        """Visualize obstacles with transparency and the path result."""
        import trimesh
        import numpy as np

        scene = trimesh.Scene()

        # --- Obstacles (The Case and Countershaft) ---------------------------
        for name in self.checker.names:
            if name == "robot":
                continue
                
            mesh = self.checker.added_meshes[name].copy()
            transform = self.checker.current_poses[name]
            
            # Apply Transparency/Colors based on the part name
            if "Case" in name:
                # Transparent Silver: R:200, G:200, B:200, Alpha:60
                mesh.visual.face_colors = (200,200,200,60)
            elif "Counter" in name or "secondary" in name:
                # Solid Blue: R:50, G:50, B:255, Alpha:255
                mesh.visual.face_colors = (50,50,255,255)
            else:
                mesh.visual.face_colors = (50,50,50,50)

            scene.add_geometry(mesh, node_name=name, transform=transform)

        # --- Path Visualization (The Moving Robot) ---------------------------
        if self.robot_mesh is not None and len(waypoints) > 0:
            # Show a Green "Ghost" at the start
            start_mesh = self.robot_mesh.copy()
            start_transform = np.eye(4)
            start_transform[:3, 3] = waypoints[0]
            start_mesh.visual.face_colors = (0,255,0,100)# Semi-transparent green
            scene.add_geometry(start_mesh, node_name="path_start", transform=start_transform)

            # Show a Red "Ghost" at the goal
            goal_mesh = self.robot_mesh.copy()
            goal_transform = np.eye(4)
            goal_transform[:3, 3] = waypoints[-1]
            goal_mesh.visual.face_colors = (255,0,0,255)# Solid red
            scene.add_geometry(goal_mesh, node_name="path_goal", transform=goal_transform)

            # Draw the path line (Orange)
            path_points = np.array(waypoints)
            path_line = trimesh.load_path(path_points)
            scene.add_geometry(path_line)

        print("--- Rendering 3D Path Result ---")
        scene.show()
    
    def save_path(self, waypoints: List[np.ndarray], filename: str = 'planned_path.npy'):
        """Save path to file"""
        path_array = np.array(waypoints)
        np.save(filename, path_array)
        print(f"✓ Path saved to {filename}")
        return filename
    
    def load_path(self, filename: str) -> List[np.ndarray]:
        """Load path from file"""
        path_array = np.load(filename)
        return list(path_array)
