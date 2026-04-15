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
        elif radius:
            self.robot_mesh = trimesh.primitives.Sphere(radius=radius)
        else:
            raise ValueError("Must provide either scad_file or radius")
        
        self.robot_start = start_position
        
        # --- NEW: Register the robot with the collision manager ---
        # We give it a fixed name "robot" so we can move it later
        self.checker.add_mesh(self.robot_mesh, name="robot", position=start_position)
        
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
        # 1. Extract and sanitize state (the fix we discussed)
        pos = (float(state[0]), float(state[1]), float(state[2]))
        
        # 2. Update the robot position in the manager
        self.checker.update_position("robot", pos)
        
        # 3. Use the manager's triangle-accurate check
        # This replaces the manual 'for' loop and 'bounds' checks
        return not self.checker.manager.in_collision_internal()
        
    
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
        """Visualize obstacles, robot at each waypoint, and the path line."""
        import trimesh
        import numpy as np

        scene = trimesh.Scene()

        # --- Obstacles -------------------------------------------------------
        for i, mesh in enumerate(self.checker.meshes):
            m = mesh.copy()
            # Apply the stored world-space translation
            pos = self.checker.positions[i]
            if any(p != 0 for p in pos):
                m.apply_translation(pos)
            m.visual.face_colors = [100, 100, 220, 160]
            scene.add_geometry(m, node_name=f"obstacle_{i}")

        # --- Robot at each waypoint (ghost trail) ----------------------------
        if self.robot_mesh is not None and len(waypoints) > 0:
            pts = np.array(waypoints)
            n   = len(pts)

            for idx, wp in enumerate(waypoints):
                ghost = self.robot_mesh.copy()
                ghost.apply_translation(wp)

                # Fade from translucent at start to solid at end
                alpha = int(40 + 200 * idx / max(n - 1, 1))
                ghost.visual.face_colors = [220, 80, 80, alpha]
                scene.add_geometry(ghost, node_name=f"robot_{idx}")

            # Solid robot at goal
            final = self.robot_mesh.copy()
            final.apply_translation(waypoints[-1])
            final.visual.face_colors = [220, 50, 50, 255]
            scene.add_geometry(final, node_name="robot_final")

            # Solid robot at start
            start_mesh = self.robot_mesh.copy()
            start_mesh.apply_translation(waypoints[0])
            start_mesh.visual.face_colors = [50, 200, 50, 255]
            scene.add_geometry(start_mesh, node_name="robot_start")

        # --- Path line -------------------------------------------------------
        if len(waypoints) > 1:
            pts = np.array(waypoints)
            for i in range(len(pts) - 1):
                seg_start = pts[i]
                seg_end   = pts[i + 1]
                direction  = seg_end - seg_start
                length     = np.linalg.norm(direction)
                if length < 1e-6:
                    continue

                cyl = trimesh.creation.cylinder(radius=1.5, height=length)

                # Orient cylinder along the segment
                z     = np.array([0, 0, 1])
                vec   = direction / length
                cross = np.cross(z, vec)
                cross_norm = np.linalg.norm(cross)
                if cross_norm > 1e-6:
                    axis  = cross / cross_norm
                    angle = np.arccos(np.clip(np.dot(z, vec), -1, 1))
                    R     = trimesh.transformations.rotation_matrix(angle, axis)
                else:
                    R = np.eye(4)

                T = trimesh.transformations.translation_matrix(
                    (seg_start + seg_end) / 2
                )
                cyl.apply_transform(T @ R)
                cyl.visual.face_colors = [255, 200, 0, 255]
                scene.add_geometry(cyl, node_name=f"path_seg_{i}")

            # Start marker (green sphere)
            s = trimesh.creation.icosphere(radius=4)
            s.apply_translation(pts[0])
            s.visual.face_colors = [0, 255, 0, 255]
            scene.add_geometry(s, node_name="marker_start")

            # Goal marker (red sphere)
            g = trimesh.creation.icosphere(radius=4)
            g.apply_translation(pts[-1])
            g.visual.face_colors = [255, 0, 0, 255]
            scene.add_geometry(g, node_name="marker_goal")

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
