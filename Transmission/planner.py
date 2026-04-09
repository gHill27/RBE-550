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
    
    def __init__(self, bounds: List[Tuple[float, float]]):
        """
        Initialize 3D planner.
        
        Args:
            bounds: [(x_min, x_max), (y_min, y_max), (z_min, z_max)]
        """
        if not OMPL_AVAILABLE:
            raise ImportError("OMPL not installed. Run: pip install ompl")
        
        self.bounds = bounds
        self.checker = CollisionChecker3D()
        self.robot_mesh = None
        self.robot_radius = None
        
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
        """Add obstacle from OpenSCAD file"""
        self.checker.add_from_scad(scad_file, name, position, parameters)
        print(f"  Obstacle: {name} at {position}")
    
    def set_robot(self, scad_file: str = None, radius: float = None,
                  start_position: Tuple[float, float, float] = (0, 0, 0)):
        """
        Set robot as either mesh or sphere.
        
        Args:
            scad_file: OpenSCAD file for robot mesh
            radius: Use sphere robot with given radius
            start_position: Starting position
        """
        if scad_file:
            generator = MeshGenerator()
            self.robot_mesh = generator.from_scad(scad_file)
            self.robot_radius = None
            print(f"✓ Robot loaded from {scad_file}")
        elif radius:
            self.robot_mesh = trimesh.primitives.Sphere(radius=radius)
            self.robot_radius = radius
            print(f"✓ Robot set as sphere (radius={radius})")
        else:
            raise ValueError("Must provide either scad_file or radius")
        
        self.robot_start = start_position
        
        # Setup collision checking
        self.si.setStateValidityChecker(ob.StateValidityCheckerFn(
            lambda state: self._is_state_valid(state)
        ))
    
    def _is_state_valid(self, state) -> bool:
        """Check if robot at given state collides with any obstacle"""
        # Get position from state
        x = state[0]
        y = state[1]
        z = state[2]
        
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
        # Create start and goal states
        start_state = ob.State(self.space)
        start_state[0] = start[0]
        start_state[1] = start[1]
        start_state[2] = start[2]
        
        goal_state = ob.State(self.space)
        goal_state[0] = goal[0]
        goal_state[1] = goal[1]
        goal_state[2] = goal[2]
        
        # Setup problem
        pdef = ob.ProblemDefinition(self.si)
        pdef.setStartAndGoalStates(start_state, goal_state, goal_tolerance)
        
        # Select planner
        if planner_type == 'rrt':
            planner = og.RRT(self.si)
        elif planner_type == 'rrt_connect':
            planner = og.RRTConnect(self.si)
        elif planner_type == 'rrt_star':
            planner = og.RRTstar(self.si)
        else:
            raise ValueError(f"Unknown planner: {planner_type}")
        
        planner.setProblemDefinition(pdef)
        
        # Plan
        print(f"\n🚀 Planning with {planner_type.upper()}...")
        print(f"   Start: {start}")
        print(f"   Goal: {goal}")
        print(f"   Time limit: {max_time}s")
        
        solved = planner.solve(max_time)
        
        if solved:
            print("✓ Path found!")
            
            # Get path
            path = pdef.getSolutionPath()
            
            # Simplify
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
            # Create path line
            points = np.array(waypoints)
            for i in range(len(points) - 1):
                line = trimesh.creation.cylinder(
                    radius=0.05,
                    segment=[points[i], points[i+1]]
                )
                line.visual.face_colors = [255, 0, 0, 255]
                scene.add_geometry(line)
            
            # Add start marker
            start_sphere = trimesh.primitives.Sphere(radius=0.1, center=points[0])
            start_sphere.visual.face_colors = [0, 255, 0, 255]
            scene.add_geometry(start_sphere)
            
            # Add goal marker
            goal_sphere = trimesh.primitives.Sphere(radius=0.1, center=points[-1])
            goal_sphere.visual.face_colors = [0, 0, 255, 255]
            scene.add_geometry(goal_sphere)
        
        scene.show()
    
    def save_path(self, waypoints: List[np.ndarray], filename: str = 'path.npy'):
        """Save path to file"""
        np.save(filename, np.array(waypoints))
        print(f"✓ Path saved to {filename}")
    
    def load_path(self, filename: str) -> List[np.ndarray]:
        """Load path from file"""
        return list(np.load(filename))


# Example usage
def example_simple_3d():
    """Simple 3D example with cubes and spheres"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Simple 3D Navigation")
    print("="*60)
    
    # Create test OpenSCAD files
    with open('wall.scad', 'w') as f:
        f.write("cube([4, 0.5, 3], center=true);")
    
    with open('pillar.scad', 'w') as f:
        f.write("cylinder(r=0.8, h=4, center=true, $fn=32);")
    
    # Setup planner
    bounds = [(-5, 5), (-5, 5), (-2, 2)]
    planner = RRTPlanner3D(bounds)
    
    # Add obstacles
    planner.add_obstacle('wall.scad', 'Wall', position=(0, 2, 0))
    planner.add_obstacle('pillar.scad', 'Pillar1', position=(1, -1, 0))
    planner.add_obstacle('pillar.scad', 'Pillar2', position=(-1, -1, 0))
    
    # Set robot (sphere)
    planner.set_robot(radius=0.4, start_position=(-4, -3, 0))
    
    # Plan path
    waypoints = planner.plan_path(
        start=(-4, -3, 0),
        goal=(4, 3, 0),
        planner_type='rrt_connect',
        max_time=3.0
    )
    
    if waypoints:
        planner.visualize_path(waypoints)
        planner.save_path(waypoints)
    
    # Cleanup
    import os
    for f in ['wall.scad', 'pillar.scad']:
        if os.path.exists(f):
            os.remove(f)


def example_maze_3d():
    """3D maze navigation example"""
    print("\n" + "="*60)
    print("EXAMPLE 2: 3D Maze Navigation")
    print("="*60)
    
    # Create maze walls
    wall_scad = """
    module wall() {
        cube([0.3, 3, 2], center=true);
    }
    wall();
    """
    
    with open('maze_wall.scad', 'w') as f:
        f.write(wall_scad)
    
    # Setup planner
    bounds = [(-5, 5), (-5, 5), (-1, 1)]
    planner = RRTPlanner3D(bounds)
    
    # Create maze obstacles (vertical walls)
    wall_positions = [
        (-2, 0, 0), (0, 2, 0), (0, -2, 0), (2, 0, 0),
        (-1, 1, 0), (1, -1, 0)
    ]
    
    for i, pos in enumerate(wall_positions):
        planner.add_obstacle('maze_wall.scad', f'Wall{i}', position=pos)
    
    # Set robot
    planner.set_robot(radius=0.2, start_position=(-4, -4, 0))
    
    # Plan through maze
    waypoints = planner.plan_path(
        start=(-4, -4, 0),
        goal=(4, 4, 0),
        planner_type='rrt_star',
        max_time=5.0
    )
    
    if waypoints:
        print(f"\n✓ Maze path found!")
        print(f"  Path length: {sum(np.linalg.norm(waypoints[i] - waypoints[i-1]) for i in range(1, len(waypoints))):.2f}")
    
    # Cleanup
    import os
    if os.path.exists('maze_wall.scad'):
        os.remove('maze_wall.scad')


def example_complex_obstacles():
    """Example with complex 3D obstacles from OpenSCAD"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Complex 3D Obstacles")
    print("="*60)
    
    # Create complex obstacle (a torus/ring)
    complex_obstacle = """
    $fn=48;
    difference() {
        cylinder(r=2, h=1, center=true);
        cylinder(r=1.2, h=1.1, center=true);
    }
    """
    
    with open('ring.scad', 'w') as f:
        f.write(complex_obstacle)
    
    # Create simple obstacle
    with open('block.scad', 'w') as f:
        f.write("cube([2, 2, 2], center=true);")
    
    # Setup planner
    bounds = [(-6, 6), (-6, 6), (-2, 2)]
    planner = RRTPlanner3D(bounds)
    
    # Add obstacles
    planner.add_obstacle('ring.scad', 'Ring', position=(0, 0, 0))
    planner.add_obstacle('block.scad', 'Block1', position=(3, 2, 0))
    planner.add_obstacle('block.scad', 'Block2', position=(-3, -2, 0))
    planner.add_obstacle('block.scad', 'Block3', position=(2, -3, 0))
    
    # Set robot as sphere
    planner.set_robot(radius=0.5, start_position=(-5, -5, 0))
    
    # Plan path
    waypoints = planner.plan_path(
        start=(-5, -5, 0),
        goal=(5, 5, 0),
        planner_type='rrt_connect',
        max_time=5.0
    )
    
    if waypoints:
        print(f"\n✓ Path through complex obstacles found!")
        print(f"  Waypoints: {len(waypoints)}")
    
    # Cleanup
    import os
    for f in ['ring.scad', 'block.scad']:
        if os.path.exists(f):
            os.remove(f)


if __name__ == "__main__":
    if not OMPL_AVAILABLE:
        print("\n❌ OMPL not installed!")
        print("Install with: pip install ompl")
        exit(1)
    
    # Run examples
    example_simple_3d()
    example_maze_3d()
    example_complex_obstacles()
    
    print("\n" + "="*60)
    print("✓ All 3D examples complete!")
    print("="*60)