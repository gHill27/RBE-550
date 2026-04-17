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
        # SE(3) state space = position (3) + rotation (quaternion)
        self.space = ob.SE3StateSpace()
        # Set position bounds
         # Translation bounds only — SO3 is automatically bounded to unit quaternion sphere
        pos_bounds = ob.RealVectorBounds(3)
        for i, (low, high) in enumerate(self.bounds):
            pos_bounds.setLow(i, low)
            pos_bounds.setHigh(i, high)
        self.space.setBounds(pos_bounds)

        self.si = ob.SpaceInformation(self.space)
    
    def add_obstacle(self, scad_file: str, name: str, 
                     position: Tuple[float, float, float] = (0, 0, 0),
                     parameters: dict = None):
        """Add obstacle from OpenSCAD file in models folder"""
        self.checker.add_from_scad(scad_file, name, position, parameters)
    
    def set_robot(self, scad_file: str = None, radius: float = None,
                  start_position: Tuple[float, float, float] = (0, 0, 0),
                  start_orientation: Tuple[float,float,float,float] = (1,0,0,0)):
        if scad_file:
            generator = MeshGenerator(models_folder=self.models_folder)
            self.robot_mesh = generator.from_scad(scad_file)

            # self.robot_mesh.apply_translation(-self.robot_mesh.center_mass)
        elif radius:
            self.robot_mesh = trimesh.primitives.Sphere(radius=radius)
        else:
            raise ValueError("Must provide either scad_file or radius")
        
        self.robot_start = start_position
        self.robot_start_orientation = start_orientation
        
        self.si.setStateValidityCheckingResolution(0.001)
        self.robot_mesh = self.robot_mesh.copy()
        self.robot_mesh.apply_translation([self.robot_mesh.extents[2]/2, 0, 0])
        
        import weakref
        class ValidityChecker(ob.StateValidityChecker):
            def __init__(self, si, planner):
                super().__init__(si)
                self.planner_ref = weakref.ref(planner)   # breaks the cycle

            def isValid(self, state):
                planner = self.planner_ref()
                if planner is None:
                    return False
                return planner._is_state_valid(state)
        
        self.validity_checker = ValidityChecker(self.si, self)
        self.si.setStateValidityChecker(self.validity_checker)
        self.si.setup()
    

    def _is_state_valid(self, state) -> bool:
        x = state.getX()
        y = state.getY()
        z = state.getZ()
        # Extract rotation quaternion (w, x, y, z)
        rot = state.rotation()
        qw, qx, qy, qz = rot.w, rot.x, rot.y, rot.z
        # Build 4x4 transform matrix from quaternion + translation
        transform = trimesh.transformations.quaternion_matrix([qw, qx, qy, qz])
        transform[:3, 3] = [x, y, z]
        return not self.checker.check_mesh_against_manager(self.robot_mesh, transform)
        
        
    
    def plan_path(self, 
                  start: Tuple[float, float, float],
                  goal: Tuple[float, float, float],
                  start_orientation: Tuple[float, float, float, float] = (1, 0, 0, 0),
                  goal_orientation: Tuple[float, float, float, float] = (1, 0, 0, 0),
                  planner_type: str = 'rrt_star',
                  max_time: float = 5.0,
                  goal_tolerance: float = 0.5) -> Optional[List[np.ndarray]]:
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
        # Start state
        start_state = self.space.allocState()
        start_state.setX(start[0])
        start_state.setY(start[1])
        start_state.setZ(start[2])
        rot_start = start_state.rotation()
        rot_start.w = 1.0
        rot_start.x = 0.0
        rot_start.y = 0.0
        rot_start.z = 0.0
        
        # --- Goal state (identity quaternion) ---
        goal_state = self.space.allocState()
        goal_state.setX(goal[0])
        goal_state.setY(goal[1])
        goal_state.setZ(goal[2])
        rot_goal = goal_state.rotation()
        rot_goal.w = 1.0
        rot_goal.x = 0.0
        rot_goal.y = 0.0
        rot_goal.z = 0.0
        
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
        planner.setRange(0.2) #0.2mm steps
        
        # Plan
        print(f"\n🚀 Planning with {planner_type.upper()}...")
        print(f"   Start: {start}")
        print(f"   Goal: {goal}")
        print(f"   Time limit: {max_time}s")
        
        # Solve
        solved = planner.solve(max_time)
        self.planner_data = ob.PlannerData(self.si)
        planner.getPlannerData(self.planner_data)
        self.tree_edges = []
        self.tree_nodes = []  # ADD THIS

        num_vertices = self.planner_data.numVertices()

        for i in range(num_vertices):
            v1 = self.planner_data.getVertex(i)
            try:
                s1 = v1.getState()
                if s1 is None:
                    continue
            except Exception:
                continue
            
            # Collect ALL sampled nodes (including dead-end explorations)
            self.tree_nodes.append((s1.getX(), s1.getY(), s1.getZ()))  # ADD THIS
            
            neighbors = self.planner_data.getEdges(i)
            for ni in neighbors:
                v2 = self.planner_data.getVertex(ni)
                if v2 is None: continue
                s2 = v2.getState()
                self.tree_edges.append([
                    (s1.getX(), s1.getY(), s1.getZ()),
                    (s2.getX(), s2.getY(), s2.getZ())
                ])



        
        for i in range(num_vertices):
            v1 = self.planner_data.getVertex(i)
            s1 = v1.getState()
            
            # FIX: Change this line from getEdges(i, neighbors)
            # to returning the list directly:
            neighbors = self.planner_data.getEdges(i)
            
            for ni in neighbors:
                v2 = self.planner_data.getVertex(ni)
                if v2 is None: continue
                s2 = v2.getState()
                
                # Store coordinates as simple tuples
                self.tree_edges.append([
                    (s1.getX(), s1.getY(), s1.getZ()),
                    (s2.getX(), s2.getY(), s2.getZ())
                ])
        if solved:
            print("✓ Path found!")
            
            # Get the path
            path = pdef.getSolutionPath()
            
            
            # Extract waypoints
            waypoints = []
            for i in range(path.getStateCount()):
                st = path.getState(i)
                waypoints.append(np.array([st.getX(), st.getY(), st.getZ()]))

            status = solved 
            if status == ob.PlannerStatus.EXACT_SOLUTION:
                print("✓ Exact path found! Final state is within goal tolerance.")
                # Process the exact path as you currently do...
                path = pdef.getSolutionPath()
                # ... rest of your waypoint extraction ...
                return waypoints

            elif status == ob.PlannerStatus.APPROXIMATE_SOLUTION:
                print("⚠️ Approximate path found. The robot may not have reached the exact goal.")
                # You can still get the path, but be aware it's only an approximation
                path = pdef.getSolutionPath()
                # ... rest of your waypoint extraction ...
                print(f"   The final waypoint is at: {waypoints[-1]}, while the goal is at: {goal}")
                return waypoints  # or return None if you only want exact solutions

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
        """
        Visualize obstacles, the path line, and coloured spheres at start and goal.
        
        Parameters:
            waypoints: list of (x,y,z) arrays
        """
        import trimesh
        import numpy as np
        from trimesh.path.entities import Line
        from trimesh.path import Path3D

        scene = trimesh.Scene()
        self.cleanup()
        # --- Obstacles (static) ---
        for name in self.checker.names:
            mesh = self.checker.added_meshes[name].copy()
            transform = self.checker.current_poses[name]
            if "Case" in name:
                mesh.visual.face_colors = (200, 200, 200, 80)   # transparent grey
            elif "Counter" in name or "secondary" in name:
                mesh.visual.face_colors = (50, 50, 255, 200)    # solid blue
            else:
                mesh.visual.face_colors = (50, 50, 50, 50)
            scene.add_geometry(mesh, node_name=name, transform=transform)

        # --- Path line (orange) ---
        path_points = np.array(waypoints)
        line_entity = Line(np.arange(len(path_points)))
        path_line = Path3D(entities=[line_entity], vertices=path_points, colors=[[255, 165, 0, 255]])
        scene.add_geometry(path_line)

        # --- Start sphere (blue) ---
        start_sphere = trimesh.primitives.Sphere(radius=10.0, center=waypoints[0])
        start_sphere.visual.face_colors = (0, 0, 255, 200)   # blue, semi‑transparent
        scene.add_geometry(start_sphere, node_name="start_sphere")

        # --- Goal sphere (green) ---
        goal_sphere = trimesh.primitives.Sphere(radius=10.0, center=waypoints[-1])
        goal_sphere.visual.face_colors = (0, 255, 0, 200)    # green, semi‑transparent
        scene.add_geometry(goal_sphere, node_name="goal_sphere")

        print("--- Rendering Path with Start (blue) and Goal (green) Spheres ---")
        scene.show()

    def visualize_polyscope(self, waypoints=None):
        """Interactive 3D visualization using Polyscope (High Stability)."""
        import polyscope as ps
        import numpy as np
        
        self.cleanup()
        ps.init()
        ps.set_up_dir("z_up")

        # 1. ADD OBSTACLES
        if hasattr(self.checker, 'added_meshes'):
            for name in self.checker.names:
                mesh = self.checker.added_meshes[name]
                transform = self.checker.current_poses[name]
                
                v_hom = np.hstack([mesh.vertices.copy(), np.ones((len(mesh.vertices), 1))])
                v_transformed = (v_hom @ transform.T)[:, :3]
                
                ps_mesh = ps.register_surface_mesh(name, v_transformed, mesh.faces)
                if "case" in name.lower():
                    ps_mesh.set_color((0.8, 0.8, 0.8))
                    ps_mesh.set_transparency(0.3)
                elif "counter" in name.lower() or "secondary" in name.lower():
                    ps_mesh.set_color((0.2, 0.5, 0.8))
                else:
                    ps_mesh.set_color((0.4, 0.4, 0.4))

        # 2. ADD ALL SAMPLED NODES + TREE EDGES
        if hasattr(self, 'tree_nodes') and self.tree_nodes:
            all_nodes = np.array(self.tree_nodes)  # (N, 3) — every sampled state

            # Build path node set for coloring
            path_node_set = set()
            path_edge_set = set()
            if waypoints is not None and len(waypoints) > 1:
                wp = np.array(waypoints)
                for pt in wp:
                    path_node_set.add(tuple(np.round(pt, 4)))
                for i in range(len(wp) - 1):
                    p1 = tuple(np.round(wp[i], 4))
                    p2 = tuple(np.round(wp[i + 1], 4))
                    path_edge_set.add((p1, p2))
                    path_edge_set.add((p2, p1))

            # Split nodes into path vs exploration
            node_index = {tuple(np.round(n, 4)): i for i, n in enumerate(self.tree_nodes)}

            exploration_mask = np.array([
                tuple(np.round(n, 4)) not in path_node_set
                for n in self.tree_nodes
            ])
            exploration_nodes = all_nodes[exploration_mask]
            path_nodes_arr   = all_nodes[~exploration_mask]

            # Show exploration samples as a point cloud (cyan dots)
            if len(exploration_nodes) > 0:
                ps_samples = ps.register_point_cloud("sampled_nodes", exploration_nodes)
                ps_samples.set_color((0.0, 0.8, 0.8))  # cyan
                ps_samples.set_radius(0.0015)

            # Show path nodes distinctly (yellow dots)
            if len(path_nodes_arr) > 0:
                ps_path_nodes = ps.register_point_cloud("path_nodes", path_nodes_arr)
                ps_path_nodes.set_color((1.0, 1.0, 0.0))  # yellow
                ps_path_nodes.set_radius(0.003)

            # Draw tree edges (exploration branches only, skip path edges)
            if hasattr(self, 'tree_edges') and self.tree_edges:
                tree_only_edges = []
                for edge in self.tree_edges:
                    p1 = tuple(np.round(edge[0], 4))
                    p2 = tuple(np.round(edge[1], 4))
                    if (p1, p2) not in path_edge_set:
                        i1 = node_index.get(p1)
                        i2 = node_index.get(p2)
                        if i1 is not None and i2 is not None:
                            tree_only_edges.append([i1, i2])

                if tree_only_edges:
                    ps_tree = ps.register_curve_network(
                        "rrt_tree", all_nodes, np.array(tree_only_edges)
                    )
                    ps_tree.set_color((0.0, 0.5, 0.5))  # darker cyan than nodes
                    ps_tree.set_radius(0.0006)

        # 3. ADD THE SOLUTION PATH
        if waypoints is not None and len(waypoints) > 1:
            wp = np.array(waypoints)
            path_edges = np.array([[i, i + 1] for i in range(len(wp) - 1)])
            
            ps_path = ps.register_curve_network("planned_path", wp, path_edges)
            ps_path.set_color((1.0, 0.0, 0.0))   # red
            ps_path.set_radius(0.004)

            ps_start = ps.register_point_cloud("start", wp[0:1])
            ps_start.set_color((0.0, 1.0, 0.0))  # green
            ps_start.set_radius(0.012)

            ps_goal = ps.register_point_cloud("goal", wp[-1:])
            ps_goal.set_color((1.0, 0.6, 0.0))   # orange
            ps_goal.set_radius(0.012)

        print("🚀 Opening Polyscope Viewer...")
        ps.show()

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

    def cleanup(self):
        """Explicitly release OMPL C++ objects to prevent nanobind leaks."""
        # Destroy in reverse dependency order
        if hasattr(self, 'validity_checker'):
            del self.validity_checker
        if hasattr(self, 'planner_data'):
            del self.planner_data
        if hasattr(self, 'si'):
            del self.si
        if hasattr(self, 'space'):
            del self.space
        import gc
        gc.collect()
