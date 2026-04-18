#!/usr/bin/env python3
"""
ompl_planner.py - 3D RRT path planning with OpenSCAD obstacles
"""

import numpy as np
import trimesh
from typing import List, Tuple, Optional
from pathlib import Path
import weakref

try:
    from ompl import base as ob
    from ompl import geometric as og
    OMPL_AVAILABLE = True
except ImportError:
    OMPL_AVAILABLE = False
    print("Warning: OMPL not installed. Run: pip install ompl")

from mesh_gen import MeshGenerator
from collision import CollisionChecker3D

class ShaftPositionGoal(ob.GoalSampleableRegion):
    """Goal region comparing only shaft center XYZ, ignoring orientation.
    
    OMPL's default SE3 distance metric combines translational AND rotational
    components, which can make two states appear far apart even when their
    positions are close. This class strips rotation out entirely so the
    planner only cares about whether the shaft center is within tolerance
    of the goal position.
    """
    def __init__(self, si, goal_pos, tolerance):
        super().__init__(si)
        self.goal_pos = np.array(goal_pos[:3], dtype=float)
        self.setThreshold(tolerance)

    def distanceGoal(self, state) -> float:
        """Pure Euclidean XYZ distance between shaft centers."""
        pos = np.array([state.getX(), state.getY(), state.getZ()])
        return float(np.linalg.norm(pos - self.goal_pos))

    def sampleGoal(self, state):
        """Return a valid goal sample at the goal position."""
        state.setX(float(self.goal_pos[0]))
        state.setY(float(self.goal_pos[1]))
        state.setZ(float(self.goal_pos[2]))
        rot = state.rotation()
        rot.w = 1.0
        rot.x = 0.0
        rot.y = 0.0
        rot.z = 0.0

    def maxSampleCount(self) -> int:
        return 100


class ShaftValidityChecker(ob.StateValidityChecker):
    """Validity checker using a weakref to avoid circular reference
    between RRTPlanner3D and the OMPL SpaceInformation object.
    
    Defined at module level so the class definition is never freed
    while an instance is alive inside OMPL's C++ side.
    """
    def __init__(self, si, planner):
        super().__init__(si)
        self.planner_ref = weakref.ref(planner)

    def isValid(self, state) -> bool:
        planner = self.planner_ref()
        if planner is None:
            return False
        return planner._is_state_valid(state)

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
              start_orientation: Tuple[float, float, float, float] = (1, 0, 0, 0)):
        if scad_file:
            generator = MeshGenerator(models_folder=self.models_folder)
            self.robot_mesh = generator.from_scad(scad_file)
        elif radius:
            self.robot_mesh = trimesh.primitives.Sphere(radius=radius)
        else:
            raise ValueError("Must provide either scad_file or radius")

        self.robot_start = start_position
        self.robot_start_orientation = start_orientation

        # Center the mesh along its longest axis before registering
        self.robot_mesh = self.robot_mesh.copy()
        midpoint = (self.robot_mesh.bounds[0] + self.robot_mesh.bounds[1]) / 2
        self.robot_mesh.apply_translation(-midpoint)


        center = self.robot_mesh.center_mass
        print(f"Robot mesh center after centering: {center}")
        print(f"Robot mesh bounds: {self.robot_mesh.bounds}")
        self.si.setStateValidityCheckingResolution(0.01)

        self.validity_checker = ShaftValidityChecker(self.si, self)
        self.si.setStateValidityChecker(self.validity_checker)
        self.si.setup()
    

    def _is_state_valid(self, state) -> bool:
        x = state.getX()
        y = state.getY()
        z = state.getZ()
        
        rot = state.rotation()
        # Normalize quaternion defensively — OMPL can return
        # slightly denormalized quaternions during interpolation
        q = np.array([rot.w, rot.x, rot.y, rot.z])
        norm = np.linalg.norm(q)
        if norm < 1e-6:
            return False  # degenerate state, reject
        q /= norm
        qw, qx, qy, qz = q

        transform = self.checker.quaternion_to_matrix(qw, qx, qy, qz)
        transform[:3, 3] = [x, y, z]
        
        return not self.checker.check_mesh_against_manager(self.robot_mesh, transform)
        
    
    def make_se3_state(self, pos, quat):
        """Helper: build an SE3 state from position + (qw,qx,qy,qz) quaternion."""
        state = self.space.allocState()
        state.setX(pos[0])
        state.setY(pos[1])
        state.setZ(pos[2])
        # Normalize defensively before setting
        q = np.array(quat, dtype=float)
        q /= np.linalg.norm(q)
        rot = state.rotation()
        rot.w = q[0]
        rot.x = q[1]
        rot.y = q[2]
        rot.z = q[3]
        return state
    
    def plan_path(self,
              start: Tuple[float, float, float],
              goal: Tuple[float, float, float],
              start_orientation: Tuple[float, float, float, float] = (1, 0, 0, 0),
              goal_orientation: Tuple[float, float, float, float] = (1, 0, 0, 0),
              planner_type: str = 'rrt_connect',
              max_time: float = 5.0,
              goal_tolerance: float = 0.5) -> Optional[List[np.ndarray]]:

        pdef = ob.ProblemDefinition(self.si)


        start_state = self.make_se3_state(start, start_orientation)
        # goal_state  = self.make_se3_state(goal,  goal_orientation)
        self.goal_region = ShaftPositionGoal(self.si, goal, goal_tolerance)

        pdef.addStartState(start_state)
        pdef.setGoal(self.goal_region)
        # Diagnostic: confirm SE3 vs XYZ distances before wasting planning time
        goal_state = self.space.allocState()
        goal_state.setX(float(goal[0]))
        goal_state.setY(float(goal[1]))
        goal_state.setZ(float(goal[2]))
        rot_g = goal_state.rotation()
        rot_g.w, rot_g.x, rot_g.y, rot_g.z = 1.0, 0.0, 0.0, 0.0
        # pdef.setStartAndGoalStates(start_state, goal_state, goal_tolerance)

        se3_dist = self.si.distance(start_state, goal_state)
        xyz_dist = np.linalg.norm(np.array(goal[:3]) - np.array(start[:3]))
        print(f"   SE3 distance start→goal: {se3_dist:.1f}")
        print(f"   XYZ distance start→goal: {xyz_dist:.1f}mm")
        print(f"   Goal tolerance: {goal_tolerance}mm (position only)")


        # Planner selection
        if planner_type == 'rrt':
            planner = og.RRT(self.si)
            planner.setGoalBias(0.2)
        elif planner_type == 'rrt_connect':
            planner = og.RRTConnect(self.si)
        elif planner_type == 'rrt_star':
            planner = og.RRTstar(self.si)
            planner.setGoalBias(0.2)
        else:
            print(f"Unknown planner type '{planner_type}', using RRTConnect")
            planner = og.RRTConnect(self.si)

        planner.setProblemDefinition(pdef)
        planner.setRange(2.0)  # mm — proportional to your 800mm space

        print(f"\n🚀 Planning with {planner_type.upper()}...")
        print(f"   Start: pos={start} quat={start_orientation}")
        print(f"   Goal:  pos={goal}  quat={goal_orientation}")
        print(f"   Time limit: {max_time}s")
        # After setting up start/goal states, before planner.solve():
        print(f"SE3 distance start→goal: {self.si.distance(start_state, goal_state):.3f}")
        print(f"Pure XYZ distance: {np.linalg.norm(np.array(goal[:3]) - np.array(start[:3])):.3f}mm")
        solved = planner.solve(max_time)

        # Collect full SE3 planner data (position + rotation at every node)
        self.planner_data = ob.PlannerData(self.si)
        planner.getPlannerData(self.planner_data)
        self.tree_nodes = []  # (x, y, z, qw, qx, qy, qz)
        self.tree_edges = []  # [((x1,y1,z1), (x2,y2,z2)), ...]

        num_vertices = self.planner_data.numVertices()
        for i in range(num_vertices):
            v1 = self.planner_data.getVertex(i)
            try:
                s1 = v1.getState()
                if s1 is None:
                    continue
            except Exception:
                continue

            rot1 = s1.rotation()
            self.tree_nodes.append((
                s1.getX(), s1.getY(), s1.getZ(),
                rot1.w, rot1.x, rot1.y, rot1.z
            ))

            neighbors = self.planner_data.getEdges(i)
            for ni in neighbors:
                v2 = self.planner_data.getVertex(ni)
                try:
                    s2 = v2.getState()
                    if s2 is None:
                        continue
                except Exception:
                    continue

                self.tree_edges.append([
                    (s1.getX(), s1.getY(), s1.getZ()),
                    (s2.getX(), s2.getY(), s2.getZ())
                ])

        if solved:
            path = pdef.getSolutionPath()
            waypoints = []
            for i in range(path.getStateCount()):
                st = path.getState(i)
                rot = st.rotation()
                # Store full SE3: [x, y, z, qw, qx, qy, qz]
                waypoints.append(np.array([
                    st.getX(), st.getY(), st.getZ(),
                    rot.w, rot.x, rot.y, rot.z
                ]))

            status = solved
            if status == ob.PlannerStatus.APPROXIMATE_SOLUTION:
                print(f"⚠️  Approximate solution. Final: {waypoints[-1][:3]}, Goal: {goal}")

            length = sum(
                np.linalg.norm(waypoints[i][:3] - waypoints[i-1][:3])
                for i in range(1, len(waypoints))
            )
            print(f"   Waypoints: {len(waypoints)}, Path length: {length:.2f}mm")
            return waypoints

        else:
            print("✗ No path found!")
            return None

    def visualize_polyscope(self, waypoints=None):
        import polyscope as ps
        import numpy as np

        self.cleanup()
        ps.init()
        ps.set_up_dir("z_up")

        # 1. OBSTACLES
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

        # 2. SAMPLED NODES + TREE EDGES
        if hasattr(self, 'tree_nodes') and self.tree_nodes:
            # tree_nodes is (x, y, z, qw, qx, qy, qz) — extract positions only for display
            all_nodes_se3 = np.array(self.tree_nodes)          # (N, 7)
            all_positions  = all_nodes_se3[:, :3]              # (N, 3)

            path_pos_set = set()
            path_edge_set = set()
            if waypoints is not None and len(waypoints) > 1:
                wp = np.array(waypoints)
                wp_pos = wp[:, :3]  # SE3 waypoints: grab positions
                for pt in wp_pos:
                    path_pos_set.add(tuple(np.round(pt, 4)))
                for i in range(len(wp_pos) - 1):
                    p1 = tuple(np.round(wp_pos[i], 4))
                    p2 = tuple(np.round(wp_pos[i + 1], 4))
                    path_edge_set.add((p1, p2))
                    path_edge_set.add((p2, p1))

            node_index = {tuple(np.round(p, 4)): i for i, p in enumerate(all_positions)}

            exploration_mask = np.array([
                tuple(np.round(p, 4)) not in path_pos_set
                for p in all_positions
            ])

            if exploration_mask.any():
                ps_samples = ps.register_point_cloud("sampled_nodes", all_positions[exploration_mask])
                ps_samples.set_color((0.0, 0.8, 0.8))
                ps_samples.set_radius(0.0015)

            if (~exploration_mask).any():
                ps_path_nodes = ps.register_point_cloud("path_nodes", all_positions[~exploration_mask])
                ps_path_nodes.set_color((1.0, 1.0, 0.0))
                ps_path_nodes.set_radius(0.003)

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
                        "rrt_tree", all_positions, np.array(tree_only_edges)
                    )
                    ps_tree.set_color((0.0, 0.5, 0.5))
                    ps_tree.set_radius(0.0006)

        # Highlight start and goal poses more visibly
        for label, idx, color in [("robot_start", 0, (0.0, 0.8, 0.0)),
                                   ("robot_goal", -1, (1.0, 0.5, 0.0))]:
            pose = wp[idx]
            transform = self.checker.quaternion_to_matrix(*pose[3:])
            transform[:3, 3] = pose[:3]
            v_hom = np.hstack([self.robot_mesh.vertices.copy(),
                                np.ones((len(self.robot_mesh.vertices), 1))])
            v_transformed = (v_hom @ transform.T)[:, :3]
            ps_ghost = ps.register_surface_mesh(label, v_transformed, self.robot_mesh.faces)
            ps_ghost.set_color(color)
            ps_ghost.set_transparency(0.1)  # less transparent so they stand out
            ps_ghost.set_smooth_shade(True)

        # 3. SOLUTION PATH
        if waypoints is not None and len(waypoints) > 1:
            wp = np.array(waypoints)
            wp_pos = wp[:, :3]  # positions only for curve display
            path_edges = np.array([[i, i + 1] for i in range(len(wp_pos) - 1)])

            ps_path = ps.register_curve_network("planned_path", wp_pos, path_edges)
            ps_path.set_color((1.0, 0.0, 0.0))
            ps_path.set_radius(0.004)

            ps_start = ps.register_point_cloud("start", wp_pos[0:1])
            ps_start.set_color((0.0, 1.0, 0.0))
            ps_start.set_radius(0.012)

            ps_goal = ps.register_point_cloud("goal", wp_pos[-1:])
            ps_goal.set_color((1.0, 0.6, 0.0))
            ps_goal.set_radius(0.012)

        print("🚀 Opening Polyscope Viewer...")
        ps.show()

    def save_path(self, waypoints: List[np.ndarray], filename: str = 'planned_path.npy'):
        path_array = np.array(waypoints)  # shape (N, 7): x y z qw qx qy qz
        np.save(filename, path_array)
        print(f"✓ Path saved to {filename} — shape: {path_array.shape}")
        return filename

    def load_path(self, filename: str) -> List[np.ndarray]:
        path_array = np.load(filename)
        assert path_array.shape[1] == 7, "Expected SE3 path with 7 columns (x,y,z,qw,qx,qy,qz)"
        return list(path_array)

    def cleanup(self):
        """Explicitly release OMPL C++ objects to prevent nanobind leaks."""
        # Destroy in reverse dependency order
        if hasattr(self, 'validity_checker'):
            del self.validity_checker
        if hasattr(self, 'goal_region'):
            del self.goal_region
        if hasattr(self, 'planner_data'):
            del self.planner_data
        if hasattr(self, 'si'):
            del self.si
        if hasattr(self, 'space'):
            del self.space
        import gc
        gc.collect()
