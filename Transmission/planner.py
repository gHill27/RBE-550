# planner.py
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
from config import sample_config, steer, goal_reached, Q_START, Q_GOAL, distance_config
from collision import is_collision_free, init_collision_checker
from geometry import build_mainshaft, build_countershaft, build_case, get_countershaft_pose

class RRTConnect:
    def __init__(self, max_iters=5000, step_size=30.0, goal_bias=0.05):
        self.max_iters = max_iters
        self.step_size = step_size
        self.goal_bias = goal_bias
        
        # Initialize trees with nodes and parents
        self.tree_a = {
            'nodes': [Q_START.copy()], 
            'parents': [-1],
            'kd_tree': None  # Will be rebuilt when needed
        }
        self.tree_b = {
            'nodes': [Q_GOAL.copy()], 
            'parents': [-1],
            'kd_tree': None
        }
        
        # Load geometry for collision checking
        
        self.mainshaft_cyls = build_mainshaft()
        self.countershaft_cyls = build_countershaft()
        self.case_walls = build_case()
        self.countershaft_pose = get_countershaft_pose()
        
        init_collision_checker(
            self.mainshaft_cyls, 
            self.countershaft_cyls, 
            self.case_walls, 
            self.countershaft_pose
        )
        
        # KD-Tree rebuild threshold (rebuild every N additions)
        self.rebuild_threshold = 50
        self.nodes_since_rebuild_a = 0
        self.nodes_since_rebuild_b = 0
    
    def rebuild_kdtree(self, tree):
        """Rebuild KD-tree for a given tree's position components."""
        if len(tree['nodes']) == 0:
            tree['kd_tree'] = None
            return
        
        # Extract only position components (x, y, z) for KD-Tree
        positions = np.array([node[:3] for node in tree['nodes']])
        
        # Build KD-Tree with Euclidean distance
        if len(positions) > 0:
            tree['kd_tree'] = KDTree(positions)
        else:
            tree['kd_tree'] = None
    
    def nearest(self, tree, q_rand, use_kdtree=True):
        """
        Find nearest node using KD-Tree for efficiency.
        Returns index of nearest node.
        """
        if len(tree['nodes']) == 0:
            return -1
        
        if use_kdtree and tree['kd_tree'] is not None:
            # Extract query position
            q_pos = q_rand[:3].reshape(1, -1)  # Reshape to 2D for consistent return
            
            # Get k nearest neighbors (k=10 or less if tree is smaller)
            k = min(10, len(tree['nodes']))
            dists, indices = tree['kd_tree'].query(q_pos, k=k)
            
            # Flatten the results (query with 1 point returns 1D arrays)
            dists = dists.flatten() if hasattr(dists, 'flatten') else [dists]
            indices = indices.flatten() if hasattr(indices, 'flatten') else [indices]
            
            # Find best by full distance metric (position + orientation)
            best_idx = indices[0]
            best_dist = distance_config(tree['nodes'][best_idx], q_rand)
            
            for i in range(1, len(indices)):
                idx_candidate = indices[i]
                dist_candidate = distance_config(tree['nodes'][idx_candidate], q_rand)
                if dist_candidate < best_dist:
                    best_dist = dist_candidate
                    best_idx = idx_candidate
            
            return best_idx
        else:
            # Fallback to brute force (O(N))
            best_idx = 0
            best_dist = distance_config(tree['nodes'][0], q_rand)
            
            for i, node in enumerate(tree['nodes'][1:], 1):
                dist = distance_config(node, q_rand)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            
            return best_idx
    
    def add_node_to_tree(self, tree, q_new, parent_idx):
        """Add a new node and update KD-Tree if needed."""
        tree['nodes'].append(q_new)
        tree['parents'].append(parent_idx)
        
        # Track tree type for rebuild threshold
        if tree is self.tree_a:
            self.nodes_since_rebuild_a += 1
            if self.nodes_since_rebuild_a >= self.rebuild_threshold:
                self.rebuild_kdtree(tree)
                self.nodes_since_rebuild_a = 0
        else:
            self.nodes_since_rebuild_b += 1
            if self.nodes_since_rebuild_b >= self.rebuild_threshold:
                self.rebuild_kdtree(tree)
                self.nodes_since_rebuild_b = 0
    
    def is_collision_free_config(self, q):
        """Wrapper for collision checking."""
        return is_collision_free(
            q, self.mainshaft_cyls, self.countershaft_cyls, 
            self.case_walls, self.countershaft_pose
        )
    
    def extend(self, tree, q_rand):
        """Extend tree toward q_rand. Returns new node or None."""
        if len(tree['nodes']) == 0:
            return None, None
            
        idx = self.nearest(tree, q_rand)
        q_near = tree['nodes'][idx]
        q_new = steer(q_near, q_rand, self.step_size)
        
        # Check if the new node is valid
        if self.is_collision_free_config(q_new):
            self.add_node_to_tree(tree, q_new, idx)
            return q_new, len(tree['nodes']) - 1
        return None, None
    
    def connect(self, tree, q_target):
        """
        Repeatedly extend tree toward q_target until reached or blocked.
        Returns (final_node, final_idx, success_flag)
        """
        step_count = 0
        max_steps = 50  # Prevent infinite loops
        
        while step_count < max_steps:
            q_new, idx = self.extend(tree, q_target)
            if q_new is None:
                return None, None, False
            
            if distance_config(q_new, q_target) < self.step_size:
                # Close enough to target
                if self.is_collision_free_config(q_target):
                    self.add_node_to_tree(tree, q_target, idx)
                    return q_target, len(tree['nodes']) - 1, True
                return q_new, idx, False
            
            step_count += 1
        
        return None, None, False
    
    def plan(self):
        """RRT-Connect planning algorithm with KD-Tree optimization."""
        print("Starting RRT-Connect planning with KD-Tree optimization...")
        print(f"Start: {Q_START[:3]}")
        print(f"Goal: {Q_GOAL[:3]}")
        
        # Build initial KD-Trees
        self.rebuild_kdtree(self.tree_a)
        self.rebuild_kdtree(self.tree_b)
        
        for i in range(self.max_iters):
            # Sample random config (with goal bias)
            if np.random.rand() < self.goal_bias:
                q_rand = Q_GOAL.copy()
            else:
                q_rand = sample_config()
            
            # Extend tree_a
            q_new, idx_a = self.extend(self.tree_a, q_rand)
            
            if q_new is not None:
                # Try to connect tree_b to q_new
                q_con, idx_b, success = self.connect(self.tree_b, q_new)
                
                if success:
                    print(f"\n✓ Path found at iteration {i}!")
                    print(f"  Tree A nodes: {len(self.tree_a['nodes'])}")
                    print(f"  Tree B nodes: {len(self.tree_b['nodes'])}")
                    return self.extract_path(idx_a, idx_b)
            
            # Swap trees for bidirectional growth
            self.tree_a, self.tree_b = self.tree_b, self.tree_a
            
            # Progress update
            if (i + 1) % 500 == 0:
                print(f"Iteration {i+1}: Tree A={len(self.tree_a['nodes'])}, "
                      f"Tree B={len(self.tree_b['nodes'])}")
        
        print(f"\n✗ No path found after {self.max_iters} iterations")
        return None
    
    def extract_path(self, idx_a, idx_b):
        """Trace back through both trees and join."""
        # Path from start to connection point
        path_start_to_conn = []
        current_idx = idx_a
        while current_idx >= 0:
            path_start_to_conn.append(self.tree_a['nodes'][current_idx])
            current_idx = self.tree_a['parents'][current_idx]
        path_start_to_conn.reverse()
        
        # Path from connection point to goal
        path_conn_to_goal = []
        current_idx = idx_b
        while current_idx >= 0:
            path_conn_to_goal.append(self.tree_b['nodes'][current_idx])
            current_idx = self.tree_b['parents'][current_idx]
        
        # Combine
        full_path = path_start_to_conn + path_conn_to_goal
        
        # Smooth the path
        smoothed = self.smooth_path(full_path)
        
        return smoothed
    
    def slerp(self, q1, q2, t):
        """Spherical Linear Interpolation between two quaternions."""
        # Normalize inputs
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        
        # Compute dot product
        dot = np.dot(q1, q2)
        
        # If dot is negative, flip one quaternion to take the shorter path
        if dot < 0:
            q2 = -q2
            dot = -dot
        
        # Clamp dot to avoid numerical issues
        dot = np.clip(dot, -1.0, 1.0)
        
        # If quaternions are very close, use linear interpolation
        if dot > 0.9995:
            result = q1 + t * (q2 - q1)
            return result / np.linalg.norm(result)
        
        # Standard SLERP
        theta = np.arccos(dot)
        sin_theta = np.sin(theta)
        
        w1 = np.sin((1 - t) * theta) / sin_theta
        w2 = np.sin(t * theta) / sin_theta
        
        result = w1 * q1 + w2 * q2
        return result / np.linalg.norm(result)
    
    def smooth_path(self, path, attempts=500):
        """Shortcut smoothing with collision checking."""
        if len(path) < 3:
            return path
        
        smoothed = [p.copy() for p in path]
        
        for attempt in range(attempts):
            if len(smoothed) < 3:
                break
            
            # Pick two random indices (not adjacent)
            i = np.random.randint(0, len(smoothed) - 2)
            j = np.random.randint(i + 2, len(smoothed))
            
            q_from = smoothed[i]
            q_to = smoothed[j]
            
            # Check if direct path is collision-free
            num_steps = max(5, int(distance_config(q_from, q_to) / self.step_size) + 1)
            valid = True
            
            for step in range(1, num_steps):
                t = step / num_steps
                # Interpolate position
                pos = (1 - t) * q_from[:3] + t * q_to[:3]
                # SLERP for orientation
                q_from_wxyz = q_from[3:7]
                q_to_wxyz = q_to[3:7]
                q_interp = self.slerp(q_from_wxyz, q_to_wxyz, t)
                q_test = np.concatenate([pos, q_interp])
                
                if not self.is_collision_free_config(q_test):
                    valid = False
                    break
            
            if valid:
                # Replace segment with direct connection
                smoothed = smoothed[:i+1] + smoothed[j:]
        
        return smoothed