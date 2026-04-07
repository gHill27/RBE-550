# config.py
import numpy as np
from scipy.spatial.transform import Rotation

# Start: mainshaft positioned with clearance above countershaft
Q_START = np.array([0.0, 20.0, 0.0, 1.0, 0.0, 0.0, 0.0])

# Goal: resting on workbench to the right and slightly down
Q_GOAL = np.array([450.0, 20.0, -180.0, 1.0, 0.0, 0.0, 0.0])

# Configuration space bounds
POS_BOUNDS = np.array([
    [-200, -50, -200],   # min (x, y, z)
    [550, 150, 200]       # max (x, y, z)
])

def slerp_quaternion(q1, q2, t):
    """Spherical Linear Interpolation between two quaternions."""
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    dot = np.dot(q1, q2)
    if dot < 0:
        q2 = -q2
        dot = -dot
    
    dot = np.clip(dot, -1.0, 1.0)
    
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)
    
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    
    w1 = np.sin((1 - t) * theta) / sin_theta
    w2 = np.sin(t * theta) / sin_theta
    
    result = w1 * q1 + w2 * q2
    return result / np.linalg.norm(result)

def sample_config(goal_bias=0.05):
    """Sample random config — occasionally return the goal."""
    if np.random.rand() < goal_bias:
        return Q_GOAL.copy()
    
    # Sample position uniformly within bounds
    pos = np.random.uniform(POS_BOUNDS[0], POS_BOUNDS[1])
    
    # Sample random orientation
    rot = Rotation.random()
    quat = rot.as_quat()  # [x, y, z, w] from scipy
    quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])
    
    return np.concatenate([pos, quat_wxyz])

def steer(q_near, q_rand, step_size=25.0):
    """Move from q_near toward q_rand by step_size (mm)."""
    pos_near = q_near[:3]
    pos_rand = q_rand[:3]
    delta_pos = pos_rand - pos_near
    pos_dist = np.linalg.norm(delta_pos)
    
    if pos_dist <= step_size:
        new_pos = pos_rand.copy()
        new_quat = q_rand[3:7].copy()
    else:
        t = step_size / pos_dist
        new_pos = pos_near + t * delta_pos
        q_near_wxyz = q_near[3:7]
        q_rand_wxyz = q_rand[3:7]
        new_quat = slerp_quaternion(q_near_wxyz, q_rand_wxyz, t)
    
    return np.concatenate([new_pos, new_quat])

def goal_reached(q, q_goal, pos_tol=30.0, rot_tol=0.2):
    """True if config is within tolerance of goal."""
    pos_dist = np.linalg.norm(q[:3] - q_goal[:3])
    if pos_dist > pos_tol:
        return False
    
    q1 = q[3:7]
    q2 = q_goal[3:7]
    dot = abs(np.dot(q1, q2))
    dot = min(1.0, max(0.0, dot))
    angle_diff = 2 * np.arccos(dot)
    
    return angle_diff < rot_tol

def distance_config(q1, q2, pos_weight=1.0, rot_weight=0.15):
    """Distance metric in configuration space."""
    pos_dist = np.linalg.norm(q1[:3] - q2[:3])
    
    q1_quat = q1[3:7]
    q2_quat = q2[3:7]
    dot = abs(np.dot(q1_quat, q2_quat))
    dot = min(1.0, max(0.0, dot))
    angle_dist = 2 * np.arccos(dot)
    
    return pos_weight * pos_dist + rot_weight * angle_dist