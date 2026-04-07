# collision.py
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

class CylinderMesh:
    """Helper class to create and manage cylinder meshes."""
    
    @staticmethod
    def create(radius, height, center, axis, resolution=32):
        """
        Create a cylinder mesh at specified position and orientation.
        
        Args:
            radius: Cylinder radius in mm
            height: Cylinder height in mm
            center: 3D position of cylinder center
            axis: 3D direction vector for cylinder axis
            resolution: Number of segments around circumference
        
        Returns:
            trimesh.Trimesh object
        """
        # Create cylinder along Z axis
        cylinder = trimesh.creation.cylinder(
            radius=radius,
            height=height,
            sections=resolution,
            segment=None
        )
        
        # Compute rotation from Z axis to target axis
        z_axis = np.array([0, 0, 1])
        target_axis = np.array(axis) / np.linalg.norm(axis)
        
        if not np.allclose(z_axis, target_axis):
            # Calculate rotation axis and angle
            rot_axis = np.cross(z_axis, target_axis)
            rot_axis = rot_axis / np.linalg.norm(rot_axis)
            angle = np.arccos(np.clip(np.dot(z_axis, target_axis), -1, 1))
            
            # Create rotation matrix
            R = Rotation.from_rotvec(rot_axis * angle).as_matrix()
            
            # Apply rotation to vertices
            cylinder.vertices = cylinder.vertices @ R.T
        
        # Apply translation
        cylinder.apply_translation(center)
        
        return cylinder


class CollisionChecker:
    """
    Collision checker using trimesh for accurate geometric collision detection.
    Implements CGAL-like functionality in Python.
    """
    
    def __init__(self, mainshaft_cyls, countershaft_cyls, case_walls, 
                 countershaft_pose=None):
        """
        Initialize collision checker with geometric models.
        
        Args:
            mainshaft_cyls: List of Cylinder objects for mainshaft
            countershaft_cyls: List of Cylinder objects for countershaft
            case_walls: List of Box objects for case walls
            countershaft_pose: Fixed pose for countershaft [x,y,z,qw,qx,qy,qz]
        """
        self.mainshaft_cyls = mainshaft_cyls
        self.countershaft_cyls = countershaft_cyls
        self.case_walls = case_walls
        
        # Set default countershaft pose if not provided
        if countershaft_pose is None:
            from geometry import get_countershaft_pose
            countershaft_pose = get_countershaft_pose()
        
        self.countershaft_pose = countershaft_pose
        
        # Pre-build static obstacle meshes for faster collision checking
        self.countershaft_mesh = self._build_countershaft_mesh()
        self.case_mesh = self._build_case_mesh()
        
        # Cache for mainshaft meshes (to avoid rebuilding for same config)
        self.mesh_cache = {}
        self.cache_size = 100  # Maximum cache size
        
    def _build_countershaft_mesh(self):
        """Build a single mesh from all countershaft cylinders."""
        pos = self.countershaft_pose[:3]
        quat = self.countershaft_pose[3:]  # [qw, qx, qy, qz]
        r = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        
        meshes = []
        for cyl in self.countershaft_cyls:
            center = pos + r.apply(cyl.offset)
            axis = r.apply([0, 0, 1])
            
            mesh = CylinderMesh.create(cyl.radius, cyl.height, center, axis)
            meshes.append(mesh)
        
        # Combine all countershaft meshes into one
        if meshes:
            combined = trimesh.util.concatenate(meshes)
            # Simplify mesh for faster collision detection
            combined = combined.simplify_quadratic_decimation(
                target_count=len(combined.faces) // 2
            )
            return combined
        return None
    
    def _build_case_mesh(self):
        """Build a mesh from all case walls."""
        meshes = []
        
        for wall in self.case_walls:
            # Create box from min/max corners
            center = (wall.min + wall.max) / 2
            extents = (wall.max - wall.min) / 2
            
            # Create box mesh
            box = trimesh.creation.box(extents=extents)
            box.apply_translation(center)
            meshes.append(box)
        
        # Combine all case walls into one mesh
        if meshes:
            return trimesh.util.concatenate(meshes)
        return None
    
    def _build_mainshaft_mesh(self, config):
        """
        Build mesh for mainshaft at given configuration.
        Uses caching for performance.
        """
        # Create a cache key (rounded for tolerance)
        cache_key = tuple(np.round(config, decimals=3))
        
        # Check cache
        if cache_key in self.mesh_cache:
            return self.mesh_cache[cache_key]
        
        pos = config[:3]
        quat = config[3:]  # [qw, qx, qy, qz]
        r = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        
        meshes = []
        for cyl in self.mainshaft_cyls:
            center = pos + r.apply(cyl.offset)
            axis = r.apply([0, 0, 1])
            
            mesh = CylinderMesh.create(cyl.radius, cyl.height, center, axis)
            meshes.append(mesh)
        
        if not meshes:
            return None
        
        # Combine all mainshaft cylinders into one mesh
        combined = trimesh.util.concatenate(meshes)
        
        # Cache management
        if len(self.mesh_cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.mesh_cache))
            del self.mesh_cache[oldest_key]
        
        self.mesh_cache[cache_key] = combined
        return combined
    
    def is_collision_free(self, config, clearance=0.5):
        """
        Check if mainshaft at given config collides with any obstacles.
        
        Args:
            config: 7D configuration [x,y,z,qw,qx,qy,qz]
            clearance: Minimum allowed clearance in mm
        
        Returns:
            True if collision-free, False otherwise
        """
        # Build mainshaft mesh
        mainshaft_mesh = self._build_mainshaft_mesh(config)
        if mainshaft_mesh is None:
            return True
        
        # Check collision with countershaft
        if self.countershaft_mesh is not None:
            if mainshaft_mesh.intersects(self.countershaft_mesh):
                return False
            
            # Optional: Check clearance distance
            if clearance > 0:
                dist = mainshaft_mesh.min_distance(self.countershaft_mesh)
                if dist < clearance:
                    return False
        
        # Check collision with case walls
        if self.case_mesh is not None:
            if mainshaft_mesh.intersects(self.case_mesh):
                return False
            
            # Optional: Check clearance distance
            if clearance > 0:
                dist = mainshaft_mesh.min_distance(self.case_mesh)
                if dist < clearance:
                    return False
        
        return True
    
    def get_clearance(self, config):
        """
        Get minimum clearance distance to any obstacle.
        
        Args:
            config: 7D configuration [x,y,z,qw,qx,qy,qz]
        
        Returns:
            Minimum distance to any obstacle in mm
        """
        mainshaft_mesh = self._build_mainshaft_mesh(config)
        if mainshaft_mesh is None:
            return float('inf')
        
        min_dist = float('inf')
        
        # Distance to countershaft
        if self.countershaft_mesh is not None:
            dist = mainshaft_mesh.min_distance(self.countershaft_mesh)
            if dist < min_dist:
                min_dist = dist
        
        # Distance to case walls
        if self.case_mesh is not None:
            dist = mainshaft_mesh.min_distance(self.case_mesh)
            if dist < min_dist:
                min_dist = dist
        
        return min_dist
    
    def get_collision_details(self, config):
        """
        Get detailed collision information.
        
        Returns:
            Dictionary with collision status and details
        """
        mainshaft_mesh = self._build_mainshaft_mesh(config)
        if mainshaft_mesh is None:
            return {'collision': False, 'details': 'No mesh'}
        
        result = {
            'collision': False,
            'countershaft_collision': False,
            'case_collision': False,
            'min_distance': float('inf')
        }
        
        # Check countershaft
        if self.countershaft_mesh is not None:
            if mainshaft_mesh.intersects(self.countershaft_mesh):
                result['collision'] = True
                result['countershaft_collision'] = True
            
            dist = mainshaft_mesh.min_distance(self.countershaft_mesh)
            if dist < result['min_distance']:
                result['min_distance'] = dist
        
        # Check case
        if self.case_mesh is not None:
            if mainshaft_mesh.intersects(self.case_mesh):
                result['collision'] = True
                result['case_collision'] = True
            
            dist = mainshaft_mesh.min_distance(self.case_mesh)
            if dist < result['min_distance']:
                result['min_distance'] = dist
        
        return result


# Global collision checker instance (to avoid rebuilding)
_COLLISION_CHECKER = None

def init_collision_checker(mainshaft_cyls, countershaft_cyls, case_walls, 
                           countershaft_pose=None):
    """
    Initialize the global collision checker.
    Call this once at the start of planning.
    """
    global _COLLISION_CHECKER
    _COLLISION_CHECKER = CollisionChecker(
        mainshaft_cyls, countershaft_cyls, case_walls, countershaft_pose
    )
    return _COLLISION_CHECKER

def is_collision_free(config, mainshaft_cyls=None, countershaft_cyls=None, 
                     case_walls=None, countershaft_pose=None):
    """
    Master collision check function.
    Compatible with the original function signature.
    """
    global _COLLISION_CHECKER
    
    # Initialize if needed
    if _COLLISION_CHECKER is None:
        if mainshaft_cyls is None or countershaft_cyls is None or case_walls is None:
            raise ValueError("Must provide geometry for first call or call init_collision_checker first")
        
        _COLLISION_CHECKER = CollisionChecker(
            mainshaft_cyls, countershaft_cyls, case_walls, countershaft_pose
        )
    
    return _COLLISION_CHECKER.is_collision_free(config)

def get_clearance(config):
    """Get minimum clearance for current configuration."""
    if _COLLISION_CHECKER is None:
        raise RuntimeError("Collision checker not initialized. Call init_collision_checker first.")
    
    return _COLLISION_CHECKER.get_clearance(config)


# Backward compatibility functions (matching original interface)
def transform_cylinder(cyl, position, quaternion):
    """
    Legacy function for backward compatibility.
    Returns cylinder in world frame.
    """
    r = Rotation.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
    center_world = position + r.apply(cyl.offset)
    axis_world = r.apply(np.array([0, 0, 1]))
    return center_world, cyl.radius, cyl.height, axis_world

def cylinder_cylinder_collision(c1_center, c1_r, c1_h, c1_axis,
                                c2_center, c2_r, c2_h, c2_axis,
                                clearance=0.5):
    """
    Legacy cylinder-cylinder collision function.
    Uses trimesh for accurate detection.
    """
    # Create temporary meshes for the two cylinders
    mesh1 = CylinderMesh.create(c1_r, c1_h, c1_center, c1_axis)
    mesh2 = CylinderMesh.create(c2_r, c2_h, c2_center, c2_axis)
    
    # Check intersection
    if mesh1.intersects(mesh2):
        return True
    
    # Check clearance
    if clearance > 0:
        dist = mesh1.min_distance(mesh2)
        return dist < clearance
    
    return False

def cylinder_box_collision(cyl_center, cyl_r, cyl_h, cyl_axis, box, clearance=0.5):
    """
    Legacy cylinder-box collision function.
    """
    # Create cylinder mesh
    cylinder = CylinderMesh.create(cyl_r, cyl_h, cyl_center, cyl_axis)
    
    # Create box mesh
    center = (box.min + box.max) / 2
    extents = (box.max - box.min) / 2
    box_mesh = trimesh.creation.box(extents=extents)
    box_mesh.apply_translation(center)
    
    # Check intersection
    if cylinder.intersects(box_mesh):
        return True
    
    # Check clearance
    if clearance > 0:
        dist = cylinder.min_distance(box_mesh)
        return dist < clearance
    
    return False