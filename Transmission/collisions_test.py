# debug_collision.py
import numpy as np
from geometry import build_mainshaft, build_countershaft, build_case, get_countershaft_pose
from collision import is_collision_free, transform_cylinder, cylinder_cylinder_collision, cylinder_box_collision

def debug_start_config():
    print("=" * 60)
    print("DEBUGGING START CONFIGURATION")
    print("=" * 60)
    
    # Build geometry
    mainshaft_cyls = build_mainshaft()
    countershaft_cyls = build_countershaft()
    case_walls = build_case()
    countershaft_pose = get_countershaft_pose()
    
    # Start config
    q_start = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    
    print(f"\nStart config: pos={q_start[:3]}, quat={q_start[3:]}")
    
    # Check collision
    collision_free = is_collision_free(
        q_start, mainshaft_cyls, countershaft_cyls, 
        case_walls, countershaft_pose
    )
    
    print(f"\nIs start collision-free? {collision_free}")
    
    if not collision_free:
        print("\nDEBUGGING COLLISIONS AT START:")
        
        # Transform mainshaft cylinders
        pos = q_start[:3]
        quat = q_start[3:]
        
        from scipy.spatial.transform import Rotation
        r = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        
        print("\nMainshaft cylinders in world frame:")
        for i, cyl in enumerate(mainshaft_cyls):
            center = pos + r.apply(cyl.offset)
            axis = r.apply([0, 0, 1])
            print(f"  Cyl {i}: center={center}, r={cyl.radius}, h={cyl.height}, z_range=[{center[2]-cyl.height/2:.1f}, {center[2]+cyl.height/2:.1f}]")
        
        # Transform countershaft
        c_pos = countershaft_pose[:3]
        c_quat = countershaft_pose[3:]
        r_c = Rotation.from_quat([c_quat[1], c_quat[2], c_quat[3], c_quat[0]])
        
        print("\nCountershaft cylinders in world frame:")
        for i, cyl in enumerate(countershaft_cyls):
            center = c_pos + r_c.apply(cyl.offset)
            axis = r_c.apply([0, 0, 1])
            print(f"  Cyl {i}: center={center}, r={cyl.radius}, h={cyl.height}")
        
        # Check specific collisions
        print("\nChecking collisions:")
        
        # Check mainshaft vs countershaft
        for mi, m_cyl in enumerate(mainshaft_cyls):
            m_center = pos + r.apply(m_cyl.offset)
            m_axis = r.apply([0, 0, 1])
            
            for ci, c_cyl in enumerate(countershaft_cyls):
                c_center = c_pos + r_c.apply(c_cyl.offset)
                c_axis = r_c.apply([0, 0, 1])
                
                # Quick distance check
                center_dist = np.linalg.norm(m_center - c_center)
                radial_dist = center_dist - (m_cyl.radius + c_cyl.radius)
                
                if radial_dist < 5:  # Close or colliding
                    print(f"  WARNING: Mainshaft cyl {mi} vs Countershaft cyl {ci}: distance={radial_dist:.1f}mm")
        
        # Check mainshaft vs case walls
        for mi, m_cyl in enumerate(mainshaft_cyls):
            m_center = pos + r.apply(m_cyl.offset)
            m_radius = m_cyl.radius
            
            for wi, wall in enumerate(case_walls):
                # Find closest point on wall to cylinder center
                closest = np.clip(m_center, wall.min, wall.max)
                dist = np.linalg.norm(closest - m_center) - m_radius
                
                if dist < 5:
                    print(f"  WARNING: Mainshaft cyl {mi} vs Wall {wi}: distance={dist:.1f}mm")
                    print(f"    Wall bounds: x=[{wall.min[0]:.0f},{wall.max[0]:.0f}], "
                          f"y=[{wall.min[1]:.0f},{wall.max[1]:.0f}], "
                          f"z=[{wall.min[2]:.0f},{wall.max[2]:.0f}]")
    
    print("\n" + "=" * 60)
    
    # Test a simple move
    print("\nTesting a simple upward move:")
    q_test = np.array([0.0, 50.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # Move up 50mm
    collision_free = is_collision_free(
        q_test, mainshaft_cyls, countershaft_cyls, 
        case_walls, countershaft_pose
    )
    print(f"  Move up 50mm: collision_free={collision_free}")
    
    q_test = np.array([50.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # Move right 50mm
    collision_free = is_collision_free(
        q_test, mainshaft_cyls, countershaft_cyls, 
        case_walls, countershaft_pose
    )
    print(f"  Move right 50mm: collision_free={collision_free}")
    
    q_test = np.array([0.0, 0.0, 50.0, 1.0, 0.0, 0.0, 0.0])  # Move forward 50mm
    collision_free = is_collision_free(
        q_test, mainshaft_cyls, countershaft_cyls, 
        case_walls, countershaft_pose
    )
    print(f"  Move forward 50mm: collision_free={collision_free}")

if __name__ == "__main__":
    debug_start_config()