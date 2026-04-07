# main.py - Updated with proper PyVista handling
import numpy as np
from geometry import build_mainshaft, build_countershaft, build_case, get_countershaft_pose
from planner import RRTConnect
from config import Q_START, Q_GOAL
from viz_matplotlib import save_report_figures
import time

def compute_clearance(config, mainshaft_cyls, countershaft_cyls, case_walls, countershaft_pose):
    """Compute minimum clearance distance to any obstacle."""
    from collision import transform_cylinder
    
    pos = config[:3]
    quat = config[3:]
    
    # Transform mainshaft cylinders
    mainshaft_world = []
    for cyl in mainshaft_cyls:
        center, r, h, axis = transform_cylinder(cyl, pos, quat)
        mainshaft_world.append((center, r, h, axis))
    
    # Transform countershaft
    cshaft_world = []
    c_pos = countershaft_pose[:3]
    c_quat = countershaft_pose[3:]
    for cyl in countershaft_cyls:
        center, r, h, axis = transform_cylinder(cyl, c_pos, c_quat)
        cshaft_world.append((center, r, h, axis))
    
    min_dist = float('inf')
    
    # Check distances to countershaft
    for m_cyl in mainshaft_world:
        for c_cyl in cshaft_world:
            center_dist = np.linalg.norm(m_cyl[0] - c_cyl[0])
            radial_dist = center_dist - (m_cyl[1] + c_cyl[1])
            if radial_dist < min_dist:
                min_dist = radial_dist
    
    # Check distances to case walls
    for m_cyl in mainshaft_world:
        for wall in case_walls:
            closest = np.clip(m_cyl[0], wall.min, wall.max)
            dist = np.linalg.norm(closest - m_cyl[0]) - m_cyl[1]
            if dist < min_dist:
                min_dist = dist
    
    return max(0, min_dist)

if __name__ == "__main__":
    print("=" * 60)
    print("Transmission Mainshaft Removal - RRT Motion Planner")
    print("=" * 60)
    
    # Build models
    mainshaft = build_mainshaft()
    countershaft = build_countershaft()
    case_walls = build_case()
    countershaft_pose = get_countershaft_pose()
    
    print(f"Mainshaft: {len(mainshaft)} cylinders")
    print(f"Countershaft: {len(countershaft)} cylinders")
    print(f"Case walls: {len(case_walls)}")
    
    # Create planner
    planner = RRTConnect(max_iters=20000, step_size=20.0, goal_bias=0.1)
    
    # Plan
    start_time = time.time()
    path = planner.plan()
    elapsed = time.time() - start_time
    
    if path is None:
        print("\n No path found — increase iterations or check collision checker")
        print("Try adjusting step_size, max_iters, or sampling bounds")
    else:
        print(f"\n Path found with {len(path)} waypoints in {elapsed:.2f} seconds")
        
        # Verify path is collision-free
        all_valid = True
        for i, q in enumerate(path):
            if not planner.is_collision_free_config(q):
                print(f"  Warning: Waypoint {i} is in collision!")
                all_valid = False
        
        if all_valid:
            print("  ✓ All waypoints are collision-free")
        
        # Create clearance function for reporting
        def clearance_fn(q):
            return compute_clearance(q, mainshaft, countershaft, case_walls, countershaft_pose)
        
        # Generate report figures
        print("\nGenerating report figures...")
        save_report_figures(
            path, planner.tree_a, planner.tree_b,
            mainshaft, countershaft,
            Q_START, Q_GOAL,
            clearance_fn=clearance_fn,
            output_dir="figures/"
        )
        
        print("\n Done! Check the 'figures/' directory for output images.")
        
        # Optional: Try PyVista visualization
        print("\nAttempting PyVista visualization...")
        try:
            from viz_pyvista import build_scene, draw_rrt_tree, draw_solution_path, animate_path
            plotter, mainshaft_meshes = build_scene(mainshaft, countershaft, countershaft_pose)
            draw_rrt_tree(plotter, planner.tree_a, planner.tree_b)
            draw_solution_path(plotter, path)
            plotter.show()
        except ImportError:
            print("  PyVista not installed. Install with: pip install pyvista")
        except Exception as e:
            print(f"  PyVista error: {e}")
            print("  Continuing without 3D visualization - figures are still saved.")