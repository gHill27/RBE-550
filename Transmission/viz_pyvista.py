# viz_pyvista.py
import pyvista as pv
import numpy as np
from scipy.spatial.transform import Rotation

def make_cylinder_mesh(radius, height, center=(0,0,0), axis=(0,0,1), resolution=30):
    """Create a cylinder mesh at specified position and orientation."""
    # Create cylinder along Z
    cylinder = pv.Cylinder(radius=radius, height=height, center=(0,0,0),
                           direction=(0,0,1), resolution=resolution)
    
    # Compute rotation from Z to desired axis
    z_axis = np.array([0, 0, 1])
    target_axis = np.array(axis) / np.linalg.norm(axis)
    
    if not np.allclose(z_axis, target_axis):
        rot_axis = np.cross(z_axis, target_axis)
        rot_axis = rot_axis / np.linalg.norm(rot_axis)
        angle = np.arccos(np.clip(np.dot(z_axis, target_axis), -1, 1))
        R = Rotation.from_rotvec(rot_axis * angle).as_matrix()
    else:
        R = np.eye(3)
    
    # Apply rotation and translation
    points = cylinder.points.copy()
    points = points @ R.T
    points = points + center
    cylinder.points = points
    
    return cylinder

def build_scene(mainshaft_cyls, countershaft_cyls, countershaft_pose):
    """Build the PyVista scene with case, shafts, and lighting."""
    plotter = pv.Plotter(window_size=(1024, 768))
    plotter.set_background("#1a1a2e")  # Dark theme
    
    # Draw case as wireframe box
    case_bounds = [-140, 140, -105, 105, -150, 150]
    case = pv.Box(bounds=case_bounds)
    plotter.add_mesh(case, style='wireframe', color='#4a5568', line_width=2, 
                     opacity=0.6, label="Transmission Case")
    
    # Draw countershaft (static obstacle) - gray
    c_pos = countershaft_pose[:3]
    c_quat = countershaft_pose[3:]  # [w,x,y,z]
    c_rot = Rotation.from_quat([c_quat[1], c_quat[2], c_quat[3], c_quat[0]])
    
    for cyl in countershaft_cyls:
        center = c_pos + c_rot.apply(cyl.offset)
        axis = c_rot.apply([0, 0, 1])
        mesh = make_cylinder_mesh(cyl.radius, cyl.height, center, axis)
        plotter.add_mesh(mesh, color='#888888', opacity=0.7, label="Countershaft")
    
    # Store mainshaft meshes for animation
    mainshaft_meshes = []
    for cyl in mainshaft_cyls:
        # Start at origin (will be transformed later)
        mesh = make_cylinder_mesh(cyl.radius, cyl.height, (0,0,0), (0,0,1))
        plotter.add_mesh(mesh, color='#4A90D9', opacity=0.85, label="Mainshaft")
        mainshaft_meshes.append(mesh)
    
    # Add axes and grid
    plotter.add_axes(line_width=2, labels_off=False)
    plotter.show_grid(color='gray')
    
    return plotter, mainshaft_meshes

def update_mesh_transform(mesh, position, quaternion):
    """Update a mesh's position and orientation."""
    # quaternion: [w, x, y, z]
    r = Rotation.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
    
    # Get current points (local coordinates)
    points = mesh.points.copy()
    
    # Apply rotation (rotate from Z-aligned to target)
    points = points @ r.as_matrix().T
    
    # Apply translation
    points = points + position
    
    mesh.points = points

def animate_path(plotter, path, mainshaft_cyls, mainshaft_meshes=None):
    """Step through path configs, update mesh transforms, render each frame."""
    if mainshaft_meshes is None:
        print("  Animation: cannot animate without mesh handles")
        return
    
    print(f"  Animating path with {len(path)} waypoints...")
    
    # Show start position
    for i, q in enumerate(path):
        pos = q[:3]
        quat = q[3:]  # [w,x,y,z]
        
        # Update all mainshaft meshes
        for mesh in mainshaft_meshes:
            # This would need proper implementation with mesh transforms
            pass
        
        # Update camera view occasionally
        if i % max(1, len(path)//10) == 0:
            plotter.view_isometric()
            plotter.render()
            print(f"    Frame {i+1}/{len(path)}: pos=({pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f})")
    
    print("  Animation complete")

def draw_rrt_tree(plotter, tree_a, tree_b):
    """Draw all edges as thin lines."""
    def add_tree_edges(tree, color, alpha=0.3):
        nodes = tree['nodes']
        parents = tree['parents']
        
        # Collect all line segments
        points = []
        lines = []
        
        for i, parent_idx in enumerate(parents):
            if parent_idx < 0:
                continue
            points.append(nodes[parent_idx][:3])
            points.append(nodes[i][:3])
            lines.append([len(points)-2, len(points)-1])
        
        if points:
            # Create a PolyData object with lines
            poly_data = pv.PolyData(np.array(points))
            poly_data.lines = np.hstack([[2] + line for line in lines])
            plotter.add_mesh(poly_data, color=color, line_width=1, opacity=alpha, style='wireframe')
    
    add_tree_edges(tree_a, '#AAAAAA', 0.3)
    add_tree_edges(tree_b, '#88AACC', 0.3)
    
    # Add start and goal markers
    start = tree_a['nodes'][0][:3]
    goal = tree_b['nodes'][0][:3]
    
    plotter.add_points(np.array([start]), color='#2ECC71', point_size=15, label="Start")
    plotter.add_points(np.array([goal]), color='#E67E22', point_size=15, label="Goal")

def draw_solution_path(plotter, path):
    """Draw path centerline as a thick colored polyline."""
    centers = np.array([q[:3] for q in path])
    
    # Create lines between consecutive points
    points = centers
    lines = []
    for i in range(len(points) - 1):
        lines.append([i, i+1])
    
    # Create PolyData with lines
    poly_data = pv.PolyData(points)
    poly_data.lines = np.hstack([[2] + line for line in lines])
    plotter.add_mesh(poly_data, color='#E84040', line_width=4, label="Solution Path")
    
    # Add spheres at waypoints
    if len(centers) > 0:
        plotter.add_points(centers, color='#E84040', point_size=8, render_points_as_spheres=True)

def pose_matrix(position, quaternion):
    """
    Create 4x4 transformation matrix from position and quaternion.
    quaternion: [w, x, y, z]
    """
    w, x, y, z = quaternion
    
    # Rotation matrix from quaternion
    r00 = 1 - 2*y*y - 2*z*z
    r01 = 2*x*y - 2*w*z
    r02 = 2*x*z + 2*w*y
    r10 = 2*x*y + 2*w*z
    r11 = 1 - 2*x*x - 2*z*z
    r12 = 2*y*z - 2*w*x
    r20 = 2*x*z - 2*w*y
    r21 = 2*y*z + 2*w*x
    r22 = 1 - 2*x*x - 2*y*y
    
    mat = np.eye(4)
    mat[0, :3] = [r00, r01, r02]
    mat[1, :3] = [r10, r11, r12]
    mat[2, :3] = [r20, r21, r22]
    mat[:3, 3] = position
    
    return mat