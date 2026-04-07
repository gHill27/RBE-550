"""
viz_matplotlib.py
-----------------
Report-quality figures for the RRT transmission planning project.
All functions accept the path and tree data structures produced by planner.py
and write publication-ready PNGs suitable for embedding in a report.

Usage:
    from viz_matplotlib import save_report_figures
    save_report_figures(path, tree_a, tree_b, output_dir="figures/")
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scipy.spatial.transform import Rotation
import os

# ---------------------------------------------------------------------------
# Figure style defaults — tweak once here, applies everywhere
# ---------------------------------------------------------------------------
STYLE = {
    "fig_dpi":       150,
    "fig_size":      (10, 7),
    "path_color":    "#E84040",   # red solution path
    "tree_color":    "#AAAAAA",   # gray RRT tree edges
    "case_color":    "#444444",   # case wireframe
    "shaft_color":   "#4A90D9",   # mainshaft cylinders
    "counter_color": "#888888",   # countershaft cylinders
    "start_color":   "#2ECC71",   # start marker
    "goal_color":    "#E67E22",   # goal marker
    "bg_color":      "#F8F8F8",
    "font":          "DejaVu Sans",
}

# Case dimensions (mm) — must match geometry.py
CASE = dict(lx=280, ly=210, lz=300, wall=25)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_fig(title):
    """Create a styled figure + 3D axes."""
    fig = plt.figure(figsize=STYLE["fig_size"], dpi=STYLE["fig_dpi"],
                     facecolor=STYLE["bg_color"])
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(STYLE["bg_color"])
    ax.set_xlabel("X (mm)", fontsize=9, labelpad=6)
    ax.set_ylabel("Y (mm)", fontsize=9, labelpad=6)
    ax.set_zlabel("Z (mm)", fontsize=9, labelpad=6)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
    ax.tick_params(labelsize=7)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    return fig, ax


def _draw_case_wireframe(ax):
    """Draw the transmission case as a wireframe box."""
    hx = CASE["lx"] / 2
    hy = CASE["ly"] / 2
    hz = CASE["lz"] / 2
    # 8 corners of the case interior
    corners = np.array([
        [-hx, -hy, -hz], [ hx, -hy, -hz],
        [ hx,  hy, -hz], [-hx,  hy, -hz],
        [-hx, -hy,  hz], [ hx, -hy,  hz],
        [ hx,  hy,  hz], [-hx,  hy,  hz],
    ])
    # 12 edges
    edges = [
        (0,1),(1,2),(2,3),(3,0),  # bottom face
        (4,5),(5,6),(6,7),(7,4),  # top face
        (0,4),(1,5),(2,6),(3,7),  # verticals
    ]
    for i, j in edges:
        ax.plot(*zip(corners[i], corners[j]),
                color=STYLE["case_color"], lw=0.8, alpha=0.5, zorder=1)


def _cylinder_surface(center, radius, height, axis=(0, 0, 1), n=20):
    """
    Return (X, Y, Z) arrays for a cylinder surface suitable for ax.plot_surface.
    axis: direction the cylinder's long axis points (unit vector).
    """
    theta = np.linspace(0, 2 * np.pi, n)
    z_pts = np.array([-height / 2, height / 2])
    theta_grid, z_grid = np.meshgrid(theta, z_pts)

    # Local frame: cylinder axis along Z
    X_loc = radius * np.cos(theta_grid)
    Y_loc = radius * np.sin(theta_grid)
    Z_loc = z_grid

    # Rotate local frame so Z aligns with `axis`
    axis = np.array(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    z_hat = np.array([0, 0, 1.0])
    if not np.allclose(axis, z_hat):
        rot_axis = np.cross(z_hat, axis)
        rot_axis /= np.linalg.norm(rot_axis)
        angle = np.arccos(np.clip(np.dot(z_hat, axis), -1, 1))
        R = Rotation.from_rotvec(rot_axis * angle).as_matrix()
    else:
        R = np.eye(3)

    pts_local = np.stack([X_loc.ravel(), Y_loc.ravel(), Z_loc.ravel()])
    pts_world = R @ pts_local + np.array(center)[:, None]

    X = pts_world[0].reshape(theta_grid.shape) + center[0]  # already added above — simplify below
    # Redo cleanly
    pts_world = (R @ np.stack([X_loc.ravel(), Y_loc.ravel(), Z_loc.ravel()])).T
    pts_world += np.array(center)
    X = pts_world[:, 0].reshape(theta_grid.shape)
    Y = pts_world[:, 1].reshape(theta_grid.shape)
    Z = pts_world[:, 2].reshape(theta_grid.shape)
    return X, Y, Z


def _draw_stacked_cylinders(ax, cylinders, config, color, alpha=0.6, label=None):
    """
    Draw a list of Cylinder objects transformed by a 7-DOF config [x,y,z,qw,qx,qy,qz].
    cylinders: list of geometry.Cylinder
    config: np.array of shape (7,)
    """
    pos  = config[:3]
    quat = config[3:]   # [qw, qx, qy, qz]

    # scipy uses [x,y,z,w] — convert
    R = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()

    first = True
    for cyl in cylinders:
        # Rotate local offset into world frame
        center_world = pos + R @ cyl.offset

        # Rotate cylinder axis (local Z) into world frame
        axis_world = R @ np.array([0, 0, 1.0])

        X, Y, Z = _cylinder_surface(center_world, cyl.radius, cyl.height, axis=axis_world)
        lbl = label if first else None
        ax.plot_surface(X, Y, Z, color=color, alpha=alpha, linewidth=0,
                        antialiased=True, label=lbl, zorder=3)
        first = False


def _path_centers(path):
    """Extract XYZ centroid positions from a list of configs."""
    return np.array([q[:3] for q in path])


# ---------------------------------------------------------------------------
# Figure 1 — 3D solution path with case and shafts
# ---------------------------------------------------------------------------

def fig_solution_path(path, mainshaft_cyls, countershaft_cyls,
                      q_start, q_goal, output_dir="figures/"):
    """
    Full 3D view: case wireframe, countershaft in gray,
    mainshaft at start (transparent) and goal (solid),
    solution path as a red line.
    """
    fig, ax = _setup_fig("RRT Solution Path — Mainshaft Removal")
    _draw_case_wireframe(ax)

    # Countershaft — static gray
    _draw_stacked_cylinders(ax, countershaft_cyls,
                             np.array([0, -60, 0, 1, 0, 0, 0]),   # fixed pose
                             color=STYLE["counter_color"], alpha=0.35,
                             label="Countershaft")

    # Mainshaft at start — ghost
    _draw_stacked_cylinders(ax, mainshaft_cyls, q_start,
                             color=STYLE["shaft_color"], alpha=0.2,
                             label="Mainshaft (start)")

    # Mainshaft at goal — solid
    _draw_stacked_cylinders(ax, mainshaft_cyls, q_goal,
                             color=STYLE["shaft_color"], alpha=0.7,
                             label="Mainshaft (goal)")

    # Solution path centerline
    centers = _path_centers(path)
    ax.plot(centers[:, 0], centers[:, 1], centers[:, 2],
            color=STYLE["path_color"], lw=2.5, zorder=5, label="Solution path")

    # Start / goal markers
    ax.scatter(*q_start[:3], color=STYLE["start_color"], s=60, zorder=6, label="Start")
    ax.scatter(*q_goal[:3],  color=STYLE["goal_color"],  s=60, zorder=6, label="Goal")

    ax.legend(fontsize=7, loc="upper left", framealpha=0.8)
    _save(fig, output_dir, "fig1_solution_path.png")


# ---------------------------------------------------------------------------
# Figure 2 — RRT tree growth (both trees for bidirectional)
# ---------------------------------------------------------------------------

def fig_rrt_tree(tree_a, tree_b, path, output_dir="figures/"):
    """
    Draw the full RRT tree(s) in 3D as thin gray lines,
    with the solution path highlighted in red on top.
    """
    fig, ax = _setup_fig("RRT Tree Growth")
    _draw_case_wireframe(ax)

    def _draw_tree(tree, color, alpha):
        nodes   = tree["nodes"]
        parents = tree["parents"]
        segs = []
        for i, parent_idx in enumerate(parents):
            if parent_idx < 0:
                continue
            segs.append([nodes[parent_idx][:3], nodes[i][:3]])
        if segs:
            lc = Line3DCollection(segs, colors=color, linewidths=0.3, alpha=alpha)
            ax.add_collection3d(lc)

    _draw_tree(tree_a, color=STYLE["tree_color"], alpha=0.4)
    if tree_b is not None:
        _draw_tree(tree_b, color="#AABBCC", alpha=0.3)  # second tree slightly blue-tinted

    # Solution path on top
    if path:
        centers = _path_centers(path)
        ax.plot(centers[:, 0], centers[:, 1], centers[:, 2],
                color=STYLE["path_color"], lw=2.5, zorder=5, label="Solution path")

    n_nodes = len(tree_a["nodes"]) + (len(tree_b["nodes"]) if tree_b else 0)
    ax.set_title(f"RRT Tree ({n_nodes} total nodes)", fontsize=11, fontweight="bold", pad=12)

    patch_a = mpatches.Patch(color=STYLE["tree_color"], label=f"Tree A ({len(tree_a['nodes'])} nodes)")
    patch_p = mpatches.Patch(color=STYLE["path_color"], label="Solution path")
    ax.legend(handles=[patch_a, patch_p], fontsize=7, loc="upper left")

    _save(fig, output_dir, "fig2_rrt_tree.png")


# ---------------------------------------------------------------------------
# Figure 3 — Position trajectory (X, Y, Z vs waypoint index)
# ---------------------------------------------------------------------------

def fig_position_trajectory(path, output_dir="figures/"):
    """
    2D line plot of X, Y, Z position along the path.
    Good for checking the path is smooth and monotonically leaving the case.
    """
    centers = _path_centers(path)
    idx = np.arange(len(centers))

    fig, axes = plt.subplots(3, 1, figsize=(9, 6), dpi=STYLE["fig_dpi"],
                             facecolor=STYLE["bg_color"], sharex=True)
    fig.suptitle("Mainshaft Position Along Solution Path", fontsize=11, fontweight="bold")

    labels = ["X (mm)", "Y (mm)", "Z (mm)"]
    colors = ["#E84040", "#2ECC71", "#4A90D9"]
    for i, (ax, lbl, col) in enumerate(zip(axes, labels, colors)):
        ax.set_facecolor(STYLE["bg_color"])
        ax.plot(idx, centers[:, i], color=col, lw=1.8)
        ax.set_ylabel(lbl, fontsize=9)
        ax.grid(True, lw=0.3, alpha=0.5)
        ax.tick_params(labelsize=7)

    axes[-1].set_xlabel("Waypoint index", fontsize=9)
    plt.tight_layout()
    _save(fig, output_dir, "fig3_position_trajectory.png")


# ---------------------------------------------------------------------------
# Figure 4 — Orientation trajectory (Euler angles vs waypoint index)
# ---------------------------------------------------------------------------

def fig_orientation_trajectory(path, output_dir="figures/"):
    """
    Plot roll, pitch, yaw along the path.
    Useful to verify no wild flips in orientation.
    """
    eulers = []
    for q in path:
        quat = q[3:]  # [qw, qx, qy, qz]
        r = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        eulers.append(np.degrees(r.as_euler("xyz")))
    eulers = np.array(eulers)
    idx = np.arange(len(eulers))

    fig, axes = plt.subplots(3, 1, figsize=(9, 6), dpi=STYLE["fig_dpi"],
                             facecolor=STYLE["bg_color"], sharex=True)
    fig.suptitle("Mainshaft Orientation Along Solution Path", fontsize=11, fontweight="bold")

    labels = ["Roll (°)", "Pitch (°)", "Yaw (°)"]
    colors = ["#E84040", "#2ECC71", "#4A90D9"]
    for i, (ax, lbl, col) in enumerate(zip(axes, labels, colors)):
        ax.set_facecolor(STYLE["bg_color"])
        ax.plot(idx, eulers[:, i], color=col, lw=1.8)
        ax.set_ylabel(lbl, fontsize=9)
        ax.grid(True, lw=0.3, alpha=0.5)
        ax.tick_params(labelsize=7)

    axes[-1].set_xlabel("Waypoint index", fontsize=9)
    plt.tight_layout()
    _save(fig, output_dir, "fig4_orientation_trajectory.png")


# ---------------------------------------------------------------------------
# Figure 5 — Top-down (XZ) and side (XY) 2D projections
# ---------------------------------------------------------------------------

def fig_2d_projections(path, tree_a, output_dir="figures/"):
    """
    2D projections of the tree and path — useful for the report
    to show how the planner navigates around the case in plan view.
    """
    nodes   = np.array([n[:3] for n in tree_a["nodes"]])
    parents = tree_a["parents"]
    centers = _path_centers(path)

    fig, (ax_top, ax_side) = plt.subplots(1, 2, figsize=(12, 5),
                                          dpi=STYLE["fig_dpi"],
                                          facecolor=STYLE["bg_color"])
    fig.suptitle("RRT Path Projections", fontsize=11, fontweight="bold")

    for ax, (xi, yi), xlabel, ylabel, title in [
        (ax_top,  (0, 2), "X (mm)", "Z (mm)", "Top view (XZ)"),
        (ax_side, (0, 1), "X (mm)", "Y (mm)", "Side view (XY)"),
    ]:
        ax.set_facecolor(STYLE["bg_color"])
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, lw=0.3, alpha=0.4)
        ax.set_aspect("equal")

        # Draw case rectangle
        hx = CASE["lx"] / 2
        hz = CASE["lz"] / 2
        hy = CASE["ly"] / 2
        if xi == 0 and yi == 2:
            ax.add_patch(mpatches.Rectangle((-hx, -hz), CASE["lx"], CASE["lz"],
                         fill=False, edgecolor=STYLE["case_color"], lw=1.2))
        else:
            ax.add_patch(mpatches.Rectangle((-hx, -hy), CASE["lx"], CASE["ly"],
                         fill=False, edgecolor=STYLE["case_color"], lw=1.2))

        # Tree edges
        for i, parent_idx in enumerate(parents):
            if parent_idx < 0:
                continue
            p = nodes[parent_idx]
            c = nodes[i]
            ax.plot([p[xi], c[xi]], [p[yi], c[yi]],
                    color=STYLE["tree_color"], lw=0.3, alpha=0.5)

        # Solution path
        ax.plot(centers[:, xi], centers[:, yi],
                color=STYLE["path_color"], lw=2.0, zorder=5, label="Path")

        # Start / goal
        ax.scatter(path[0][xi],  path[0][yi],
                   color=STYLE["start_color"], s=50, zorder=6)
        ax.scatter(path[-1][xi], path[-1][yi],
                   color=STYLE["goal_color"],  s=50, zorder=6)

    plt.tight_layout()
    _save(fig, output_dir, "fig5_2d_projections.png")


# ---------------------------------------------------------------------------
# Figure 6 — Collision clearance along path
# ---------------------------------------------------------------------------

def fig_clearance(path, clearance_fn, output_dir="figures/"):
    """
    Plot minimum clearance distance at each waypoint along the path.
    clearance_fn: callable(config) -> float (minimum distance to any obstacle)
    Zero clearance = in collision, which should never happen on a valid path.
    """
    clearances = [clearance_fn(q) for q in path]
    idx = np.arange(len(clearances))

    fig, ax = plt.subplots(figsize=(9, 4), dpi=STYLE["fig_dpi"],
                           facecolor=STYLE["bg_color"])
    ax.set_facecolor(STYLE["bg_color"])
    ax.fill_between(idx, clearances, alpha=0.2, color="#4A90D9")
    ax.plot(idx, clearances, color="#4A90D9", lw=1.8, label="Min clearance")
    ax.axhline(0, color=STYLE["path_color"], lw=1.0, ls="--", label="Zero clearance")
    ax.set_xlabel("Waypoint index", fontsize=9)
    ax.set_ylabel("Clearance (mm)", fontsize=9)
    ax.set_title("Minimum Obstacle Clearance Along Path", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, lw=0.3, alpha=0.5)
    ax.tick_params(labelsize=7)
    plt.tight_layout()
    _save(fig, output_dir, "fig6_clearance.png")


# ---------------------------------------------------------------------------
# Master export — call this from main.py
# ---------------------------------------------------------------------------

def save_report_figures(path, tree_a, tree_b,
                        mainshaft_cyls, countershaft_cyls,
                        q_start, q_goal,
                        clearance_fn=None,
                        output_dir="figures/"):
    """
    Generate and save all report figures to output_dir.

    Parameters
    ----------
    path            : list of np.array configs [x,y,z,qw,qx,qy,qz]
    tree_a, tree_b  : dict with keys 'nodes' and 'parents' (tree_b may be None)
    mainshaft_cyls  : list of geometry.Cylinder
    countershaft_cyls: list of geometry.Cylinder
    q_start, q_goal : np.array configs
    clearance_fn    : optional callable(config) -> float
    output_dir      : folder to write PNGs into
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Saving figure 1: solution path...")
    fig_solution_path(path, mainshaft_cyls, countershaft_cyls,
                      q_start, q_goal, output_dir)

    print("Saving figure 2: RRT tree...")
    fig_rrt_tree(tree_a, tree_b, path, output_dir)

    print("Saving figure 3: position trajectory...")
    fig_position_trajectory(path, output_dir)

    print("Saving figure 4: orientation trajectory...")
    fig_orientation_trajectory(path, output_dir)

    print("Saving figure 5: 2D projections...")
    fig_2d_projections(path, tree_a, output_dir)

    if clearance_fn is not None:
        print("Saving figure 6: clearance...")
        fig_clearance(path, clearance_fn, output_dir)

    print(f"All figures saved to {output_dir}")


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _save(fig, output_dir, filename):
    path_out = os.path.join(output_dir, filename)
    fig.savefig(path_out, dpi=STYLE["fig_dpi"], bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  -> {path_out}")