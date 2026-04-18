#!/usr/bin/env python3
"""
test_planner.py - Comprehensive tests for RRT planner components.

Run with: python3 test_planner.py
Each test prints PASS/FAIL and a reason. Fix failures before running main.py.
"""

import sys
import traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

# ── Geometry constants (must match main.py exactly) ────────────────────────
CASE_THICKNESS  = 25
BEARING_OFFSET  = 215
BEARING_Z       = BEARING_OFFSET + (CASE_THICKNESS / 2)   # 227.5
CS_BEARING_Z    = 100 + CASE_THICKNESS / 2                # 112.5
PRIMARY_LENGTH  = 330
SECONDARY_LENGTH = 330

START = np.array([165 + (PRIMARY_LENGTH / 2), 0.0, BEARING_Z + 1])  # [330, 0, 228.5]
GOAL  = np.array([-148.0, 0.0, BEARING_Z + 1])                       # [-148, 0, 228.5]

# ── Helpers ─────────────────────────────────────────────────────────────────
passed = []
failed = []

def run_test(name, fn):
    try:
        result = fn()
        if result is True or result is None:
            print(f"  ✅ PASS  {name}")
            passed.append(name)
        else:
            print(f"  ❌ FAIL  {name}: {result}")
            failed.append((name, str(result)))
    except Exception as e:
        print(f"  ❌ FAIL  {name}: {e}")
        traceback.print_exc()
        failed.append((name, str(e)))

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — Geometry constants
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Section 1: Geometry Constants ──────────────────────────────────────")

def test_start_goal_direction():
    """START should be at larger X than GOAL (shaft moves left into case)."""
    if START[0] <= GOAL[0]:
        return f"START X={START[0]} must be > GOAL X={GOAL[0]}"
    return True

def test_start_goal_same_z():
    """START and GOAL should share the same Z (bearing centerline)."""
    if not np.isclose(START[2], GOAL[2], atol=1.0):
        return f"START Z={START[2]} vs GOAL Z={GOAL[2]} — should match"
    return True

def test_start_goal_distance():
    """START→GOAL distance should be ~478mm (330 half-length + 148 + gap)."""
    dist = np.linalg.norm(GOAL - START)
    if dist < 200:
        return f"Distance={dist:.1f}mm is suspiciously small — check START/GOAL"
    return True

def test_bearing_z_value():
    """BEARING_Z should be 227.5mm."""
    expected = 227.5
    if not np.isclose(BEARING_Z, expected):
        return f"BEARING_Z={BEARING_Z}, expected {expected}"
    return True

run_test("start_x > goal_x (correct travel direction)", test_start_goal_direction)
run_test("start and goal share same Z", test_start_goal_same_z)
run_test("start-goal distance > 200mm", test_start_goal_distance)
run_test("BEARING_Z == 227.5", test_bearing_z_value)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — quaternion_to_matrix
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Section 2: quaternion_to_matrix ────────────────────────────────────")

from collision import CollisionChecker3D

def test_identity_quaternion():
    """Identity quaternion (1,0,0,0) should produce identity rotation."""
    R = CollisionChecker3D.quaternion_to_matrix(1, 0, 0, 0)
    if not np.allclose(R, np.eye(4), atol=1e-9):
        return f"Identity quaternion gave non-identity matrix:\n{R}"
    return True

def test_90_deg_y():
    """90° about Y: X→-Z, Z→X."""
    import scipy.spatial.transform as sst
    r = sst.Rotation.from_euler('y', 90, degrees=True)
    qx, qy, qz, qw = r.as_quat()
    R = CollisionChecker3D.quaternion_to_matrix(qw, qx, qy, qz)
    # x-axis should map to -z
    x_mapped = R[:3, :3] @ np.array([1, 0, 0])
    expected = np.array([0, 0, -1])
    if not np.allclose(x_mapped, expected, atol=1e-6):
        return f"90°Y: X mapped to {x_mapped}, expected {expected}"
    return True

def test_quaternion_normalization():
    """Slightly denormalized quaternion should still give valid rotation matrix."""
    # Simulate OMPL interpolation denormalization
    q = np.array([1.0001, 0.0001, 0.0001, 0.0001])
    q /= np.linalg.norm(q)
    R = CollisionChecker3D.quaternion_to_matrix(*q)
    det = np.linalg.det(R[:3, :3])
    if not np.isclose(det, 1.0, atol=1e-6):
        return f"Rotation matrix determinant={det:.6f}, expected 1.0"
    return True

def test_180_deg_z():
    """180° about Z: X→-X, Y→-Y."""
    import scipy.spatial.transform as sst
    r = sst.Rotation.from_euler('z', 180, degrees=True)
    qx, qy, qz, qw = r.as_quat()
    R = CollisionChecker3D.quaternion_to_matrix(qw, qx, qy, qz)
    x_mapped = R[:3, :3] @ np.array([1, 0, 0])
    if not np.allclose(x_mapped, np.array([-1, 0, 0]), atol=1e-6):
        return f"180°Z: X mapped to {x_mapped}, expected [-1,0,0]"
    return True

run_test("identity quaternion → identity matrix", test_identity_quaternion)
run_test("90° about Y rotates correctly", test_90_deg_y)
run_test("denormalized quaternion stays valid", test_quaternion_normalization)
run_test("180° about Z rotates correctly", test_180_deg_z)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — CollisionChecker3D mesh operations
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Section 3: CollisionChecker3D ───────────────────────────────────────")
import trimesh

def test_add_mesh_no_collision():
    """Two well-separated boxes should not collide."""
    checker = CollisionChecker3D.__new__(CollisionChecker3D)
    import trimesh.collision
    checker.manager = trimesh.collision.CollisionManager()
    checker.names = []
    checker.meshes = []
    checker.added_meshes = {}
    checker.current_poses = {}

    box1 = trimesh.creation.box([10, 10, 10])
    box2 = trimesh.creation.box([10, 10, 10])
    checker.add_mesh(box1, 'box1', position=(0, 0, 0))
    checker.add_mesh(box2, 'box2', position=(100, 0, 0))

    collisions = checker.check_all_collisions()
    if collisions:
        return f"Separated boxes reported collision: {collisions}"
    return True

def test_add_mesh_with_collision():
    """Two overlapping boxes must report collision."""
    checker = CollisionChecker3D.__new__(CollisionChecker3D)
    import trimesh.collision
    checker.manager = trimesh.collision.CollisionManager()
    checker.names = []
    checker.meshes = []
    checker.added_meshes = {}
    checker.current_poses = {}

    box1 = trimesh.creation.box([20, 20, 20])
    box2 = trimesh.creation.box([20, 20, 20])
    checker.add_mesh(box1, 'box1', position=(0, 0, 0))
    checker.add_mesh(box2, 'box2', position=(5, 0, 0))  # overlapping

    collisions = checker.check_all_collisions()
    if not collisions:
        return "Overlapping boxes did not report collision"
    return True

def test_update_position_clears_collision():
    """Moving a mesh away should clear a collision."""
    checker = CollisionChecker3D.__new__(CollisionChecker3D)
    import trimesh.collision
    checker.manager = trimesh.collision.CollisionManager()
    checker.names = []
    checker.meshes = []
    checker.added_meshes = {}
    checker.current_poses = {}

    box1 = trimesh.creation.box([20, 20, 20])
    box2 = trimesh.creation.box([20, 20, 20])
    checker.add_mesh(box1, 'box1', position=(0, 0, 0))
    checker.add_mesh(box2, 'box2', position=(5, 0, 0))  # colliding

    before = checker.check_all_collisions()
    checker.update_position('box2', position=(200, 0, 0))
    after = checker.check_all_collisions()

    if not before:
        return "Setup failed — boxes not colliding initially"
    if after:
        return f"After move, still colliding: {after}"
    return True

def test_check_mesh_against_manager():
    """check_mesh_against_manager should detect single-mesh collision."""
    checker = CollisionChecker3D.__new__(CollisionChecker3D)
    import trimesh.collision
    checker.manager = trimesh.collision.CollisionManager()
    checker.names = []
    checker.meshes = []
    checker.added_meshes = {}
    checker.current_poses = {}

    obstacle = trimesh.creation.box([50, 50, 50])
    checker.add_mesh(obstacle, 'obstacle', position=(0, 0, 0))

    robot = trimesh.creation.box([10, 10, 10])

    # Inside obstacle — should collide
    t_collide = np.eye(4)
    t_collide[:3, 3] = [0, 0, 0]
    if not checker.check_mesh_against_manager(robot, t_collide):
        return "Robot inside obstacle not detected as collision"

    # Far away — should not collide
    t_clear = np.eye(4)
    t_clear[:3, 3] = [200, 0, 0]
    if checker.check_mesh_against_manager(robot, t_clear):
        return "Robot far from obstacle falsely reported as collision"

    return True

def test_quaternion_collision_transform():
    """Rotated mesh should still collide correctly."""
    checker = CollisionChecker3D.__new__(CollisionChecker3D)
    import trimesh.collision
    checker.manager = trimesh.collision.CollisionManager()
    checker.names = []
    checker.meshes = []
    checker.added_meshes = {}
    checker.current_poses = {}

    # Long thin obstacle along X
    obstacle = trimesh.creation.box([100, 5, 5])
    checker.add_mesh(obstacle, 'obstacle', position=(0, 0, 0))

    # Long thin robot along X — rotated 90° about Z it becomes tall along Y
    robot = trimesh.creation.box([100, 5, 5])

    # At identity: both along X — should collide at origin
    t_identity = np.eye(4)
    if not checker.check_mesh_against_manager(robot, t_identity):
        return "Coaxial meshes at origin should collide"

    # Rotated 90° about Z and offset — should clear
    import scipy.spatial.transform as sst
    r = sst.Rotation.from_euler('z', 90, degrees=True)
    qx, qy, qz, qw = r.as_quat()
    t_rotated = CollisionChecker3D.quaternion_to_matrix(qw, qx, qy, qz)
    t_rotated[:3, 3] = [0, 100, 0]  # move away in Y
    if checker.check_mesh_against_manager(robot, t_rotated):
        return "Rotated and offset robot falsely collides"

    return True

run_test("separated boxes: no collision", test_add_mesh_no_collision)
run_test("overlapping boxes: collision detected", test_add_mesh_with_collision)
run_test("update_position clears collision", test_update_position_clears_collision)
run_test("check_mesh_against_manager: in/out", test_check_mesh_against_manager)
run_test("quaternion rotation respected in collision", test_quaternion_collision_transform)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — Robot mesh centering
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Section 4: Robot Mesh Centering ─────────────────────────────────────")

def test_robot_mesh_loads():
    """Primary shaft SCAD must load without error."""
    from mesh_gen import MeshGenerator
    gen = MeshGenerator(models_folder='models')
    mesh = gen.from_scad('primary_shaft.scad')
    if mesh is None:
        return "from_scad returned None"
    if len(mesh.vertices) == 0:
        return "Mesh has no vertices"
    return True

def test_robot_mesh_centering():
    """After centering by bounds midpoint, mesh bounds must be symmetric."""
    from mesh_gen import MeshGenerator
    gen = MeshGenerator(models_folder='models')
    mesh = gen.from_scad('primary_shaft.scad')
    mesh = mesh.copy()

    # Center by bounds midpoint (more reliable than center_mass for non-uniform meshes)
    midpoint = (mesh.bounds[0] + mesh.bounds[1]) / 2
    mesh.apply_translation(-midpoint)

    bounds = mesh.bounds
    # Each axis: lower bound should equal -upper bound
    for axis, label in enumerate(['X', 'Y', 'Z']):
        lo, hi = bounds[0][axis], bounds[1][axis]
        if not np.isclose(lo, -hi, atol=0.5):
            return f"Mesh not symmetric on {label}: [{lo:.2f}, {hi:.2f}]"
    return True

def test_robot_mesh_long_axis_is_x():
    """Primary shaft should be longest along X axis."""
    from mesh_gen import MeshGenerator
    gen = MeshGenerator(models_folder='models')
    mesh = gen.from_scad('primary_shaft.scad')
    extents = mesh.extents
    if extents[0] < extents[1] or extents[0] < extents[2]:
        return f"X extent {extents[0]:.1f} is not the longest axis. Extents: {extents}"
    return True

def test_robot_mesh_length():
    """Primary shaft should be ~330mm long."""
    from mesh_gen import MeshGenerator
    gen = MeshGenerator(models_folder='models')
    mesh = gen.from_scad('primary_shaft.scad')
    length = mesh.extents[0]
    if not np.isclose(length, PRIMARY_LENGTH, atol=5.0):
        return f"Shaft length={length:.1f}mm, expected ~{PRIMARY_LENGTH}mm"
    return True

run_test("primary_shaft.scad loads", test_robot_mesh_loads)
run_test("mesh centered by bounds midpoint is symmetric", test_robot_mesh_centering)
run_test("longest axis is X", test_robot_mesh_long_axis_is_x)
run_test("shaft length ~330mm", test_robot_mesh_length)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — RRTPlanner3D setup
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Section 5: RRTPlanner3D Setup ───────────────────────────────────────")

from planner import RRTPlanner3D, ShaftPositionGoal, ShaftValidityChecker
from ompl import base as ob

def test_module_level_classes_exist():
    """ShaftPositionGoal and ShaftValidityChecker must be module-level."""
    import planner as planner_module
    if not hasattr(planner_module, 'ShaftPositionGoal'):
        return "ShaftPositionGoal not defined at module level in planner.py"
    if not hasattr(planner_module, 'ShaftValidityChecker'):
        return "ShaftValidityChecker not defined at module level in planner.py"
    return True

def test_planner_initializes():
    """RRTPlanner3D must initialize without error."""
    bounds = [(-400, 400), (-400, 400), (-300, 300)]
    p = RRTPlanner3D(bounds=bounds, models_folder='models')
    if p.space is None:
        return "space is None after init"
    if p.si is None:
        return "si is None after init"
    return True

def test_set_robot_centers_mesh():
    """set_robot must produce a mesh centered near origin."""
    bounds = [(-400, 400), (-400, 400), (-300, 300)]
    p = RRTPlanner3D(bounds=bounds, models_folder='models')
    p.set_robot(scad_file='primary_shaft.scad', start_position=START)

    bounds_mesh = p.robot_mesh.bounds
    midpoint = (bounds_mesh[0] + bounds_mesh[1]) / 2
    # Each axis midpoint should be near zero
    for axis, label in enumerate(['X', 'Y', 'Z']):
        if abs(midpoint[axis]) > 5.0:
            return (f"Robot mesh midpoint {label}={midpoint[axis]:.2f}mm, "
                    f"expected ~0. Use bounds midpoint centering, not center_mass.")
    return True

def test_validity_checker_is_module_level():
    """The stored validity_checker must be ShaftValidityChecker, not a local class."""
    bounds = [(-400, 400), (-400, 400), (-300, 300)]
    p = RRTPlanner3D(bounds=bounds, models_folder='models')
    p.set_robot(scad_file='primary_shaft.scad', start_position=START)
    if not isinstance(p.validity_checker, ShaftValidityChecker):
        return (f"validity_checker is {type(p.validity_checker).__name__}, "
                f"expected ShaftValidityChecker")
    return True

run_test("ShaftPositionGoal and ShaftValidityChecker at module level",
         test_module_level_classes_exist)
run_test("RRTPlanner3D initializes", test_planner_initializes)
run_test("set_robot centers mesh near origin", test_set_robot_centers_mesh)
run_test("validity_checker is module-level ShaftValidityChecker",
         test_validity_checker_is_module_level)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — Validity checking (start/goal positions)
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Section 6: Start / Goal Validity ────────────────────────────────────")

def _make_full_planner():
    """Helper: build a fully configured planner with all obstacles."""
    bounds = [(-400, 400), (-400, 400), (-300, 300)]
    p = RRTPlanner3D(bounds=bounds, models_folder='models')
    p.checker.add_from_scad('transmission_case.scad',
                            name='TransmissionCase',
                            position=(0, 0, 0),
                            parameters={'part': 'case'})
    p.checker.add_from_scad('secondary_shaft.scad',
                            name='CounterShaft',
                            position=(SECONDARY_LENGTH/2 + 1, 0, CS_BEARING_Z),
                            parameters={'part': 'countershaft'})
    p.set_robot(scad_file='primary_shaft.scad', start_position=START)
    return p

def test_start_position_valid():
    """START position must be collision-free."""
    p = _make_full_planner()
    t = np.eye(4)
    t[:3, 3] = START
    collides = p.checker.check_mesh_against_manager(p.robot_mesh, t)
    if collides:
        return (f"START {START} is in collision. "
                f"Shaft half-length={PRIMARY_LENGTH/2}mm — "
                f"move START further from case wall.")
    return True

def test_goal_position_valid():
    """GOAL position must be collision-free."""
    p = _make_full_planner()
    t = np.eye(4)
    t[:3, 3] = GOAL
    collides = p.checker.check_mesh_against_manager(p.robot_mesh, t)
    if collides:
        return (f"GOAL {GOAL} is in collision. "
                f"Adjust GOAL or bearing seat geometry.")
    return True

def test_midpoint_is_blocked():
    """Midpoint between START and GOAL should be blocked by case wall."""
    p = _make_full_planner()
    mid = (START + GOAL) / 2
    t = np.eye(4)
    t[:3, 3] = mid
    collides = p.checker.check_mesh_against_manager(p.robot_mesh, t)
    if not collides:
        return (f"Midpoint {mid} is clear — case wall may not be an obstacle, "
                f"or mesh isn't loaded correctly.")
    return True

def test_shaft_position_goal_distance():
    """ShaftPositionGoal.distanceGoal must return pure XYZ distance."""
    bounds = [(-400, 400), (-400, 400), (-300, 300)]
    p = RRTPlanner3D(bounds=bounds, models_folder='models')
    p.set_robot(scad_file='primary_shaft.scad', start_position=START)

    goal_region = ShaftPositionGoal(p.si, GOAL, tolerance=15.0)

    # Build a state at START
    state = p.space.allocState()
    state.setX(float(START[0]))
    state.setY(float(START[1]))
    state.setZ(float(START[2]))
    rot = state.rotation()
    rot.w, rot.x, rot.y, rot.z = 1.0, 0.0, 0.0, 0.0

    expected = float(np.linalg.norm(START - GOAL))
    got = goal_region.distanceGoal(state)

    if not np.isclose(got, expected, atol=0.1):
        return f"distanceGoal={got:.2f}, expected {expected:.2f}"
    return True

def test_shaft_position_goal_ignores_rotation():
    """ShaftPositionGoal distance must be identical regardless of rotation."""
    bounds = [(-400, 400), (-400, 400), (-300, 300)]
    p = RRTPlanner3D(bounds=bounds, models_folder='models')
    p.set_robot(scad_file='primary_shaft.scad', start_position=START)

    goal_region = ShaftPositionGoal(p.si, GOAL, tolerance=15.0)

    import scipy.spatial.transform as sst

    def distance_at_rotation(roll, pitch, yaw):
        state = p.space.allocState()
        state.setX(float(START[0]))
        state.setY(float(START[1]))
        state.setZ(float(START[2]))
        r = sst.Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=True)
        qx, qy, qz, qw = r.as_quat()
        rot = state.rotation()
        rot.w, rot.x, rot.y, rot.z = float(qw), float(qx), float(qy), float(qz)
        return goal_region.distanceGoal(state)

    d_identity = distance_at_rotation(0, 0, 0)
    d_rotated  = distance_at_rotation(0, 90, 0)

    if not np.isclose(d_identity, d_rotated, atol=0.1):
        return (f"Distance changes with rotation: identity={d_identity:.2f}, "
                f"rotated={d_rotated:.2f}. Rotation component leaking into goal check.")
    return True

run_test("START position is collision-free", test_start_position_valid)
run_test("GOAL position is collision-free", test_goal_position_valid)
run_test("straight-line midpoint is blocked by case", test_midpoint_is_blocked)
run_test("ShaftPositionGoal returns pure XYZ distance", test_shaft_position_goal_distance)
run_test("ShaftPositionGoal ignores rotation", test_shaft_position_goal_ignores_rotation)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7 — euler_to_quat
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Section 7: euler_to_quat ────────────────────────────────────────────")
import scipy.spatial.transform as sst

def euler_to_quat(roll, pitch, yaw, degrees=True):
    r = sst.Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=degrees)
    qx, qy, qz, qw = r.as_quat()
    return (qw, qx, qy, qz)

def test_identity_euler():
    """euler_to_quat(0,0,0) must return (1,0,0,0)."""
    q = euler_to_quat(0, 0, 0)
    if not np.allclose(q, (1, 0, 0, 0), atol=1e-9):
        return f"Identity euler gave {q}"
    return True

def test_euler_unit_norm():
    """All quaternions from euler_to_quat must have unit norm."""
    for angles in [(0,0,0), (0,90,0), (45,0,0), (0,0,180), (30,60,90)]:
        q = euler_to_quat(*angles)
        norm = np.linalg.norm(q)
        if not np.isclose(norm, 1.0, atol=1e-9):
            return f"euler_to_quat{angles} norm={norm:.6f}"
    return True

def test_euler_qw_first():
    """euler_to_quat must return (qw, qx, qy, qz), not scipy's (qx,qy,qz,qw)."""
    # 180° about Z: qw=0, qz=1, qx=qy=0
    q = euler_to_quat(0, 0, 180)
    qw, qx, qy, qz = q
    if not (np.isclose(abs(qw), 0.0, atol=1e-6) and np.isclose(abs(qz), 1.0, atol=1e-6)):
        return (f"180°Z should give qw≈0,qz≈±1 but got qw={qw:.4f},qz={qz:.4f}. "
                f"Check scipy (x,y,z,w) → your (w,x,y,z) unpacking.")
    return True

run_test("identity euler → (1,0,0,0)", test_identity_euler)
run_test("all euler outputs are unit quaternions", test_euler_unit_norm)
run_test("output order is (qw,qx,qy,qz) not scipy default", test_euler_qw_first)

# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════
total = len(passed) + len(failed)
print(f"\n{'═'*60}")
print(f"  Results: {len(passed)}/{total} passed")
if failed:
    print(f"\n  ❌ Failed tests:")
    for name, reason in failed:
        print(f"     • {name}")
        print(f"       → {reason}")
    print(f"\n  Fix these before running main.py")
else:
    print(f"\n  🎉 All tests passed — safe to run main.py")
print(f"{'═'*60}\n")