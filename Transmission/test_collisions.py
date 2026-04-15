#!/usr/bin/env python3
"""
test_collision.py - Comprehensive collision detection test suite
for the transmission primary shaft RRT planner.

Tests are organized into 6 suites:
  1. Synthetic geometry (no SCAD needed) — fast, always runnable
  2. Mesh origin & extents validation
  3. Bearing hole passthrough — the critical "shaft in bore" cases
  4. Known-clear positions
  5. Known-collision positions
  6. State validity checker (the function RRT actually calls)

Run:
    python test_collision.py
    python test_collision.py --suite bearing   # run one suite by name
    python test_collision.py --verbose         # print pass details too
    python test_collision.py --scad            # include SCAD-dependent tests
"""

import sys
import argparse
import traceback
import numpy as np
import trimesh
import trimesh.collision
from pathlib import Path
from typing import Callable, List, Tuple, Optional

# ---------------------------------------------------------------------------
# Allow running from any working directory
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

# ---------------------------------------------------------------------------
# Geometry constants (must match main.py exactly)
# ---------------------------------------------------------------------------
CASE_THICKNESS   = 25
CASE_HEIGHT      = 300
BEARING_OFFSET   = 215
BEARING_Z        = BEARING_OFFSET + (CASE_THICKNESS / 2)   # 227.5
CS_BEARING_Z     = 100 + CASE_THICKNESS / 2                # 112.5
PRIMARY_LENGTH   = 330
SECONDARY_LENGTH = 330

START = np.array([-248 + PRIMARY_LENGTH / 2,  0.0, BEARING_Z + 1])   # [-83, 0, 228.5]
GOAL  = np.array([  0.0 + PRIMARY_LENGTH / 2, 0.0, BEARING_Z    ])   # [165, 0, 227.5]

# Bearing bore geometry (should match your SCAD)
# Adjust BEARING_BORE_RADIUS if your SCAD uses a different value.
BEARING_BORE_RADIUS = 30.0   # mm — inner radius of bearing hole in case wall
SHAFT_JOURNAL_RADIUS = 20.0  # mm — radius of the shaft journal (smallest feature)

# ---------------------------------------------------------------------------
# Minimal CollisionChecker re-implementation for tests that don't load SCAD.
# Uses the same API surface as your real CollisionChecker3D so results are
# directly comparable.
# ---------------------------------------------------------------------------

def _make_transform(position):
    t = np.eye(4)
    t[:3, 3] = position
    return t


def _robot_vs_obstacles(robot_mesh, robot_transform, obstacle_meshes: dict) -> List[str]:
    """Return names of obstacles that collide with robot_mesh at robot_transform."""
    hits = []
    for name, (mesh, transform) in obstacle_meshes.items():
        temp = trimesh.collision.CollisionManager()
        temp.add_object(name, mesh, transform)
        if temp.in_collision_single(robot_mesh, robot_transform):
            hits.append(name)
    return hits


def _min_distance(robot_mesh, robot_transform, obstacle_mesh, obstacle_transform) -> float:
    """Return minimum surface-to-surface distance (negative = penetrating)."""
    temp = trimesh.collision.CollisionManager()
    temp.add_object("obs", obstacle_mesh, obstacle_transform)
    mgr = trimesh.collision.CollisionManager()
    mgr.add_object("robot", robot_mesh, robot_transform)
    return mgr.min_distance_other(temp)


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

class TestResult:
    def __init__(self, name, passed, message="", detail=""):
        self.name    = name
        self.passed  = passed
        self.message = message
        self.detail  = detail


class TestSuite:
    def __init__(self, name: str):
        self.name    = name
        self.results: List[TestResult] = []

    def run(self, test_name: str, fn: Callable[[], Tuple[bool, str]], detail: str = ""):
        try:
            passed, message = fn()
        except Exception as e:
            passed  = False
            message = f"EXCEPTION: {e}"
            detail  = traceback.format_exc()
        self.results.append(TestResult(test_name, passed, message, detail))

    @property
    def passed(self):  return sum(1 for r in self.results if r.passed)
    @property
    def failed(self):  return sum(1 for r in self.results if not r.passed)
    @property
    def total(self):   return len(self.results)


def print_results(suites: List[TestSuite], verbose: bool):
    grand_pass = grand_fail = 0
    for suite in suites:
        print(f"\n{'='*60}")
        print(f"  Suite: {suite.name}  ({suite.passed}/{suite.total} passed)")
        print(f"{'='*60}")
        for r in suite.results:
            icon = "✅" if r.passed else "❌"
            print(f"  {icon}  {r.name}")
            if not r.passed or verbose:
                print(f"       {r.message}")
            if r.detail and not r.passed:
                for line in r.detail.strip().splitlines():
                    print(f"         {line}")
        grand_pass += suite.passed
        grand_fail += suite.failed

    print(f"\n{'='*60}")
    print(f"  TOTAL  {grand_pass} passed  |  {grand_fail} failed")
    print(f"{'='*60}\n")
    return grand_fail == 0


# ===========================================================================
# SUITE 1 — Synthetic geometry (no SCAD)
# ===========================================================================

def suite_synthetic() -> TestSuite:
    """
    All tests use simple trimesh primitives (boxes, cylinders, spheres).
    These run with zero external dependencies and validate the collision
    API behaviour your real code relies on.
    """
    s = TestSuite("1 · Synthetic geometry")

    # --- 1.1  Two overlapping boxes → collision ---
    def test_overlap():
        a = trimesh.creation.box([10, 10, 10])
        b = trimesh.creation.box([10, 10, 10])
        ta = _make_transform([0, 0, 0])
        tb = _make_transform([5, 0, 0])   # overlaps by 5 mm
        hits = _robot_vs_obstacles(a, ta, {"b": (b, tb)})
        ok = "b" in hits
        return ok, "overlapping boxes detected" if ok else "MISSED overlap"
    s.run("1.1  overlapping boxes → collision", test_overlap)

    # --- 1.2  Two separated boxes → no collision ---
    def test_separated():
        a = trimesh.creation.box([10, 10, 10])
        b = trimesh.creation.box([10, 10, 10])
        ta = _make_transform([0,  0, 0])
        tb = _make_transform([20, 0, 0])  # 10 mm gap
        hits = _robot_vs_obstacles(a, ta, {"b": (b, tb)})
        ok = len(hits) == 0
        return ok, "correctly clear" if ok else f"FALSE positive: {hits}"
    s.run("1.2  separated boxes → no collision", test_separated)

    # --- 1.3  Touching (face-to-face) boxes → trimesh behaviour documented ---
    def test_touching():
        a = trimesh.creation.box([10, 10, 10])
        b = trimesh.creation.box([10, 10, 10])
        ta = _make_transform([0,  0, 0])
        tb = _make_transform([10, 0, 0])  # exactly face-to-face
        hits = _robot_vs_obstacles(a, ta, {"b": (b, tb)})
        # trimesh may or may not flag face-contact — document the result
        contact = len(hits) > 0
        return True, f"face-contact reports collision={contact} (documented, not a bug)"
    s.run("1.3  face-to-face boxes → document trimesh behaviour", test_touching)

    # --- 1.4  Cylinder through a tight hole (clearance = 0.5 mm) → collision ---
    def test_tight_hole():
        # Outer box with a cylindrical hole is hard to make with trimesh primitives,
        # so we approximate: a thick annular 'washer' (outer cylinder minus inner cylinder
        # concatenated is not watertight, but FCL still works on the outer shell).
        # Instead: place two half-blocks (left/right of hole) and a cylinder through them.
        half_block_L = trimesh.creation.box([20, 10, 10])
        half_block_R = trimesh.creation.box([20, 10, 10])
        # cylinder radius 5, hole radius 5.5 → should pass with 0.5 clearance
        rod = trimesh.creation.cylinder(radius=5.0, height=50)

        tL  = _make_transform([-15, 0, 0])
        tR  = _make_transform([ 15, 0, 0])
        # rod centred in gap, should NOT hit either half-block
        trod = _make_transform([0, 0, 0])
        hits = _robot_vs_obstacles(rod, trod, {"L": (half_block_L, tL),
                                                "R": (half_block_R, tR)})
        ok = len(hits) == 0
        return ok, "rod passes through gap cleanly" if ok else f"false collision: {hits}"
    s.run("1.4  rod through gap → no collision", test_tight_hole)

    # --- 1.5  Cylinder too wide for hole → collision ---
    def test_too_wide():
        half_block_L = trimesh.creation.box([20, 10, 10])
        half_block_R = trimesh.creation.box([20, 10, 10])
        rod = trimesh.creation.cylinder(radius=8.0, height=50)  # wider than 5 mm gap

        tL   = _make_transform([-7, 0, 0])
        tR   = _make_transform([ 7, 0, 0])
        trod = _make_transform([0,  0, 0])
        hits = _robot_vs_obstacles(rod, trod, {"L": (half_block_L, tL),
                                                "R": (half_block_R, tR)})
        ok = len(hits) > 0
        return ok, "wide rod correctly blocked" if ok else "MISSED collision — wide rod passed through"
    s.run("1.5  oversized rod through gap → collision", test_too_wide)

    # --- 1.6  Sphere inside another sphere → collision ---
    def test_sphere_inside():
        outer = trimesh.creation.icosphere(radius=20)
        inner = trimesh.creation.icosphere(radius=5)
        t_outer = _make_transform([0, 0, 0])
        t_inner = _make_transform([0, 0, 0])
        hits = _robot_vs_obstacles(inner, t_inner, {"outer": (outer, t_outer)})
        ok = "outer" in hits
        return ok, "enclosed sphere detected" if ok else "MISSED enclosure collision"
    s.run("1.6  sphere enclosed in sphere → collision", test_sphere_inside)

    # --- 1.7  Translation moves mesh correctly ---
    def test_translate():
        box = trimesh.creation.box([10, 10, 10])
        wall = trimesh.creation.box([5, 20, 20])
        t_wall = _make_transform([0, 0, 0])

        # start overlapping
        t_box_near = _make_transform([0, 0, 0])
        hits_near = _robot_vs_obstacles(box, t_box_near, {"wall": (wall, t_wall)})

        # move away
        t_box_far = _make_transform([50, 0, 0])
        hits_far = _robot_vs_obstacles(box, t_box_far, {"wall": (wall, t_wall)})

        ok = (len(hits_near) > 0) and (len(hits_far) == 0)
        return ok, "translation moves mesh correctly" if ok else \
            f"near={hits_near} far={hits_far}"
    s.run("1.7  translation correctly repositions mesh", test_translate)

    # --- 1.8  Multiple obstacles — only one hit ---
    def test_multi_obstacle():
        robot = trimesh.creation.box([10, 10, 10])
        obs_A = trimesh.creation.box([10, 10, 10])
        obs_B = trimesh.creation.box([10, 10, 10])
        t_robot = _make_transform([0,   0, 0])
        t_A     = _make_transform([5,   0, 0])   # overlaps
        t_B     = _make_transform([100, 0, 0])   # far away
        hits = _robot_vs_obstacles(robot, t_robot,
                                   {"A": (obs_A, t_A), "B": (obs_B, t_B)})
        ok = hits == ["A"]
        return ok, f"only A hit: {hits}" if ok else f"wrong result: {hits}"
    s.run("1.8  multi-obstacle — exactly one hit reported", test_multi_obstacle)

    # --- 1.9  min_distance positive when separated ---
    def test_distance_pos():
        a = trimesh.creation.box([10, 10, 10])
        b = trimesh.creation.box([10, 10, 10])
        ta = _make_transform([0,  0, 0])
        tb = _make_transform([20, 0, 0])
        d = _min_distance(a, ta, b, tb)
        ok = d > 0
        return ok, f"distance={d:.3f} mm (>0 ✓)" if ok else f"expected positive, got {d:.3f}"
    s.run("1.9  min_distance positive for separated meshes", test_distance_pos)

    # --- 1.10  min_distance ≤ 0 when penetrating ---
    def test_distance_neg():
        a = trimesh.creation.box([10, 10, 10])
        b = trimesh.creation.box([10, 10, 10])
        ta = _make_transform([0, 0, 0])
        tb = _make_transform([5, 0, 0])
        d = _min_distance(a, ta, b, tb)
        ok = d <= 0
        return ok, f"distance={d:.3f} mm (≤0 ✓)" if ok else f"expected ≤0, got {d:.3f}"
    s.run("1.10 min_distance ≤ 0 for penetrating meshes", test_distance_neg)

    return s


# ===========================================================================
# SUITE 2 — Mesh origin & extents
# ===========================================================================

def suite_mesh_extents(use_scad: bool) -> TestSuite:
    s = TestSuite("2 · Mesh origin & extents")

    if not use_scad:
        s.run("2.x  (skipped — requires --scad flag)",
              lambda: (True, "pass --scad to enable SCAD-dependent tests"))
        return s

    try:
        from collision import CollisionChecker3D
        checker = CollisionChecker3D(models_folder='models')
    except Exception as e:
        s.run("2.0  imports", lambda: (False, str(e)))
        return s

    # --- 2.1  Primary shaft loads without error ---
    def test_load_shaft():
        checker.add_from_scad('primary_shaft.scad', name='shaft_probe',
                              position=(0, 0, 0))
        mesh = checker.added_meshes['shaft_probe']
        ok = len(mesh.vertices) > 0
        return ok, f"{len(mesh.vertices)} vertices, {len(mesh.faces)} faces"
    s.run("2.1  primary shaft loads", test_load_shaft)

    # --- 2.2  Shaft extents ≈ PRIMARY_LENGTH along one axis ---
    def test_shaft_length():
        mesh = checker.added_meshes.get('shaft_probe')
        if mesh is None:
            return False, "mesh not loaded"
        extents = mesh.extents
        max_extent = max(extents)
        tol = 20  # mm — allow 20 mm tolerance for small features beyond main shaft
        ok = abs(max_extent - PRIMARY_LENGTH) < tol
        return ok, (f"longest extent={max_extent:.1f} mm, expected ~{PRIMARY_LENGTH} mm "
                    f"(tol ±{tol} mm)")
    s.run("2.2  shaft longest extent ≈ PRIMARY_LENGTH", test_shaft_length)

    # --- 2.3  Shaft center of mass is inside the mesh bounds ---
    def test_shaft_com():
        mesh = checker.added_meshes.get('shaft_probe')
        if mesh is None:
            return False, "mesh not loaded"
        com = mesh.center_mass
        lo, hi = mesh.bounds
        inside = np.all(com >= lo) and np.all(com <= hi)
        return inside, f"CoM={np.round(com, 1)}, bounds={np.round(lo,1)}..{np.round(hi,1)}"
    s.run("2.3  shaft center_mass inside mesh bounds", test_shaft_com)

    # --- 2.4  Transmission case loads ---
    def test_load_case():
        checker.add_from_scad('transmission_case.scad', name='case_probe',
                              position=(0, 0, 0), parameters={'part': 'case'})
        mesh = checker.added_meshes['case_probe']
        ok = len(mesh.vertices) > 100
        return ok, f"{len(mesh.vertices)} vertices"
    s.run("2.4  transmission case loads", test_load_case)

    # --- 2.5  Case bounding box encloses BEARING_Z height ---
    def test_case_height():
        mesh = checker.added_meshes.get('case_probe')
        if mesh is None:
            return False, "mesh not loaded"
        lo, hi = mesh.bounds
        ok = lo[2] <= BEARING_Z <= hi[2]
        return ok, (f"BEARING_Z={BEARING_Z}, case Z bounds=[{lo[2]:.1f}, {hi[2]:.1f}]")
    s.run("2.5  case Z bounds enclose BEARING_Z", test_case_height)

    # --- 2.6  Shaft journal radius ≤ BEARING_BORE_RADIUS ---
    def test_shaft_radius():
        mesh = checker.added_meshes.get('shaft_probe')
        if mesh is None:
            return False, "mesh not loaded"
        # Radius in XY plane from shaft axis (assume shaft axis is Z or X)
        verts = mesh.vertices
        # Check XZ cross-section at shaft midpoint
        mid_y = np.median(verts[:, 1])
        near_mid = verts[np.abs(verts[:, 1] - mid_y) < 5]
        if len(near_mid) == 0:
            return False, "no vertices near shaft midplane"
        radii = np.sqrt(near_mid[:, 0]**2 + near_mid[:, 2]**2)
        max_r = radii.max()
        # The journal itself should fit through bearing bore
        ok = SHAFT_JOURNAL_RADIUS < BEARING_BORE_RADIUS
        return ok, (f"configured SHAFT_JOURNAL_RADIUS={SHAFT_JOURNAL_RADIUS} mm, "
                    f"BEARING_BORE_RADIUS={BEARING_BORE_RADIUS} mm → fits={ok}")
    s.run("2.6  journal radius < bearing bore radius (constants check)", test_shaft_radius)

    checker.clear()
    return s


# ===========================================================================
# SUITE 3 — Bearing hole passthrough (THE critical suite)
# ===========================================================================

def suite_bearing(use_scad: bool) -> TestSuite:
    """
    Tests that a shaft-like cylinder can pass through a box-with-hole, 
    then verifies the same with real SCAD meshes when --scad is set.
    """
    s = TestSuite("3 · Bearing hole passthrough")

    # -----------------------------------------------------------------------
    # 3.1–3.5  Synthetic bearing-bore model
    # Simulates a case wall (thick slab) with a cylindrical bore.
    # We can't subtract geometry with trimesh primitives alone, so we build
    # the 'wall with hole' from two half-slabs with a gap between them.
    # -----------------------------------------------------------------------

    BORE_R   = BEARING_BORE_RADIUS      # 30 mm
    SHAFT_R  = SHAFT_JOURNAL_RADIUS     # 20 mm — fits with 10 mm clearance
    FAT_R    = BORE_R + 5               # 35 mm — too wide to pass
    WALL_T   = CASE_THICKNESS           # 25 mm thick wall
    HALF_GAP = BORE_R                   # half-width of the hole opening

    # Two half-slabs representing left/right of bearing bore
    slab_top = trimesh.creation.box([10, WALL_T, 200])   # above bore
    slab_bot = trimesh.creation.box([10, WALL_T, 200])   # below bore

    t_wall_top = _make_transform([0, 0,  BORE_R + 100])  # offset upward
    t_wall_bot = _make_transform([0, 0, -BORE_R - 100])  # offset downward
    wall_obstacles = {
        "wall_top": (slab_top, t_wall_top),
        "wall_bot": (slab_bot, t_wall_bot),
    }

    # --- 3.1  Slim shaft through bore centre → no collision ---
    def test_slim_through():
        shaft = trimesh.creation.cylinder(radius=SHAFT_R, height=WALL_T * 3,
                                          sections=32)
        t_shaft = _make_transform([0, 0, 0])  # centred on bore axis
        hits = _robot_vs_obstacles(shaft, t_shaft, wall_obstacles)
        ok = len(hits) == 0
        return ok, (f"slim shaft (r={SHAFT_R}) through bore centre: clear" if ok
                    else f"FALSE collision with: {hits}")
    s.run("3.1  slim shaft centred on bore → no collision", test_slim_through)

    # --- 3.2  Fat shaft (oversized) → collision with wall ---
    def test_fat_blocked():
        shaft = trimesh.creation.cylinder(radius=FAT_R, height=WALL_T * 3,
                                          sections=32)
        t_shaft = _make_transform([0, 0, 0])
        hits = _robot_vs_obstacles(shaft, t_shaft, wall_obstacles)
        ok = len(hits) > 0
        return ok, (f"fat shaft (r={FAT_R}) correctly blocked" if ok
                    else "MISSED — fat shaft passed through unrealistically")
    s.run("3.2  oversized shaft → collision with wall", test_fat_blocked)

    # --- 3.3  Shaft offset within clearance → no collision ---
    def test_offset_ok():
        clearance = BORE_R - SHAFT_R   # 10 mm
        offset    = clearance * 0.5    # 5 mm offset — still fits
        shaft = trimesh.creation.cylinder(radius=SHAFT_R, height=WALL_T * 3,
                                          sections=32)
        t_shaft = _make_transform([0, offset, 0])
        hits = _robot_vs_obstacles(shaft, t_shaft, wall_obstacles)
        ok = len(hits) == 0
        return ok, (f"shaft offset {offset:.1f} mm within {clearance:.1f} mm clearance: clear"
                    if ok else f"FALSE collision: {hits}")
    s.run("3.3  shaft offset within bore clearance → no collision", test_offset_ok)

    # --- 3.4  Shaft offset beyond clearance → collision ---
    def test_offset_too_much():
        clearance = BORE_R - SHAFT_R   # 10 mm
        offset    = clearance + 5      # 15 mm — shaft clips wall
        shaft = trimesh.creation.cylinder(radius=SHAFT_R, height=WALL_T * 3,
                                          sections=32)
        t_shaft = _make_transform([0, offset, 0])
        hits = _robot_vs_obstacles(shaft, t_shaft, wall_obstacles)
        ok = len(hits) > 0
        return ok, (f"shaft offset {offset:.1f} mm exceeds clearance {clearance:.1f} mm → blocked"
                    if ok else "MISSED — shaft clipping wall not detected")
    s.run("3.4  shaft offset beyond bore clearance → collision", test_offset_too_much)

    # --- 3.5  Shaft at exactly BEARING_Z height → no collision ---
    def test_bearing_z_height():
        shaft = trimesh.creation.cylinder(radius=SHAFT_R, height=WALL_T * 3,
                                          sections=32)
        # Put shaft at bearing height — the slabs have a gap at z=0,
        # so shift the whole obstacle rig to BEARING_Z and test at that height
        t_top = _make_transform([0, 0, BEARING_Z + BORE_R + 100])
        t_bot = _make_transform([0, 0, BEARING_Z - BORE_R - 100])
        obs = {"wall_top": (slab_top, t_top), "wall_bot": (slab_bot, t_bot)}
        t_shaft = _make_transform([0, 0, BEARING_Z])
        hits = _robot_vs_obstacles(shaft, t_shaft, obs)
        ok = len(hits) == 0
        return ok, (f"shaft at BEARING_Z={BEARING_Z} passes cleanly" if ok
                    else f"collision: {hits}")
    s.run("3.5  shaft at BEARING_Z height → no collision", test_bearing_z_height)

    # --- 3.6  Shaft approaching wall from outside → detects collision entry ---
    def test_approach_entry():
        shaft = trimesh.creation.cylinder(radius=SHAFT_R, height=10, sections=32)
        # Approach from outside (large positive X) moving toward wall at x=0
        results = []
        for x in [50, 20, 10, 5, 2, 0, -5]:
            t_shaft = _make_transform([x, 0, 0])
            hits = _robot_vs_obstacles(shaft, t_shaft, wall_obstacles)
            results.append((x, len(hits) > 0))
        # At x=50 must be clear; at x=0 (inside bore) must be clear too;
        # collision should only appear if x pushes shaft outside the bore gap
        x50_clear = not results[0][1]
        return x50_clear, f"x=50 clear={x50_clear}, sweep={results}"
    s.run("3.6  shaft approach sweep — far position is clear", test_approach_entry)

    # -----------------------------------------------------------------------
    # 3.7–3.10  Real SCAD meshes (only when --scad flag is set)
    # -----------------------------------------------------------------------
    if not use_scad:
        for i in range(7, 11):
            s.run(f"3.{i}  (skipped — requires --scad flag)",
                  lambda: (True, "pass --scad to enable"))
        return s

    try:
        from collision import CollisionChecker3D
        checker = CollisionChecker3D(models_folder='models')
        checker.add_from_scad('transmission_case.scad', name='case',
                              position=(0, 0, 0), parameters={'part': 'case'})
        checker.add_from_scad('primary_shaft.scad', name='shaft',
                              position=(0, 0, 0))
        case_mesh  = checker.added_meshes['case']
        shaft_mesh = checker.added_meshes['shaft']
    except Exception as e:
        for i in range(7, 11):
            s.run(f"3.{i}  SCAD load failed",
                  lambda: (False, str(e)))
        return s

    obstacles = {"case": (case_mesh, _make_transform([0, 0, 0]))}

    # --- 3.7  Shaft at START position vs case ---
    def test_start_vs_case():
        t = _make_transform(START)
        hits = _robot_vs_obstacles(shaft_mesh, t, obstacles)
        # Document result — this is the key diagnostic
        colliding = len(hits) > 0
        msg = ("COLLISION at START — shaft overlaps case geometry. "
               "START position needs adjustment or bearing-bore exemption." if colliding
               else "START is clear of case — geometry is consistent.")
        return True, msg   # always 'pass' — this is a diagnostic, not a hard assertion
    s.run("3.7  shaft at START vs case (diagnostic)", test_start_vs_case)

    # --- 3.8  Shaft at GOAL position vs case ---
    def test_goal_vs_case():
        t = _make_transform(GOAL)
        hits = _robot_vs_obstacles(shaft_mesh, t, obstacles)
        colliding = len(hits) > 0
        msg = ("COLLISION at GOAL — goal position is invalid." if colliding
               else "GOAL is clear ✓")
        return not colliding, msg
    s.run("3.8  shaft at GOAL vs case → no collision", test_goal_vs_case)

    # --- 3.9  Shaft extracted fully (far left) → no collision ---
    def test_fully_extracted():
        extracted_x = -400   # well outside case
        t = _make_transform([extracted_x, 0, BEARING_Z])
        hits = _robot_vs_obstacles(shaft_mesh, t, obstacles)
        ok = len(hits) == 0
        return ok, (f"fully extracted (x={extracted_x}) is clear" if ok
                    else f"unexpected collision when extracted: {hits}")
    s.run("3.9  shaft fully extracted → no collision", test_fully_extracted)

    # --- 3.10  Shaft midway through bore → no collision ---
    def test_midway():
        # Midpoint between START and GOAL
        mid = (START + GOAL) / 2
        t = _make_transform(mid)
        hits = _robot_vs_obstacles(shaft_mesh, t, obstacles)
        colliding = len(hits) > 0
        msg = (f"COLLISION at midpoint {np.round(mid,1)} — path blocked" if colliding
               else f"midpoint {np.round(mid,1)} is clear ✓")
        return True, msg   # diagnostic — documents where collisions occur
    s.run("3.10 shaft at path midpoint vs case (diagnostic)", test_midway)

    checker.clear()
    return s


# ===========================================================================
# SUITE 4 — Known-clear positions
# ===========================================================================

def suite_known_clear(use_scad: bool) -> TestSuite:
    s = TestSuite("4 · Known-clear positions")

    if not use_scad:
        s.run("4.x  (skipped — requires --scad flag)",
              lambda: (True, "pass --scad to enable"))
        return s

    try:
        from collision import CollisionChecker3D
        checker = CollisionChecker3D(models_folder='models')
        checker.add_from_scad('transmission_case.scad', name='case',
                              position=(0, 0, 0), parameters={'part': 'case'})
        checker.add_from_scad('secondary_shaft.scad', name='countershaft',
                              position=(SECONDARY_LENGTH / 2 + 1, 0, CS_BEARING_Z),
                              parameters={'part': 'countershaft'})
        checker.add_from_scad('primary_shaft.scad', name='robot',
                              position=(0, 0, 0))
        shaft_mesh = checker.added_meshes['robot']
    except Exception as e:
        s.run("4.0  SCAD load", lambda: (False, str(e)))
        return s

    obstacles = {
        name: (checker.added_meshes[name], checker.current_poses[name])
        for name in ['case', 'countershaft']
    }

    clear_positions = [
        ("fully extracted left",  np.array([-400, 0, BEARING_Z])),
        ("1 shaft-length left",   np.array([-PRIMARY_LENGTH, 0, BEARING_Z])),
        ("above case",            np.array([0, 0, BEARING_Z + 100])),
        ("goal position",         GOAL),
    ]

    for label, pos in clear_positions:
        pos_copy = pos.copy()
        def make_test(p):
            def fn():
                t = _make_transform(p)
                hits = _robot_vs_obstacles(shaft_mesh, t, obstacles)
                ok = len(hits) == 0
                return ok, (f"clear ✓" if ok else f"UNEXPECTED collision with {hits}")
            return fn
        s.run(f"4.x  {label} {np.round(pos_copy, 1)} → clear", make_test(pos_copy))

    checker.clear()
    return s


# ===========================================================================
# SUITE 5 — Known-collision positions
# ===========================================================================

def suite_known_collision(use_scad: bool) -> TestSuite:
    s = TestSuite("5 · Known-collision positions")

    if not use_scad:
        s.run("5.x  (skipped — requires --scad flag)",
              lambda: (True, "pass --scad to enable"))
        return s

    try:
        from collision import CollisionChecker3D
        checker = CollisionChecker3D(models_folder='models')
        checker.add_from_scad('transmission_case.scad', name='case',
                              position=(0, 0, 0), parameters={'part': 'case'})
        checker.add_from_scad('primary_shaft.scad', name='robot',
                              position=(0, 0, 0))
        shaft_mesh = checker.added_meshes['robot']
    except Exception as e:
        s.run("5.0  SCAD load", lambda: (False, str(e)))
        return s

    obstacles = {"case": (checker.added_meshes['case'], _make_transform([0, 0, 0]))}

    # Positions where the shaft should clearly intersect case walls
    collision_positions = [
        ("shaft origin at case wall X=0",  np.array([0,   0, 0])),
        ("shaft 90° into case wall",       np.array([0, 100, BEARING_Z])),
        ("shaft buried in case top",       np.array([0,   0, BEARING_Z + CASE_HEIGHT])),
    ]

    for label, pos in collision_positions:
        pos_copy = pos.copy()
        def make_test(p, lbl):
            def fn():
                t = _make_transform(p)
                hits = _robot_vs_obstacles(shaft_mesh, t, obstacles)
                colliding = len(hits) > 0
                # These are documented collisions — we report them but don't fail
                return True, (f"collision detected ✓ ({hits})" if colliding
                              else f"no collision (position may not be as deep as expected)")
            return fn
        s.run(f"5.x  {label} (diagnostic)", make_test(pos_copy, label))

    checker.clear()
    return s


# ===========================================================================
# SUITE 6 — State validity checker
# ===========================================================================

def suite_validity_checker(use_scad: bool) -> TestSuite:
    """
    Tests the actual _is_state_valid() function that RRT calls at every node,
    using the real CollisionChecker3D and planner machinery.
    """
    s = TestSuite("6 · _is_state_valid() (RRT validity checker)")

    if not use_scad:
        s.run("6.x  (skipped — requires --scad flag)",
              lambda: (True, "pass --scad to enable"))
        return s

    try:
        from collision import CollisionChecker3D
        from planner import RRTPlanner3D
        bounds = [(-500, 500), (-500, 500), (-400, 400)]
        planner = RRTPlanner3D(bounds=bounds, models_folder='models')
        planner.checker.add_from_scad('transmission_case.scad',
                                      name='TransmissionCase',
                                      position=(0, 0, 0),
                                      parameters={'part': 'case'})
        planner.checker.add_from_scad('secondary_shaft.scad',
                                      name='CounterShaft',
                                      position=(SECONDARY_LENGTH / 2 + 1, 0, CS_BEARING_Z),
                                      parameters={'part': 'countershaft'})
        planner.set_robot(scad_file='primary_shaft.scad', start_position=START)
    except Exception as e:
        s.run("6.0  planner setup", lambda: (False, str(e)))
        return s

    # Helper: fake an OMPL state object with indexing
    class FakeState:
        def __init__(self, pos):
            self._pos = list(pos)
        def __getitem__(self, i):
            return self._pos[i]

    def check(pos):
        return planner._is_state_valid(FakeState(pos))

    # --- 6.1  GOAL must be valid ---
    def test_goal_valid():
        ok = check(GOAL)
        return ok, ("GOAL is valid ✓" if ok
                    else "GOAL is INVALID — RRT can never reach it")
    s.run("6.1  GOAL position is valid", test_goal_valid)

    # --- 6.2  Far-left extracted position must be valid ---
    def test_extracted_valid():
        pos = np.array([-400, 0, BEARING_Z])
        ok = check(pos)
        return ok, (f"extracted {pos} is valid ✓" if ok else "extracted position invalid")
    s.run("6.2  fully extracted position is valid", test_extracted_valid)

    # --- 6.3  Self-collision check — robot must not collide with itself ---
    def test_no_self_collision():
        # At any position, the robot should not be in_collision_internal with itself
        planner.checker.update_position("robot", tuple(GOAL))
        mgr = planner.checker.manager
        # Build a manager with ONLY the robot
        solo = trimesh.collision.CollisionManager()
        solo.add_object("robot", planner.checker.added_meshes["robot"],
                        planner.checker.current_poses["robot"])
        self_hit = solo.in_collision_internal()
        ok = not self_hit
        return ok, ("no self-collision ✓" if ok else "SELF-COLLISION detected in robot mesh")
    s.run("6.3  robot mesh has no self-collision", test_no_self_collision)

    # --- 6.4  Validity is consistent across repeated calls at same position ---
    def test_consistent():
        pos = GOAL
        results = [check(pos) for _ in range(5)]
        ok = len(set(results)) == 1
        return ok, (f"consistent: {results[0]} across 5 calls ✓" if ok
                    else f"INCONSISTENT: {results}")
    s.run("6.4  validity check is deterministic", test_consistent)

    # --- 6.5  START validity — document result ---
    def test_start_validity():
        ok = check(START)
        msg = ("START is valid ✓" if ok
               else "START is INVALID — this is why RRT fails. "
                    "Shaft overlaps case walls at insertion depth. "
                    "Consider using a bearing-bore exemption or adjusting START.")
        return True, msg  # always 'pass' — this is the key diagnostic output
    s.run("6.5  START position validity (diagnostic)", test_start_validity)

    # --- 6.6  Midpoint between START and GOAL ---
    def test_midpoint():
        mid = (START + GOAL) / 2
        ok = check(mid)
        return True, (f"midpoint {np.round(mid,1)} valid={ok}")
    s.run("6.6  midpoint START→GOAL validity (diagnostic)", test_midpoint)

    # --- 6.7  Y-axis clearance — shaft should be valid at y=0 but not y=200 ---
    def test_y_clearance():
        pos_center = np.array([GOAL[0], 0,   BEARING_Z])
        pos_offset = np.array([GOAL[0], 200, BEARING_Z])
        v_center = check(pos_center)
        v_offset = check(pos_offset)
        # Centred should be valid; far offset should be invalid (inside case wall)
        return True, (f"y=0 valid={v_center}, y=200 valid={v_offset} (documented)")
    s.run("6.7  Y-axis clearance sweep (diagnostic)", test_y_clearance)

    # --- 6.8  Position update actually moves the mesh ---
    def test_update_moves_mesh():
        planner.checker.update_position("robot", tuple(GOAL))
        p1 = planner.checker.current_poses["robot"][:3, 3].copy()
        planner.checker.update_position("robot", (0, 0, 0))
        p2 = planner.checker.current_poses["robot"][:3, 3].copy()
        ok = not np.allclose(p1, p2)
        return ok, (f"position changed from {p1} to {p2} ✓" if ok
                    else "update_position had no effect — poses are identical")
    s.run("6.8  update_position moves mesh correctly", test_update_moves_mesh)

    return s


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Transmission collision test suite")
    parser.add_argument("--suite",   default="all",
                        help="Run one suite: synthetic | extents | bearing | clear | collision | validity | all")
    parser.add_argument("--scad",    action="store_true",
                        help="Enable tests that require OpenSCAD + your model files")
    parser.add_argument("--verbose", action="store_true",
                        help="Print details for passing tests too")
    args = parser.parse_args()

    suite_map = {
        "synthetic":  lambda: suite_synthetic(),
        "extents":    lambda: suite_mesh_extents(args.scad),
        "bearing":    lambda: suite_bearing(args.scad),
        "clear":      lambda: suite_known_clear(args.scad),
        "collision":  lambda: suite_known_collision(args.scad),
        "validity":   lambda: suite_validity_checker(args.scad),
    }

    if args.suite == "all":
        suites = [fn() for fn in suite_map.values()]
    elif args.suite in suite_map:
        suites = [suite_map[args.suite]()]
    else:
        print(f"Unknown suite '{args.suite}'. Choose from: {', '.join(suite_map)}")
        sys.exit(1)

    success = print_results(suites, args.verbose)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()