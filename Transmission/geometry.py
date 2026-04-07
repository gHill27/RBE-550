# geometry.py
import numpy as np

class Cylinder:
    """A cylinder defined in local frame — axis along Z."""
    def __init__(self, radius, height, local_offset=(0,0,0)):
        self.radius = radius
        self.height = height
        self.offset = np.array(local_offset, dtype=float)

class Box:
    """An axis-aligned box — used for case walls."""
    def __init__(self, min_corner, max_corner):
        self.min = np.array(min_corner, dtype=float)
        self.max = np.array(max_corner, dtype=float)

def build_mainshaft():
    """Build mainshaft as a list of cylinders."""
    segments = [
        (30, 40, -180),   # rear journal
        (45, 25, -130),   # reverse gear
        (50, 30, -80),    # 1st gear
        (50, 30, -30),    # 2nd gear
        (45, 25, 20),     # 3rd gear
        (40, 25, 70),     # 4th gear
        (35, 30, 120),    # 5th gear
        (30, 40, 175),    # output journal
    ]
    return [Cylinder(r, h, (0, 0, z)) for r, h, z in segments]

def build_countershaft():
    """Build countershaft as a list of cylinders."""
    segments = [
        (40, 30, -160),   # input gear
        (55, 25, -110),   # reverse gear
        (60, 30, -60),    # 1st gear
        (60, 30, -10),    # 2nd gear
        (55, 25, 40),     # 3rd gear
        (50, 30, 90),     # 4th gear
        (35, 40, 140),    # output journal
    ]
    return [Cylinder(r, h, (0, 0, z)) for r, h, z in segments]

def build_case():
    """Return list of Box objects representing case interior."""
    # Interior bounds
    lx, ly, lz = 260, 190, 280
    
    walls = []
    
    # Six walls as bounding planes
    walls.append(Box(np.array([-lx/2, -ly/2, -lz/2]), np.array([lx/2, -ly/2 + 10, lz/2])))  # bottom
    walls.append(Box(np.array([-lx/2, ly/2 - 10, -lz/2]), np.array([lx/2, ly/2, lz/2])))     # top
    walls.append(Box(np.array([-lx/2, -ly/2, -lz/2]), np.array([-lx/2 + 10, ly/2, lz/2])))   # left
    walls.append(Box(np.array([lx/2 - 10, -ly/2, -lz/2]), np.array([lx/2, ly/2, lz/2])))     # right
    walls.append(Box(np.array([-lx/2, -ly/2, -lz/2]), np.array([lx/2, ly/2, -lz/2 + 10])))   # front
    walls.append(Box(np.array([-lx/2, -ly/2, lz/2 - 10]), np.array([lx/2, ly/2, lz/2])))     # back
    
    return walls

def get_countershaft_pose():
    """
    Return fixed pose for countershaft.
    Positioned below mainshaft with proper gear clearance.
    """
    # Place countershaft 130mm below mainshaft center
    # This provides 20mm clearance beyond the 110mm gear radius sum
    return np.array([0.0, -130.0, 0.0, 1.0, 0.0, 0.0, 0.0])