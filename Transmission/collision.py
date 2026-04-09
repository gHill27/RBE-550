#!/usr/bin/env python3
"""
collision_checker.py - 3D collision detection for meshes
"""

import numpy as np
import trimesh
from typing import List, Tuple, Optional, Dict, Union
from pathlib import Path
from mesh_gen import MeshGenerator

class CollisionChecker3D:
    """3D collision detection between multiple meshes"""
    
    def __init__(self):
        self.meshes = []
        self.names = []
        self.positions = []  # (x, y, z) positions
        self.mesh_generator = MeshGenerator()
    
    def add_mesh(self, mesh: trimesh.Trimesh, name: str = None, 
                 position: Tuple[float, float, float] = (0, 0, 0)):
        """Add a mesh at specified position"""
        self.meshes.append(mesh)
        self.names.append(name or f"Mesh_{len(self.meshes)}")
        self.positions.append(np.array(position))
        print(f"✓ Added '{self.names[-1]}' at {position}")
    
    def add_from_scad(self, scad_file: Union[str, Path], name: str = None,
                      position: Tuple[float, float, float] = (0, 0, 0),
                      parameters: Dict = None):
        """Load and add mesh from OpenSCAD file"""
        mesh = self.mesh_generator.from_scad(scad_file, parameters=parameters)
        self.add_mesh(mesh, name or Path(scad_file).stem, position)
    
    def get_mesh_at_position(self, index: int) -> trimesh.Trimesh:
        """Get mesh transformed to its position"""
        mesh = self.meshes[index].copy()
        mesh.apply_translation(self.positions[index])
        return mesh
    
    def check_collision(self, index1: int, index2: int) -> bool:
        """Check if two meshes collide"""
        mesh1 = self.get_mesh_at_position(index1)
        mesh2 = self.get_mesh_at_position(index2)
        return mesh1.intersects_mesh(mesh2)
    
    def check_all_collisions(self) -> List[Tuple[int, int]]:
        """Check all pairs for collisions"""
        collisions = []
        n = len(self.meshes)
        
        for i in range(n):
            for j in range(i + 1, n):
                if self.check_collision(i, j):
                    collisions.append((i, j))
        
        return collisions
    
    def get_collision_details(self, index1: int, index2: int) -> Dict:
        """Get detailed collision information"""
        mesh1 = self.get_mesh_at_position(index1)
        mesh2 = self.get_mesh_at_position(index2)
        
        intersects = mesh1.intersects_mesh(mesh2)
        
        if intersects:
            try:
                intersection = mesh1.intersection(mesh2)
                intersection_volume = intersection.volume if hasattr(intersection, 'volume') else 0
                penetration = -mesh1.min_distance(mesh2)
            except:
                intersection_volume = 0
                penetration = 0
            
            return {
                'collision': True,
                'penetration_depth': penetration,
                'intersection_volume': intersection_volume,
                'meshes': (self.names[index1], self.names[index2])
            }
        else:
            distance = mesh1.min_distance(mesh2)
            return {
                'collision': False,
                'distance': distance,
                'meshes': (self.names[index1], self.names[index2])
            }
    
    def distance_between(self, index1: int, index2: int) -> float:
        """Get minimum distance between two meshes (negative if intersecting)"""
        mesh1 = self.get_mesh_at_position(index1)
        mesh2 = self.get_mesh_at_position(index2)
        
        if mesh1.intersects_mesh(mesh2):
            return -mesh1.min_distance(mesh2)
        else:
            return mesh1.min_distance(mesh2)
    
    def is_point_inside(self, mesh_index: int, point: Tuple[float, float, float]) -> bool:
        """Check if point is inside mesh"""
        mesh = self.get_mesh_at_position(mesh_index)
        return mesh.contains([point])[0]
    
    def visualize(self):
        """Visualize all meshes in 3D"""
        scene = trimesh.Scene()
        colors = [
            [100, 100, 255, 180],  # Blue
            [255, 100, 100, 180],  # Red
            [100, 255, 100, 180],  # Green
            [255, 255, 100, 180],  # Yellow
        ]
        
        for i, mesh in enumerate(self.meshes):
            mesh_copy = mesh.copy()
            mesh_copy.apply_translation(self.positions[i])
            mesh_copy.visual.face_colors = colors[i % len(colors)]
            scene.add_geometry(mesh_copy, node_name=self.names[i])
        
        scene.show()
    
    def print_report(self):
        """Print collision report"""
        print("\n" + "="*60)
        print("3D COLLISION REPORT")
        print("="*60)
        
        print("\n📦 OBJECTS:")
        for i, name in enumerate(self.names):
            print(f"  {i}: {name} at {self.positions[i]}")
        
        collisions = self.check_all_collisions()
        
        if not collisions:
            print("\n✅ NO COLLISIONS")
            print("\n📏 DISTANCES:")
            n = len(self.meshes)
            for i in range(n):
                for j in range(i + 1, n):
                    dist = self.distance_between(i, j)
                    print(f"  {self.names[i]} ↔ {self.names[j]}: {dist:.3f}")
        else:
            print(f"\n❌ {len(collisions)} COLLISION(S):")
            for i, j in collisions:
                details = self.get_collision_details(i, j)
                print(f"\n  🔴 {self.names[i]} ↔ {self.names[j]}")
                print(f"     Penetration: {details['penetration_depth']:.3f}")
                print(f"     Intersection volume: {details['intersection_volume']:.3f}")
    
    def clear(self):
        """Clear all meshes"""
        self.meshes.clear()
        self.names.clear()
        self.positions.clear()