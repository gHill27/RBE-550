#!/usr/bin/env python3
"""
collision_checker.py - 3D collision detection for meshes
"""

import numpy as np
import trimesh
from typing import List, Tuple, Optional, Dict
from pathlib import Path
from mesh_gen import MeshGenerator

class CollisionChecker3D:
    """3D collision detection between multiple meshes"""
    def __init__(self, models_folder: str = 'models'):
        self.mesh_generator = MeshGenerator(models_folder=models_folder)
        # The manager handles all the narrow-phase geometry logic
        self.manager = trimesh.collision.CollisionManager()
        self.names = []
        self.meshes = []
        self.added_meshes = {}
        self.current_poses = {}

    def add_mesh(self, mesh: trimesh.Trimesh, name: str = None, 
                 position: Tuple[float, float, float] = (0, 0, 0)):
        """Add a mesh and register it with the CollisionManager"""
        name = name or f"Mesh_{len(self.names)}"
        
        # Create a 4x4 transformation matrix for the manager
        transform = np.eye(4)
        transform[:3, 3] = position
        
        # Add to the manager (this builds the internal BVH tree)
        self.manager.add_object(name, mesh, transform=transform)
        
        self.meshes.append(mesh)
        self.names.append(name)
        self.added_meshes[name] = mesh
        self.current_poses[name] = transform
        # print(f"✓ Added '{name}' to CollisionManager at {position}")

    def update_position(self, name: str, position: Tuple[float, float, float]):
        """Efficiently move an existing mesh without re-adding it"""
        transform = np.eye(4)
        transform[:3, 3] = position
        self.manager.set_transform(name, transform)
        self.current_poses[name] = transform

    
    def check_mesh_against_manager(self, mesh: trimesh.Trimesh,
                                   transform: np.ndarray) -> bool:
        """
        Check if a single mesh (e.g., robot) collides with any object
        currently in the manager (static obstacles).
        Returns True if collision detected.
        """
        return self.manager.in_collision_single(mesh, transform)

    def min_distance_to_environment(self, mesh: trimesh.Trimesh,
                                    transform: np.ndarray) -> float:
        """
        Return the minimum distance between the given mesh (at `transform`)
        and any object in the manager. Positive = separated, 0 = touching,
        negative = penetration depth.
        """
        return self.manager.min_distance_single(mesh, transform)


    def check_collision(self, name1: str, name2: str) -> bool:
        mesh1 = self.added_meshes[name1]
        transform1 = self.current_poses[name1]
        mesh2 = self.added_meshes[name2]
        transform2 = self.current_poses[name2]
        temp = trimesh.collision.CollisionManager()
        temp.add_object(name2, mesh2, transform2)
        return temp.in_collision_single(mesh1, transform1)
        

    def check_all_collisions(self) -> set:
        """
        Return a set of unordered pairs (tuple) that are colliding.
        Example: {('robot', 'Case'), ('shaft', 'bearing')}
        """
        colliding_pairs = set()
        names = list(self.added_meshes.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if self.check_collision(names[i], names[j]):
                    # store as sorted tuple to avoid duplicates
                    colliding_pairs.add(tuple(sorted((names[i], names[j]))))
        return colliding_pairs

    def get_collision_details(self, name1: str, name2: str) -> dict:
        """Return distance and collision status for the given pair."""
        is_colliding = self.check_collision(name1, name2)
        # distance is positive if not colliding, 0.0 if touching, negative if interpenetrating
        # Compute exact distance using trimesh's mesh.min_distance
        mesh1 = self.added_meshes[name1]
        transform1 = self.current_poses[name1]
        mesh2 = self.added_meshes[name2]
        transform2 = self.current_poses[name2]
        # min_distance accepts transforms for both meshes
        distance = mesh1.min_distance(mesh2, transform=transform1,
                                      transform_other=transform2)
        return {
            'collision': is_colliding,
            'distance': distance,
            'meshes': (name1, name2)
        }
    
    def add_from_scad(self, scad_file: str, name: str = None,
                      position: Tuple[float, float, float] = (0, 0, 0),
                      parameters: Dict = None,
                      fix_mesh: bool = False):
        """Standard wrapper to load SCAD via MeshGenerator"""
        try:
            mesh = self.mesh_generator.from_scad(
                scad_file, 
                parameters=parameters, 
                fix_mesh=fix_mesh
            )
            self.add_mesh(mesh, name or Path(scad_file).stem, position)
        except Exception as e:
            print(f"❌ Failed to load {scad_file}: {e}")
            raise
    
    def visualize(self, robot_mesh: trimesh.Trimesh = None,
              robot_transform: np.ndarray = None):
        scene = trimesh.Scene()
        for name in self.names:
            mesh = self.added_meshes[name]
            transform = self.current_poses[name]
            scene.add_geometry(mesh, node_name=name, transform=transform)
        if robot_mesh is not None and robot_transform is not None:
            scene.add_geometry(robot_mesh, node_name="robot", transform=robot_transform)
        scene.show()
        
    def clear(self):
        self.manager = trimesh.collision.CollisionManager()
        self.names.clear()
        self.meshes.clear()
        self.added_meshes.clear()   
        self.current_poses.clear()  