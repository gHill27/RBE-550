#!/usr/bin/env python3
# =============================================================================
# mesh_gen.py
# Worcester Polytechnic Institute — RBE-550 Motion Planning
# OpenSCAD-to-Trimesh Mesh Generator
# =============================================================================
# Authors:     Gavin Hill
# Course:      RBE-550 Motion Planning
# Instructor:  Daniel Montrallo Flickinger, PhD
#
# AI Assistance Disclosure:
#   Portions of this file were developed with the assistance of Claude
#   (Anthropic, claude.ai), an AI language model. AI assistance was used for:
#     - from_scad_simplified() method design using OpenSCAD -D flag injection
#       to pass simplified=true and $fn overrides at render time
#     - Mesh cache implementation to avoid redundant OpenSCAD subprocess calls
#     - Scene concatenation handling for multi-body SCAD exports
#   All AI-generated suggestions were reviewed, tested, and validated by
#   the author. Final implementation decisions remain the author's own.
# =============================================================================

import subprocess
import trimesh
import trimesh.repair
import tempfile
from pathlib import Path
from typing import Optional, Dict


class MeshGenerator:
    """Generate 3D trimesh objects from OpenSCAD files"""

    def __init__(self, models_folder: str = 'models'):
        """
        Initialize mesh generator.

        Args:
            models_folder: Folder containing OpenSCAD models
        """
        self.models_folder = Path(models_folder)
        self.openscad_bin: str = ''

        if not self.models_folder.exists():
            self.models_folder = Path.cwd()
            print(f"⚠️  Models folder not found, using current directory: {self.models_folder}")
        else:
            print(f"✓ Found models folder at: {self.models_folder}")

        self._check_openscad()

    def _check_openscad(self):
        """Verify OpenSCAD is installed, preferring the nightly build."""
        for binary in ['openscad-nightly', 'openscad']:
            try:
                result = subprocess.run(
                    [binary, '--version'], capture_output=True, text=True, check=True
                )
                self.openscad_bin = binary
                version = result.stderr.strip() or result.stdout.strip()
                print(f"✓ OpenSCAD found: {binary} ({version})")
                return
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
        raise RuntimeError(
            "OpenSCAD not found. Install via: sudo apt install openscad-nightly"
        )

    def from_scad(
        self,
        scad_file: str,
        parameters: Optional[Dict] = None,
        scale_factor: float = 1.0,
        fix_mesh: bool = False,
    ) -> trimesh.Trimesh:
        """
        Convert an OpenSCAD file to a 3D trimesh.

        Args:
            scad_file:     Filename of the .scad file (relative to models_folder).
            parameters:    Optional dict of OpenSCAD variables to override via -D flags,
                           e.g. {'bearing_radius': 45, 'case_height': 320}.
            scale_factor:  Uniform scale applied after loading (default 1.0 = no change).
            fix_mesh:      If True, attempt to repair normals and fill holes so the mesh
                           becomes watertight (enables volume calculation).

        Returns:
            A trimesh.Trimesh object.
        """
        scad_path = self.models_folder / scad_file
        if not scad_path.exists():
            raise FileNotFoundError(f"File not found: {scad_path}")

        print(f"📄 Loading: {scad_path}")

        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp:
            stl_file = tmp.name

        try:
            # Build the OpenSCAD command, injecting any parameter overrides
            cmd = [self.openscad_bin, '-o', stl_file, str(scad_path)]
            if parameters:
                for key, val in parameters.items():
                    cmd += ['-D', f'{key}={val}']

            result = subprocess.run(cmd, capture_output=True, text=True)

            # Surface non-fatal warnings even on success
            if result.stderr.strip():
                label = "OpenSCAD stderr" if result.returncode != 0 else "OpenSCAD warnings"
                # print(f"⚠️  {label}:\n{result.stderr.strip()}")

            if result.returncode != 0:
                raise RuntimeError("OpenSCAD export failed (see stderr above)")

            # Load mesh — guard against Scene results (multi-body exports)
            loaded = trimesh.load(stl_file)
            if isinstance(loaded, trimesh.Scene):
                print("ℹ️  Scene detected, concatenating all bodies into one mesh")
                mesh = trimesh.util.concatenate(loaded.dump())
            else:
                mesh = loaded

            if len(mesh.vertices) == 0:
                raise RuntimeError("Mesh has 0 vertices — SCAD file may have errors")

            # Optionally attempt mesh repair
            if fix_mesh:
                print("🔧 Attempting mesh repair...")
                trimesh.repair.fix_normals(mesh)
                trimesh.repair.fill_holes(mesh)
                if mesh.is_watertight:
                    print("✓ Mesh is now watertight")
                else:
                    print("⚠️  Mesh could not be made fully watertight")

            if scale_factor != 1.0:
                mesh.apply_scale(scale_factor)
                print(f"✓ Scale factor {scale_factor} applied")

            return mesh

        finally:
            Path(stl_file).unlink(missing_ok=True)

    def get_mesh_info(self, mesh: trimesh.Trimesh) -> Dict:
        """Get detailed information about 3D mesh"""
        try:
            # Handle volume - will be None for non-watertight meshes
            volume = mesh.volume if hasattr(mesh, 'volume') and mesh.volume is not None else 0.0
            if volume is None:
                volume = 0.0
                
            # Handle bounds
            bounds = mesh.bounds.tolist() if mesh.bounds is not None else [[0,0,0], [0,0,0]]
            
            # Handle extents
            extents = mesh.extents.tolist() if mesh.extents is not None else [0,0,0]
            
            # Handle center of mass
            center_mass = mesh.center_mass.tolist() if hasattr(mesh, 'center_mass') and mesh.center_mass is not None else [0,0,0]
            
            return {
                'vertices': len(mesh.vertices),
                'faces': len(mesh.faces),
                'edges': len(mesh.edges),
                'volume': volume,
                'surface_area': mesh.area if hasattr(mesh, 'area') else 0.0,
                'is_watertight': mesh.is_watertight if hasattr(mesh, 'is_watertight') else False,
                'is_convex': mesh.is_convex if hasattr(mesh, 'is_convex') else False,
                'bounds': bounds,
                'extents': extents,
                'center_of_mass': center_mass,
            }
        except Exception as e:
            print(f"Warning: Could not get mesh info: {e}")
            return {
                'vertices': len(mesh.vertices),
                'faces': len(mesh.faces),
                'edges': 0,
                'volume': 0.0,
                'surface_area': 0.0,
                'is_watertight': False,
                'is_convex': False,
                'bounds': [[0,0,0], [0,0,0]],
                'extents': [0,0,0],
                'center_of_mass': [0,0,0],
            }
        

    def from_scad_simplified(
                                self,
                                scad_file: str,
                                parameters: Optional[Dict] = None,
                                planning_segments: int = 16,  # low-poly cylinder approximation
                                fix_mesh: bool = False,
                            ) -> trimesh.Trimesh:
        """
        Generate a simplified mesh for collision planning by overriding
        OpenSCAD's $fn variable to reduce gear tooth polygon count,
        and optionally suppressing teeth entirely via a 'simplified' flag.

        $fn controls the number of fragments in circular shapes — 
        default is often 100+, we drop it to 16 for planning.
        """
        planning_params = dict(parameters or {})
        
        # Override circular resolution — turns gear teeth into low-poly cylinders
        planning_params['$fn'] = planning_segments
        
        # If your SCAD file supports a 'simplified' flag to skip teeth geometry:
        planning_params['simplified'] = 'true'

        return self.from_scad(scad_file, parameters=planning_params, fix_mesh=fix_mesh)