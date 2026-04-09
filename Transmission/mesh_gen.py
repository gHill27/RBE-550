#!/usr/bin/env python3
"""
mesh_generator.py - Convert OpenSCAD files to 3D trimesh meshes
"""

import subprocess
import numpy as np
import trimesh
import tempfile
from pathlib import Path
from typing import Union, Optional, Dict

class MeshGenerator:
    """Generate 3D trimesh objects from OpenSCAD files"""
    
    def __init__(self, models_folder: str = 'models'):
        """
        Initialize mesh generator.
        
        Args:
            models_folder: Folder containing OpenSCAD models
        """
        self.models_folder = Path(models_folder)
        
        # If models_folder doesn't exist, try current directory
        if not self.models_folder.exists():
            self.models_folder = Path.cwd()
            print(f"⚠️  Models folder not found, using current directory: {self.models_folder}")
        else:
            print(f"✓ Found models folder at: {self.models_folder}")
        
        self._check_openscad()
    
    def _check_openscad(self):
        """Verify OpenSCAD is installed"""
        try:
            result = subprocess.run(['openscad', '--version'], 
                                   capture_output=True, check=True)
            print("✓ OpenSCAD found")
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("OpenSCAD not found. Please install OpenSCAD first.")
    
    def from_scad(self, 
                  scad_file: Union[str, Path], 
                  parameters: Optional[Dict] = None,
                  scale_factor: float = 1.0,
                  fix_mesh: bool = False) -> trimesh.Trimesh:
        """
        Convert OpenSCAD file to 3D mesh.
        
        Args:
            scad_file: Name of OpenSCAD file (will look in models_folder)
            parameters: Dictionary of parameters (-D var=value)
            scale_factor: Scale factor for coordinates
            fix_mesh: Whether to try fixing mesh issues (ignored in this version)
        """
        # Build full path to SCAD file
        scad_path = self.models_folder / scad_file
        
        if not scad_path.exists():
            # Try just the filename as fallback
            scad_path = Path(scad_file)
            if not scad_path.exists():
                # List available files for debugging
                available = list(self.models_folder.glob("*.scad"))
                print(f"\n❌ File not found: {scad_file}")
                print(f"   Looking in: {self.models_folder}")
                if available:
                    print(f"   Available SCAD files:")
                    for f in available:
                        print(f"     - {f.name}")
                raise FileNotFoundError(f"OpenSCAD file not found: {scad_file}")
        
        print(f"📄 Loading: {scad_path}")
        
        # Build command
        cmd = ['openscad']
        
        # Add parameters
        if parameters:
            for key, value in parameters.items():
                if isinstance(value, str):
                    cmd.extend(['-D', f'{key}="{value}"'])
                else:
                    cmd.extend(['-D', f'{key}={value}'])
        
        # Export to temporary STL
        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp:
            stl_file = tmp.name
        
        try:
            cmd.extend(['-o', stl_file, '--export-format', 'stl', str(scad_path)])
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"OpenSCAD export failed: {result.stderr}")
            
            # Load mesh
            mesh = trimesh.load_mesh(stl_file)
            
            # Apply scale
            if scale_factor != 1.0:
                mesh.apply_scale(scale_factor)
            
            # Optionally fix mesh (if requested)
            if fix_mesh:
                try:
                    # Remove degenerate faces if they exist
                    if hasattr(mesh, 'degenerate_faces'):
                        mesh.update_faces(~mesh.degenerate_faces)
                    # Merge duplicate vertices
                    mesh.merge_vertices()
                except Exception as e:
                    print(f"⚠️  Warning: Could not fix mesh: {e}")
            
            return mesh
            
        finally:
            Path(stl_file).unlink(missing_ok=True)
    
    def get_mesh_info(self, mesh: trimesh.Trimesh) -> Dict:
        """Get detailed information about 3D mesh"""
        return {
            'vertices': len(mesh.vertices),
            'faces': len(mesh.faces),
            'edges': len(mesh.edges),
            'volume': mesh.volume if mesh.is_watertight else 0,
            'surface_area': mesh.area,
            'is_watertight': mesh.is_watertight,
            'is_convex': mesh.is_convex,
            'bounds': mesh.bounds.tolist(),
            'extents': mesh.extents.tolist(),
            'center_of_mass': mesh.center_mass.tolist(),
        }