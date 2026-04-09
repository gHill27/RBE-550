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
    
    def __init__(self, openscad_path: str = 'openscad'):
        self.openscad_path = openscad_path
        self._check_openscad()
    
    def _check_openscad(self):
        """Verify OpenSCAD is installed"""
        try:
            subprocess.run([self.openscad_path, '--version'], 
                          capture_output=True, check=True)
            print("✓ OpenSCAD found")
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("OpenSCAD not found. Please install OpenSCAD first.")
    
    def from_scad(self, 
                  scad_file: Union[str, Path], 
                  parameters: Optional[Dict] = None,
                  scale_factor: float = 1.0) -> trimesh.Trimesh:
        """
        Convert OpenSCAD file to 3D mesh.
        
        Args:
            scad_file: Path to OpenSCAD file
            parameters: Dictionary of parameters (-D var=value)
            scale_factor: Scale factor for coordinates
        
        Returns:
            3D trimesh object
        """
        scad_path = Path(scad_file)
        if not scad_path.exists():
            raise FileNotFoundError(f"OpenSCAD file not found: {scad_file}")
        
        # Build command
        cmd = [self.openscad_path]
        
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
            
            # Fix normals if needed
            if not mesh.is_watertight:
                mesh.fix_normals()
                mesh.remove_degenerate_faces()
            
            return mesh
            
        finally:
            Path(stl_file).unlink(missing_ok=True)
    
    def get_mesh_info(self, mesh: trimesh.Trimesh) -> Dict:
        """Get detailed information about 3D mesh"""
        return {
            'vertices': len(mesh.vertices),
            'faces': len(mesh.faces),
            'edges': len(mesh.edges),
            'volume': mesh.volume,
            'surface_area': mesh.area,
            'is_watertight': mesh.is_watertight,
            'is_convex': mesh.is_convex,
            'bounds': mesh.bounds.tolist(),
            'extents': mesh.extents.tolist(),
            'center_of_mass': mesh.center_mass.tolist(),
        }


# Quick conversion function
def scad_to_mesh(scad_file: Union[str, Path], **kwargs) -> trimesh.Trimesh:
    """Quick one-shot conversion from OpenSCAD to mesh"""
    generator = MeshGenerator()
    return generator.from_scad(scad_file, **kwargs)