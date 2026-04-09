#!/usr/bin/env python3
"""
main.py - Main script for transmission shaft path planning
"""

import sys
import os
from pathlib import Path

# Add current directory to path (in case of import issues)
sys.path.insert(0, str(Path(__file__).parent))

# Import your modules
from mesh_gen import MeshGenerator
from collision import CollisionChecker3D
from planner import RRTPlanner3D

import numpy as np

def check_environment():
    """Check if everything is set up correctly"""
    print("="*60)
    print("ENVIRONMENT CHECK")
    print("="*60)
    
    # Check current directory
    cwd = Path.cwd()
    print(f"\n📁 Current directory: {cwd}")
    
    # List contents
    print(f"\n📋 Directory contents:")
    for item in cwd.iterdir():
        if item.is_dir():
            print(f"  📁 {item.name}/")
        else:
            print(f"  📄 {item.name}")
    
    # Check for models folder
    models_path = cwd / 'models'
    if models_path.exists():
        print(f"\n✅ Found 'models' folder!")
        print(f"   Contents:")
        for item in models_path.iterdir():
            if item.suffix == '.scad':
                print(f"     - {item.name}")
    else:
        print(f"\n❌ 'models' folder not found at: {models_path}")
        
        # Check parent directory
        parent_models = cwd.parent / 'models'
        if parent_models.exists():
            print(f"   Found 'models' in parent directory: {parent_models}")
            print(f"   Consider moving 'models' to current directory or updating path")
        else:
            print(f"\n   Please create a 'models' folder and put your SCAD files in it:")
            print(f"   mkdir models")
            print(f"   # Then copy your .scad files into models/")
    
    return models_path.exists()

def main():
    """Main function for transmission path planning"""
    
    print("="*60)
    print("TRANSMISSION SHAFT PATH PLANNING")
    print("="*60)
    
    # Check environment first
    if not check_environment():
        print("\n❌ Cannot proceed without models folder!")
        return
    
    # Transmission parameters (from your SCAD)
    case_length = 280
    case_width = 210
    case_height = 300
    bearing_offset_height = 215
    cs_bearing_offset_height = 100
    case_thickness = 25
    
    # Calculate positions
    primary_shaft_start = (-150, 0, bearing_offset_height + 0.5 * case_thickness)
    secondary_shaft_position = (
        0.5 * case_length + case_thickness,  # ~165
        0,
        cs_bearing_offset_height + 0.5 * case_thickness  # ~112.5
    )
    goal_position = (150, 0, bearing_offset_height + 0.5 * case_thickness)
    
    # Define planning bounds
    bounds = [
        (-200, 200),  # X: wider than case
        (-150, 150),  # Y: wider than case
        (0, 350)      # Z: cover bearing height
    ]
    
    print(f"\n📍 Positions:")
    print(f"   Primary shaft start: {primary_shaft_start}")
    print(f"   Secondary shaft: {secondary_shaft_position}")
    print(f"   Goal: {goal_position}")
    
    try:
        # Initialize planner
        print(f"\n🔧 Initializing planner...")
        planner = RRTPlanner3D(
            bounds=bounds,
            models_folder='models'  # Your SCAD files are in this folder
        )
        
        # Add obstacles
        print(f"\n📦 Loading obstacles from 'models' folder...")
        
        # Add transmission case
        planner.add_obstacle(
            'transmission_case.scad',  # Should exist in models/
            'Case',
            position=(0, 0, 0)
        )
        
        # Add countershaft (secondary shaft)
        planner.add_obstacle(
            'secondary_shaft.scad',  # Should exist in models/
            'Countershaft',
            position=secondary_shaft_position
        )
        
        # Set robot (primary shaft / mainshaft)
        print(f"\n🤖 Loading robot (mainshaft)...")
        planner.set_robot(
            scad_file='primary_shaft.scad',  # Should exist in models/
            start_position=primary_shaft_start
        )
        
        # Plan path
        waypoints = planner.plan_path(
            start=primary_shaft_start,
            goal=goal_position,
            planner_type='rrt_connect',
            max_time=10.0,
            goal_tolerance=5.0
        )
        
        if waypoints:
            print(f"\n✅ Success! Found path with {len(waypoints)} waypoints")
            
            # Check if path goes through hole
            hole_center = np.array([152.5, 0, 215])
            path_array = np.array(waypoints)
            distances = np.linalg.norm(path_array[:, :3] - hole_center, axis=1)
            min_distance = np.min(distances)
            
            if min_distance < 40:
                print(f"✓ Path goes through bearing hole (clearance: {40 - min_distance:.1f}mm)")
            else:
                print(f"⚠️ Path may not go through hole (distance: {min_distance:.1f}mm)")
            
            # Save path
            np.save('planned_path.npy', path_array)
            print("✓ Path saved to 'planned_path.npy'")
            
            # Ask to visualize
            response = input("\n🎨 Visualize path? (y/n): ")
            if response.lower() == 'y':
                planner.visualize_path(waypoints)
        else:
            print("\n❌ Failed to find path!")
            
    except FileNotFoundError as e:
        print(f"\n❌ File error: {e}")
        print("\nPlease make sure your models folder contains:")
        print("  - transmission_case.scad")
        print("  - secondary_shaft.scad")
        print("  - primary_shaft.scad")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()