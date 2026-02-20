import numpy as np
import math

def generate_arc_lookup(max_dist=3.0, res=0.5):
    """
    Generates a table of (x, y, theta, curvature, length)
    """
    lookup_table = []
    
    # Sample a grid of potential target points
    x_coords = np.arange(0.5, max_dist, res)
    y_coords = np.arange(-max_dist, max_dist, res)
    
    for x in x_coords:
        for y in y_coords:
            if x == 0 and y == 0: continue
            
            # 1. Calculate the radius of the circle passing through (0,0) and (x,y)
            # Formula: R = (x^2 + y^2) / (2y)
            if y == 0: # Straight line case
                kappa = 0.0
                length = x
                theta = 0.0
            else:
                radius = (x**2 + y**2) / (2 * y)
                kappa = 1.0 / radius
                
                # 2. Calculate the arc length and final heading
                # Angle of the sector
                alpha = math.atan2(x, radius - y)
                length = radius * alpha
                theta = alpha # For a circle, change in heading = angle of sector
            
            # Store the primitive if it's physically reasonable
            if length < max_dist * 1.5:
                lookup_table.append([x, y, theta, kappa, length])
                
    return np.array(lookup_table)

# Generate and save
table = generate_arc_lookup()
np.savetxt("simple_lookup.csv", table, delimiter=",", 
           header="x,y,theta,kappa,length", comments='')
print(f"Generated {len(table)} primitives.")