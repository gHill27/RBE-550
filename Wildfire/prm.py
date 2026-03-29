import numpy as np
import matplotlib.pyplot as plt
import roboticstoolbox as rtb
from spatialmath import SE2

# 1. Setup the Environment
# We'll create a 10x10 binary occupancy grid
grid = np.zeros((100, 100))
grid[30:70, 40:50] = 1  # Add a simple vertical wall obstacle
map = rtb.OccupancyGrid(grid)

# 2. Define Robot Parameters
rho = 1.0  # Minimum turning radius for the Dubins car

# 3. Initialize the Dubins Planner
# The DubinsPlanner in RTB handles the steering geometry
dubins = rtb.DubinsPlanner(curvature=1/rho)

def is_collision_free(start, end, map):
    """
    Checks if the Dubins path between two nodes hits an obstacle.
    """
    path, _ = dubins.query(start, end)
    # Check points along the calculated Dubins path
    for point in path:
        if map.isoccupied(point[:2]):
            return False
    return True

# 4. PRM Sampling
n_samples = 50
nodes = []
while len(nodes) < n_samples:
    # Sample x, y, theta
    sample = np.random.uniform(low=[0, 0, -np.pi], high=[10, 10, np.pi])
    if not map.isoccupied(sample[:2]):
        nodes.append(sample)

# 5. Connect Nodes (The Roadmap)
adj_list = []
max_dist = 4.0 # Connection radius

for i, n1 in enumerate(nodes):
    for j, n2 in enumerate(nodes):
        if i == j: continue
        
        # Calculate Dubins distance
        # We use the planner to find the path length
        _, length = dubins.query(n1, n2)
        
        if length < max_dist:
            if is_collision_free(n1, n2, map):
                adj_list.append((i, j))

# 6. Visualization
plt.figure(figsize=(8, 8))
map.plot()
for i, j in adj_list:
    n1, n2 = nodes[i], nodes[j]
    path, _ = dubins.query(n1, n2)
    plt.plot(path[:, 0] * 10, path[:, 1] * 10, 'b-', alpha=0.3, lw=1) # Scale if map is 100x100

nodes = np.array(nodes)
plt.scatter(nodes[:, 0] * 10, nodes[:, 1] * 10, c='red', s=10)
plt.title("PRM with Dubins Path Connections")
plt.show()