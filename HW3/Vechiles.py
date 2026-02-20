import math
from math import *
from Map_Generator import Map

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import heapq

class Vechile:
    def __init__(self,height:float,width:float):
        self.height = height
        self.width = width
        pass

    def plan(self):
        #should be based off the type of vechile
        pass

    def plot(self):
        pass


class Delivery(Vechile):
    def __init__(self):
        super().__init__(height = 0.7, width = 0.57)
        self.map = Map(12,100,0.1)
        self.exploredNodes : dict[tuple[float,float],tuple[float,float]] = {}
        self.start_pos = (0.4,0.4)
        pass

    def plan(self):
        """A* planner using motion primitives at a 0.5m step size"""
        mp = self.calculate_motion_primitives()
        start_node = self.start_pos
        #self.map.goal_pos = (1.3,2.2) #TODO fix this so it doesn't brick the code everytime
        
        # open_list stores: (f_score, current_coord)
        open_list = []
        heapq.heappush(open_list, (0 + self.calculate_heurisitic(start_node), start_node))
        
        # Track the best cost to reach a coordinate
        costHistory = {start_node: 0}
        # Track the path: {child_coord: parent_coord}
        came_from = {}

        while open_list:
            # Get the node with the lowest f_score
            current_f, current_pos = heapq.heappop(open_list)

            # Check if we are "close enough" to the goal (floating point friendly)
            if self.calculate_heurisitic(current_pos) < 0.1: 
                print("Goal Reached!")
                self.exploredNodes = came_from
                return self.reconstruct_path(came_from, current_pos)

            # Expand neighbors using motion primitives
            for dx, dy in mp:
                neighbor = (current_pos[0] + dx, current_pos[1] + dy)

                if (math.floor(neighbor[0]),math.floor(neighbor[1])) in self.map.obstacle_coordinate_list:
                    continue
                
                # Assume constant cost of 0.5 (step size) for each primitive
                tentativeCostToCome = costHistory[current_pos] + 0.5
                
                if neighbor not in costHistory or tentativeCostToCome < costHistory[neighbor]:
                    # This path to neighbor is better than any previous one
                    came_from[neighbor] = current_pos
                    costHistory[neighbor] = tentativeCostToCome
                    estimatedCost = tentativeCostToCome + self.calculate_heurisitic(neighbor)
                    heapq.heappush(open_list, (estimatedCost, neighbor))

        self.exploredNodes = came_from
        print("No path found.")
        return None


    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1] # Return reversed path

    # def plot_path(self, path, g_score_dict=None):
    #     """
    #     Plots the planned path, the start/goal, and the explored lattice nodes.
    #     """
    #     plt.figure(figsize=(10, 8))
        
    #     # 1. Plot all explored nodes (the lattice) if provided
    #     if g_score_dict:
    #         all_nodes = list(g_score_dict.keys())
    #         xs, ys = zip(*all_nodes)
    #         plt.scatter(xs, ys, c='lightgray', s=5, label='Explored Nodes')

    #     # 2. Plot the final path
    #     if path:
    #         px, py = zip(*path)
    #         plt.plot(px, py, 'b-o', linewidth=2, markersize=4, label='Planned Path')
        
    #     # 3. Plot Start and Goal
    #     plt.plot(0, 0, 'go', markersize=10, label='Start') # Start at (0,0)
    #     plt.plot(self.map.goal_pos[0], self.map.goal_pos[1], 'ro', markersize=10, label='Goal')

    #     # Formatting
    #     plt.title("A* State Lattice Planner Path")
    #     plt.xlabel("X (meters)")
    #     plt.ylabel("Y (meters)")
    #     plt.legend()
    #     plt.grid(True, linestyle='--', alpha=0.6)
    #     plt.axis('equal') # Vital for seeing 0.5m steps correctly
    #     plt.show()

    def plot_path(self, path, g_score_dict=None):
        """
        Plots the path on a 12x12 grid with (0,0) at the top-left.
        """
        fig, ax = plt.subplots(figsize=(20, 20))
        
        
        # 1. Plot Explored Lattice
        if g_score_dict:
            all_nodes = list(g_score_dict.keys())
            xs, ys = zip(*all_nodes)
            ax.scatter(xs, ys, c='lightgray', s=5, label='Explored')

        # 2. Plot Final Path
        if path:
            px, py = zip(*path)
            ax.plot(px, py, 'b-o', linewidth=2, markersize=4, label='Path')
        # 3 plot obstacles in environment:
        for coord in self.map.obstacle_coordinate_list:
            rect = patches.Rectangle((coord[0],coord[1]),1,1,linewidth =1, edgecolor='orange',facecolor = 'black',alpha=0.7,label='obstacle')
            ax.add_patch(rect)
        
        # 3. Plot Start and Goal
        ax.plot(self.start_pos[0],self.start_pos[1], 'go', markersize=10, label='Start (0,0)') 
        ax.plot(self.map.goal_pos[0], self.map.goal_pos[1], 'ro', markersize=10, label='Goal')

        # --- THE "FLIP" AND "SCALE" LOGIC ---
        
        # Set the window to exactly 12x12
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 12) 
        
        # Invert the Y-axis: This moves 0 to the top and 12 to the bottom
        ax.invert_yaxis()
        
        # Move the X-axis to the top to match the "Top-Left" origin feel
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')

        ax.set_title("12x12 State Lattice (Top-Left Origin)", pad=20)
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        #ax.legend(loc='lower right')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_aspect('equal') 
        
        plt.show()
        
    def calculate_heurisitic(self,coordinate):
        """Calculates the euclidian distance between two points"""
        cost = round(sqrt((self.map.goal_pos[0]-coordinate[0])**2 + (self.map.goal_pos[1]-coordinate[1])**2),2)
        return cost

    def calculate_motion_primitives(self,step_precision = 16):
        motion_primatives = []
        for i in range(step_precision):
            angle = 0 + i*(360/step_precision)
            coordinate = self.motion_primitive_equation(angle)
            motion_primatives.append(coordinate)
        return motion_primatives

    def motion_primitive_equation(self,angle:float,distance_traveled = 0.5) -> tuple[int]:
        angle_rad = math.radians(angle)
        point = (round(distance_traveled*cos(angle_rad),2), round(distance_traveled*sin(angle_rad),2))
        return point
    
    def minkowski_sum():

        pass

    def collision_checker():
        pass

    def main_run(self,plot_explored = True):
        # Run the planner
        path = self.plan() 
        match plot_explored:
            case True:
                if path:
                    print(f"Path found with {len(path)} steps!")
                    self.plot_path(path,self.exploredNodes)
                else:
                    print("No path found, but plotting explored area...")
                    self.plot_path(None)
            case False:
                if path:
                    print(f"Path found with {len(path)} steps!")
                    self.plot_path(path)
                else:
                    print("No path found, but plotting explored area...")
                    self.plot_path(None)

class Police(Vechile):
    def __init__(self):
        super().__init__(height = 5.2, width = 1.8)

class Truck(Vechile):
    def __init__(self):
        super().__init__(height = 5.4,width = 2.0)

if __name__ == '__main__':
    deliver = Delivery()
    print(deliver.main_run())