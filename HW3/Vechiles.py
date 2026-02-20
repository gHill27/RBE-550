import math
from math import *
from Map_Generator import Map

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import heapq

from shapely.geometry import box
from shapely.affinity import rotate, translate

from pathVisualizer import PlannerVisualizer
import time

class Vechile:
    def __init__(self,height:float,width:float):
        self.height = height
        self.width = width
        self.viz = PlannerVisualizer()
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
        self.start_pos = (0.4,0.4,0) #(x,y,theta)
        pass

    def plan(self,step_size = 500):
        """A* planner using motion primitives at a 0.5m step size"""
        mp = self.calculate_motion_primitives()
        count = 0
        start_node = self.start_pos
        #self.map.goal_pos = (1.3,2.2) #TODO fix this so it doesn't brick the code everytime
        
        # open_list stores: (f_score, current_coord)
        open_list = []
        heapq.heappush(open_list, (0 + self.calculate_heurisitic(start_node), start_node))
        
        # Track the best cost to reach a coordinate
        costHistory = {start_node: 0}
        # Track the path: {child_coord: parent_coord}
        #each coord should include a tuple of (x,y,theta)
        came_from = {}

        while open_list:
            # Get the node with the lowest f_score
            current_f, current_pos = heapq.heappop(open_list)

            # Check if we are "close enough" to the goal (floating point friendly)
            if self.calculate_heurisitic(current_pos) < 0.1: 
                print("Goal Reached!")
                self.exploredNodes = came_from
                final_path = self.reconstruct_path(came_from, current_pos)
                self.viz.show_final(final_path, costHistory, self.map.obstacle_coordinate_list, self.map.goal_pos)
                return final_path

            # Expand neighbors using motion primitives
            for key, value in mp.items():
                dx,dy,dtheta = value[0],value[1],key
                neighbor = (current_pos[0] + dx, current_pos[1] + dy, current_pos[2]+dtheta)
                # Rounding for consistency in the dictionary keys
                neighbor = (round(neighbor[0], 2), round(neighbor[1], 2), round(neighbor[2], 1))

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
                # LIVE UPDATE CALL
                if self.viz and count % step_size == 0:
                    self.viz.update(neighbor, costHistory, self.map.obstacle_coordinate_list, self.map.goal_pos)
                count += 1

        self.exploredNodes = came_from
        print("No path found.")
        return None


    def reconstruct_path(self, came_from, current) -> list[tuple[float,float]]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1] # Return reversed path

        
    def calculate_heurisitic(self,pose:tuple[float,float,float]) -> float:
        """Calculates the euclidian distance between two points"""
        cost = round(sqrt((self.map.goal_pos[0]-pose[0])**2 + (self.map.goal_pos[1]-pose[1])**2),2)
        return cost

    def calculate_motion_primitives(self,step_precision:int = 16) -> dict[float,tuple[float,float]]:
        motion_primatives = {}
        for i in range(step_precision):
            angle = 0 + i*(360/step_precision)
            coordinate = self.motion_primitive_equation(angle)
            motion_primatives[angle] = coordinate 
        return motion_primatives

    def motion_primitive_equation(self,angle:float,distance_traveled = 0.5) -> tuple[float]:
        angle_rad = math.radians(angle)
        point = (round(distance_traveled*cos(angle_rad),2), round(distance_traveled*sin(angle_rad),2))
        return point
    


    def get_robot_polygon(self, x, y, theta_degrees):
        """
        Creates a 0.7m x 0.57m rectangle centered at (x, y) 
        and rotated by theta_degrees.
        """
        # 1. Create the base rectangle centered at (0,0)
        # box(minx, miny, maxx, maxy)
        width = 0.7
        height = 0.57
        rect = box(-width/2, -height/2, width/2, height/2)
        
        # 2. Rotate the rectangle around its center (0,0)
        # Note: In your flipped grid, you might need to check if 
        # positive rotation is clockwise or counter-clockwise.
        rotated_rect = rotate(rect, theta_degrees, origin=self.start_pos)
        
        # 3. Translate the rotated rectangle to the robot's position
        robot_shape = translate(rotated_rect, xoff=x, yoff=y)
        
        return robot_shape
    
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
                    #self.viz.draw(path,self.exploredNodes)
                else:
                    print("No path found, but plotting explored area...")
                    #self.viz.draw(None)
            case False:
                if path:
                    print(f"Path found with {len(path)} steps!")
                    #self.plot_path(path)
                else:
                    print("No path found, but plotting explored area...")
                    #self.plot_path(None)

class Police(Vechile):
    def __init__(self):
        super().__init__(height = 5.2, width = 1.8)

class Truck(Vechile):
    def __init__(self):
        super().__init__(height = 5.4,width = 2.0)

if __name__ == '__main__':
    deliver = Delivery()
    print(deliver.main_run())