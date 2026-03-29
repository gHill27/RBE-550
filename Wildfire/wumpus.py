# uses a generic A* star map traversal.

from Map_Generator import Map, Status
import math
import heapq


class Wumpus:
    def __init__(self, map:Map):
        self.map : Map = map
        self.pose = self.map.wumpus_pose
        self.directions = [(1,0),(1,1),(0,1),(-1,1),(-1,-1),(1,-1),(-1,0),(0,-1)]
        pass

    def plan(self):
        #use knowledge of map to traverse 
        goal = self.find_closest_obstacle()
        start_node = self.map.wumpus_pose
        # open_list stores: (f_score, current_coord)
        open_list = []
        heapq.heappush(
            open_list, (0 + self.calculate_heurisitic(start_node, goal), start_node)
        )

        # Track the best cost to reach a coordinate
        costHistory: dict[tuple, float] = {start_node: 0}
        # Track the path: {child_coord: parent_coord}
        # each coord should include a tuple of (x,y,theta)
        came_from: dict[tuple, tuple] = {}

        while open_list:
            # Get the node with the lowest f_score
            _, current_pose = heapq.heappop(open_list)
            if current_pose == goal:
                path = self._unwind(came_from,current_pose)
                return path
            
            for direction in self.directions:
                grid_x,grid_y = current_pose[0] + direction[0], current_pose[1] + direction[1]
                cost = math.dist((grid_x,grid_y),goal)
            
        

        pass
    def _unwind(came_from: dict[int, int], current: int) -> list[int]:
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(current)
        return path[::-1]

    def burn(self):
        #for obstacles within radius
        # map.set_status(burning)
        pass

    def find_closest_obstacle(self):
        closest_dist = math.inf
        closest_obstacle = None
        for obstacle in self.map.obstacle_set:
            dist = math.dist((self.map.wumpus_pose[0],self.map.wumpus_pose[1]),obstacle)
            if dist < closest_dist:
                closest_dist = dist
                closest_obstacle = obstacle

        return closest_obstacle
    