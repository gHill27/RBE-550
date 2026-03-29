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
        visited_set = set()
        heapq.heappush(
            open_list, (0 + self.calculate_heurisitic(start_node, goal), start_node)
        )

        # Track the best cost to reach a coordinate
        costHistory: dict[tuple, float] = {start_node: 0}
        # Track the path: {child_coord: parent_coord}
        # each coord should include a tuple of (x,y,theta)
        came_from: dict[tuple, tuple] = {}

        while open_list:
            # 1. Get the node with the lowest f_score (g + h)
            _, current_pose = heapq.heappop(open_list)

            # 2. Check if we reached the goal
            if current_pose == goal:
                return self._unwind(came_from, current_pose)
            
            # 3. Skip if we've already processed this pose with a better cost
            if current_pose in visited_set:
                continue
            visited_set.add(current_pose)
            
            for direction in self.directions:
                # 4. Calculate neighbor coordinates
                # Assuming direction is (dx, dy)
                neighbor = (current_pose[0] + direction[0], 
                            current_pose[1] + direction[1])

                # 5. Calculate cost to reach neighbor (g_score)
                # We assume a cost of 1.0 per grid step
                tentative_g = costHistory[current_pose] + 1.0

                # 6. Update if this path to the neighbor is better than any found before
                if tentative_g < costHistory.get(neighbor, float('inf')):
                    came_from[neighbor] = current_pose
                    costHistory[neighbor] = tentative_g
                    
                    # f = g + h
                    h_cost = self.calculate_heurisitic(neighbor, goal)
                    f_score = tentative_g + h_cost
                    
                    heapq.heappush(open_list, (f_score, neighbor))

        return None


    def _unwind(came_from: dict[int, int], current: int) -> list[int]:
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(current)
        return path[::-1]

    def burn(self):
        nearby_obstacles = []
        for direction in self.directions:
            adjectent_square = (self.map.wumpus_pose[0] + direction[0], self.map.wumpus_pose[1] + direction[1])
            if adjectent_square in self.map.obstacle_set():
                nearby_obstacles.append(adjectent_square)

        self.map.set_status_on_obstacles(adjectent_square,Status.BURNING)

    def find_closest_obstacle(self):
        closest_dist = math.inf
        closest_obstacle = None
        for obstacle in self.map.obstacle_set:
            dist = math.dist((self.map.wumpus_pose[0],self.map.wumpus_pose[1]),obstacle)
            if dist < closest_dist:
                closest_dist = dist
                closest_obstacle = obstacle

        return closest_obstacle
    
    
    