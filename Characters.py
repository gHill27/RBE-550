from Map_Generator import Map

import random

class Character:
    def __init__(self, map:Map, id:int):
        self.coordinate = (-1,-1)
        self.map = map
        self.id = id
        
    def check_collision(self):
        retval = False
        if self.coordinate in self.map.obstacle_coordinate_list:
            retval = True
        return retval

class Enemy(Character):
    
    def __init__(self,map:Map, id:int):
        super().__init__(map,id)
        self.coordinate = self.map.find_open_square()
        self.map.enemy_coordinate_list.append(self.coordinate)
    
    def move(self):
        next_point = self.map.Move_enemy_cell(self.coordinate)
        if self.check_collision():
            self.become_obstacle()
        else:
            self.coordinate = next_point

    def become_obstacle(self):
        self.map.append_new_obstacle(self.coordinate)


class Hero(Character):
    def __init__(self, map:Map, id:int):
        super().__init__(map, id)
        self.coordinate = self.map.find_open_square()
        self.map.hero_coordinate = self.coordinate
        self.teleport_counter = 0
        self.visited_squares = []
        self.stack = []
        self.directions = [(1,0), (0,1) ,(-1,0), (0,-1)]
        self.visit(self.coordinate)
        
    def check_at_goal(self,coordinate_to_visit):
        if coordinate_to_visit == self.map.goal_pos:
            print("reached goal")
            return True
        else:
            return False
        
    def teleport_hero(self):
        if self.teleport_counter < 5:
            self.coordinate = self.map.find_open_square()
            self.teleport_counter = self.teleport_counter + 1

    def visit(self,coordinate_to_visit):
        retval = False
        self.visited_squares.append(coordinate_to_visit)
        if self.check_at_goal(coordinate_to_visit): 
            retval = True ##################################### Fill in return algorithm here
        else: 
            self.determine_next_neighbor(coordinate_to_visit)
        self.move(coordinate_to_visit)
        return retval


    def calculate_search_algorithm(self):
        coord_to_visit = self.stack.pop()
        if self.visit(coord_to_visit):
            self.stack = []
        

        
    def determine_next_neighbor(self,coordinate_visited):
        for direction in self.directions:
            new_coordinatex = coordinate_visited[0] + direction[0]
            new_coordinatey = coordinate_visited[1] + direction[1]
            new_coordinate = (new_coordinatex, new_coordinatey)
            if new_coordinate not in self.visited_squares and self.is_valid_square(new_coordinate) and new_coordinate not in self.stack:
                self.stack.append(new_coordinate)

    def is_valid_square(self, coordinate):
        """
        This will be used to check if the hero has access to the square
        
        This is intended be a safety check to ensure smooth motion
        """
        row, col = coordinate

        # inside grid
        if row < 0 or row >= self.map.grid_num:
            return False
        if col < 0 or col >= self.map.grid_num:
            return False

        # not an obstacle
        if coordinate in self.map.obstacle_coordinate_list:
            return False

        return True


    def move(self, coordinate_to_move):
        self.coordinate = coordinate_to_move