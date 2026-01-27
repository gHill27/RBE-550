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
        self.crashed = False
        self.map.enemy_coordinate_list.append(self.coordinate)
    
    def move(self):
        if not self.crashed:
            next_point = self.map.Move_enemy_cell(self.coordinate)
            if self.check_collision():
                self.become_obstacle()
            else:
                self.coordinate = next_point

    def become_obstacle(self):
        self.map.append_new_obstacle(self.coordinate)
        self.crashed = True


class Hero(Character):
    def __init__(self, map:Map, id:int):
        super().__init__(map, id)
        self.planned_coordinate = self.map.find_open_square()
        self.real_coordinate = self.planned_coordinate
        self.map.hero_coordinate = self.planned_coordinate
        self.teleport_counter = 0
        self.visited_squares = []
        self.stack = []
        self.directions = [(1,0), (0,1) ,(-1,0), (0,-1)]
        self.enemy_radius = [(1,0), (0,1) ,(-1,0), (0,-1), (2,0), (0,2), (-2,0), (0,-2)]
        self.parent_dict = {self.planned_coordinate:None}
        self.path_to_victory = []
        self.visit(self.planned_coordinate)
        self.is_route_planned = False
        
    def check_at_goal(self):
        """
        Checks if the hero's current position matches the goal
        """
        if self.planned_coordinate == self.map.goal_pos:
            print("reached goal")
            return True
        else:
            return False
        
    def teleport_hero(self):
        """ Teleports the hero to a random unoccupied cell, clearing his previous trail"""
        if self.teleport_counter < 5:
            self.reset()
            self.real_coordinate = self.map.find_open_square()
            self.planned_coordinate = self.real_coordinate
            self.parent_dict = {self.real_coordinate:None}
            self.visit(self.planned_coordinate)
            self.teleport_counter = self.teleport_counter + 1
            
            
    
    def reset(self):
        """ Clears the previous trail of pathfinding algorithm"""
        for square in self.visited_squares:
            self.map.color_cell(self.map.canvas, square[1], square[0], "white")
            self.map.color_cell(self.map.canvas, self.map.goal_pos[1], self.map.goal_pos[0], "green")
        self.visited_squares = []
        self.stack = []
        self.path_to_victory = []
        self.is_route_planned = False
        
        

    def visit(self,coordinate_to_visit):
        if coordinate_to_visit not in self.visited_squares:    
            self.visited_squares.append(coordinate_to_visit)
            self.planned_coordinate = coordinate_to_visit
            self.determine_next_neighbor()
        


    def calculate_search_algorithm(self):
        if len(self.stack) == 0:
            print("teleporting due to bad spawn!")
            self.teleport_hero() 
        coord_to_visit = self.stack[0]
        self.stack.remove(coord_to_visit)
        self.visit(coord_to_visit)
        if self.check_at_goal(): 
            self.path_to_victory = self.reconstruct_path()
            self.is_route_planned = True
            
            
    def reconstruct_path(self):
        reversed_path = []
        current_coordinate = self.map.goal_pos
        while current_coordinate is not None:
            reversed_path.append(current_coordinate)
            current_coordinate = self.parent_dict[current_coordinate]
        return reversed_path

    def detect_enemy_nearby(self):
        retval = False
        for direction in self.enemy_radius:
            new_coordinatex = self.real_coordinate[0] + direction[0]
            new_coordinatey = self.real_coordinate[1] + direction[1]
            new_coordinate = (new_coordinatex, new_coordinatey)
            if new_coordinate in self.map.enemy_coordinate_list:
                retval = True
            #check for collisions:
        return retval
    
    def determine_next_neighbor(self):
        for direction in self.directions:
            new_coordinatex = self.planned_coordinate[0] + direction[0]
            new_coordinatey = self.planned_coordinate[1] + direction[1]
            new_coordinate = (new_coordinatex, new_coordinatey)
            if new_coordinate not in self.visited_squares and self.is_valid_square(new_coordinate):
                self.stack.append(new_coordinate)
                self.parent_dict[new_coordinate] = self.planned_coordinate

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


    def move(self):
        if(self.detect_enemy_nearby()):
            self.teleport_hero()
        else:
            self.real_coordinate = self.path_to_victory.pop()
        
        
