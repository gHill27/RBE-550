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
        self.coordinate = self.map.find_open_square()
        self.map.hero_coordinate = self.coordinate
        self.teleport_counter = 0
        self.visited_squares = []
        self.stack = []
        self.directions = [(1,0), (0,1) ,(-1,0), (0,-1)]
        self.visit(self.coordinate)
        
    def check_at_goal(self,coordinate_to_visit):
        """
        Checks if the hero's current position matches the goal
        """
        if coordinate_to_visit == self.map.goal_pos:
            print("reached goal")
            return True
        else:
            return False
        
    def teleport_hero(self):
        """ Teleports the hero to a random unoccupied cell, clearing his previous trail"""
        if self.teleport_counter < 5:
            self.coordinate = self.map.find_open_square()
            self.teleport_counter = self.teleport_counter + 1
            self.reset_trail()
    
    def reset_trail(self):
        """ Clears the previous trail of pathfinding algorithm"""
        for square in self.visited_squares:
            self.map.color_cell(self.map.canvas, square[1], square[0], "white")
            self.map.color_cell(self.map.canvas, self.map.goal_pos[1], self.map.goal_pos[0], "green")
        self.visited_squares = []
        self.stack = []

    def visit(self,coordinate_to_visit):
        if coordinate_to_visit not in self.visited_squares:    
            self.visited_squares.append(coordinate_to_visit)
            self.determine_next_neighbor(coordinate_to_visit)
            self.move(coordinate_to_visit)
        


    def calculate_search_algorithm(self):
        coord_to_visit = self.stack.pop()
        self.visit(coord_to_visit)
        if self.check_at_goal(coord_to_visit): 
            ##################################### Fill in return algorithm here
            self.reset_trail()
            
        
    def determine_next_neighbor(self,coordinate_visited):
        for direction in self.directions:
            new_coordinatex = coordinate_visited[0] + direction[0]
            new_coordinatey = coordinate_visited[1] + direction[1]
            new_coordinate = (new_coordinatex, new_coordinatey)
            if new_coordinate not in self.visited_squares and self.is_valid_square(new_coordinate):
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