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
        next_point = self.map.random_adjacent_cell(self.coordinate)
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
        

    def teleport_hero(self):
        if self.teleport_counter < 5:
            self.coordinate = self.map.find_open_square()
            self.teleport_counter = self.teleport_counter + 1

    def calculate_search_algorithm(self):
        pass

    def move(self, coordinate_to_move):
        pass