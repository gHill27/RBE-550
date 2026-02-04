import random
from collections import deque

class Character:
    def __init__(self, start_coordinate: tuple[int,int]):
        self._coordinate = start_coordinate
        

    def getCoordinate(self) -> tuple[int,int]:
        return self._coordinate
    
    def move(self, coordinate: tuple[int,int]):
        self._coordinate = coordinate

class Enemy(Character):

    def __init__(self, start_coordinate: tuple[int,int]):
        super().__init__(start_coordinate)
        self.crashed = False
        self.directions = [(1, 0), (1,1), (0, 1), (-1,1), (-1, 0), (1,-1), (0, -1), (-1,-1)]

    def move(self, coordinate:tuple[int,int]):
        """Moves the enemy to the closest adjecent square to the hero"""
        if not self.crashed:
            self._coordinate = coordinate

    def become_obstacle(self):
        """Makes the enemy become an obstacle"""
        self.crashed = True


class Hero(Character):
    def __init__(self, start_coordinate:tuple[int,int]):
        super().__init__(start_coordinate)
        self.teleport_counter = 0
        self.is_route_planned = False
        self.visited = []
        self.path_to_victory = []
        self.parent_dict = {}
        self.queue = deque([start_coordinate])
        self.directions = [(0, 1),(-1, 0),(0, -1),(1, 0)]
        self.enemy_radius = [
            (1, 0),
            (1, 1),
            (0, 1),
            (1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (-1, -1),
            (2, 0),
            (2,2),
            (0, 2),
            (-2,2),
            (-2, 0),
            (2,-2),
            (0, -2),
            (-2,-2)
        ] #uses a large radius to ensure the robot never comes in contact with another approaching robot

    

    def teleport_hero(self,coordinate:tuple[int,int]):
        """Teleports the hero to a random unoccupied cell, clearing his previous trail"""
        if self.teleport_counter < 5:
            self.reset(coordinate)
            self.teleport_counter = self.teleport_counter + 1

    def reset(self,start_coord:tuple[int,int]):
        """Clears the previous trail of pathfinding algorithm"""
        self.visited = []
        self.queue = deque([start_coord])
        self.path_to_victory = []
        self.is_route_planned = False
        self._coordinate = start_coord
        self.parent_dict = {start_coord: None}
