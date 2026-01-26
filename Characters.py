from Map_Generator import Map


class Character:
    def __init__(self, coordinate, map:Map):
        self.coordinate = coordinate
        self.map = map
        
    def check_collision(self):
        retval = False
        if self.coordinate in self.map.obstacle_coordinate_list:
            retval = True
        return retval

class Enemy(Character):
    
    def __init__(self):
        super().__init__()
    
    def move(self):
        pass

    def become_obstacle():
        pass

    def generate_hero():
        pass

    def generate_enemies():
        pass

    def move_enemies():
        pass

    def generate_goal():
        pass

    def teleport_hero():
        pass