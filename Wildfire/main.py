from Map_Generator import Map, Status
from wumpus import Wumpus
from firetruck import Firetruck


def main():
    map = Map(10, 5, 0.1)
    print(map.obstacle_coordinate_dict)
    firetruck = Firetruck((1, 1, 0), map, (10, 10, 0))
    firetruck.plan((10, 10, 0))
    # wumpus = Wumpus()


if __name__ == "__main__":
    main()
