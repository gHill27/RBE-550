from Map_Generator import Map, Status
from wumpus import Wumpus
from firetruck import Firetruck


def main():
    map = Map(Grid_num=50,cell_size=5, fill_percent=0.1)
    print(map.obstacle_coordinate_dict)
    firetruck = Firetruck((10, 10, 0), map, (20, 10, 0))
    firetruck.main_run()
    # wumpus = Wumpus()


if __name__ == "__main__":
    main()
