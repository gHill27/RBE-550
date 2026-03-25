from Map_Generator import Map, Status
from wumpus import Wumpus
from firetruck import Firetruck
import time


def main():
    wumpus_pose = (100,100,0)
    firetruck_pose = (10,10,0)
    map = Map(Grid_num=50,cell_size=5, fill_percent=0.1,firetruck_pose=firetruck_pose,wumpus_pose=wumpus_pose)
    # print(map.obstacle_set)
    firetruck = Firetruck(firetruck_pose, map, plot=True)
    firetruck.build_tree()
    firetruck.viz.plot_prm(map,firetruck.graph,firetruck.nodes)
    # print(firetruck.graph)
    # firetruck.main_run()
    wumpus = Wumpus()


if __name__ == "__main__":
    main()
