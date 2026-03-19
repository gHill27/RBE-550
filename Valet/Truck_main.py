from police import Police
from truck import Truck
from delivery import Delivery
from Vehicles import Vehicle,State,Map

if __name__ == "__main__":
    vechile = Truck(
        startPose=(7, 30, 0), map=Map(12, 3, 0.1), goalPose=(30,7,0), plot=True
    )
    #print(f"Is Start Valid? {vechile.is_state_valid(vechile.start_pos)}")
    vechile.main_run()