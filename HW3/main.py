from police import Police
from delivery import Delivery
from Vehicles import Vehicle,State,Map

if __name__ == "__main__":
    # Add this to your __main__ to debug
    police_car = Police(
        startPose=(3, 30, 0), map=Map(12, 3, 0.1), goalPose=(25.5,4.5,0), plot=True
    )
    print(f"Is Start Valid? {police_car.is_state_valid(police_car.start_pos)}")
    police_car.main_run()