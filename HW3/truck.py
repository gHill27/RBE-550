from Vehicles import Vehicle

class Truck(Vehicle):
    def __init__(self, startState):
        super().__init__(height=5.4, width=2.0, startState=startState)
