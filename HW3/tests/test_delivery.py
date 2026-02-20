import pytest
from Vechiles import Delivery
import math 
from math import *

@pytest.fixture
def delivery_bot():
    bot = Delivery()
    return bot

def test_motion_primitives_range(delivery_bot:Delivery):
    mps = delivery_bot.calculate_motion_primitives()
    for key,value in mps.items():
        x,y = value[0],value[1]
        assert -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5

def test_motion_primitives_total(delivery_bot:Delivery):
    mps = delivery_bot.calculate_motion_primitives()
    for key,value in mps.items():
        x,y = value[0],value[1]
        assert sqrt(x**2 + y**2) == pytest.approx(0.5)