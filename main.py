from Map_Generator import Map
from Characters import Character
from Characters import Enemy

#create map
map1 = Map(64,20,0.2)

#enemy generation
enemy1 = Character((1,1),map1)


map1.update_characters([enemy1], (2,2))
map1.Fill_map()
map1.Open_map()
