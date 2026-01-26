from Map_Generator import Map

from Characters import Enemy
from Characters import Hero

#create map
map1 = Map(30,30,0.2)
map1.Fill_map()

#enemy generation function
def generate_enemies(number_of_enemies):
    enemy_list = []
    for i in range(number_of_enemies):
        enemy = Enemy(map1,i)
        enemy_list.append(enemy)
    return enemy_list
#real enemy generation
enemies = generate_enemies(10)

#hero generation
hero = Hero(map1,0)

#finally goal setting
map1.generate_goal()

def main_loop():
    enemy_coord_list = []
    for enemy in enemies:
        enemy.move()
        enemy_coord_list.append(enemy.coordinate)
    hero.move((1,1))
    map1.update_characters(enemy_coord_list,hero.coordinate)
    map1.root.after(300,main_loop)

main_loop()
map1.Open_map()