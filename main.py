from Map_Generator import Map

from Characters import Enemy
from Characters import Hero

#create map
map1 = Map(64,20,0.1)
map1.Fill_map()
map1.generate_goal()

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


def main_loop():
    enemy_coord_list = []
    
    while(not hero.is_route_planned):
        hero.calculate_search_algorithm()

    if(len(hero.path_to_victory) > 0):
        hero.move()
        for enemy in enemies:
            enemy.move()
            if not enemy.crashed:
                enemy_coord_list.append(enemy.coordinate)

    map1.update_characters(enemy_coord_list,hero.real_coordinate)
    if map1.check_game_over(): map1.game_over_screen()
    else:
        map1.root.after(300,main_loop)

main_loop()
map1.Open_map()