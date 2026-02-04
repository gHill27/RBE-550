from Map_Generator import Map




# create map
map1 = Map(64, 20, 0.1)
map1.generate_hero()
map1.generate_enemies(50)

def main_loop():
    map1.update_characters()
    if map1.check_game_over():
        map1.renderer.game_over_screen()
    else:
        map1.renderer.root.after(300, main_loop)


main_loop()
map1.renderer.Open_map()
