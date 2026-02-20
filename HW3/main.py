import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Renderer:
    def __init__(self, grid_num):
        self.grid_num = grid_num
        self.hero_pos = [0, 0] # Start at top-left
        self.hero_shape = None
        
        # 1. Setup Figure
        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self._setup_grid()
        
        # 2. Bind the "key_press_event" to our function
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        
        # 3. Draw the initial Hero
        self._update_hero()

    def _setup_grid(self):
        self.ax.set_xlim(0, self.grid_num)
        self.ax.set_ylim(0, self.grid_num)
        self.ax.invert_yaxis()
        self.ax.set_xticks(range(self.grid_num + 1))
        self.ax.set_yticks(range(self.grid_num + 1))
        self.ax.grid(True, which='both', color='lightgray', linewidth=0.5)
        self.ax.set_aspect('equal')
        # Clean up labels
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])

    def _update_hero(self):
        """Draws or moves the hero without leaving a trail"""
        x, y = self.hero_pos
        if self.hero_shape is None:
            self.hero_shape = patches.Circle((x + 0.5, y + 0.5), 0.4, color="blue")
            self.ax.add_patch(self.hero_shape)
        else:
            self.hero_shape.set_center((x + 0.5, y + 0.5))
        
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        """Handle movement logic"""
        if event.key == 'up' and self.hero_pos[1] > 0:
            self.hero_pos[1] -= 1
        elif event.key == 'down' and self.hero_pos[1] < self.grid_num - 1:
            self.hero_pos[1] += 1
        elif event.key == 'left' and self.hero_pos[0] > 0:
            self.hero_pos[0] -= 1
        elif event.key == 'right' and self.hero_pos[0] < self.grid_num - 1:
            self.hero_pos[0] += 1
        
        self._update_hero()

    def Open_map(self):
        # This is the "Main Loop". It blocks until the window is closed.
        print("Use Arrow Keys to move. Close window to exit.")
        plt.show()

if __name__ == "__main__":
    game = Renderer(10)
    game.Open_map()