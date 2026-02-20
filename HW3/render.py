import tkinter as tk


class Renderer:
    # init function
    def __init__(self, grid_num, cell_size, fill_percent):
        self.grid_num = grid_num
        self.cell_size = cell_size
        self.fill_percent = fill_percent
        self.root = tk.Tk()
        self.root.title("The Hero's Jounery")
        canvas_dim = self.grid_num * self.cell_size
        self.canvas = tk.Canvas(
            self.root, width=canvas_dim, height=canvas_dim, bg="white"
        )
        self.canvas.pack(padx=10, pady=10)
        self._draw_grid()
        self.shaded_dict = {}

    def _draw_grid(self):
        """This function creates a canvas of size 128 x 128"""
        max_dim = self.grid_num * self.cell_size
        # Draw Vertical Lines
        for x in range(0, max_dim + self.cell_size, self.cell_size):
            self.canvas.create_line(x, 0, x, max_dim, fill="lightgray")
        # Draw Horizontal Lines
        for y in range(0, max_dim + self.cell_size, self.cell_size):
            self.canvas.create_line(0, y, max_dim, y, fill="lightgray")

    def color_cell(self, coordinate, color="black", shape="square"):
        """Color a cell with a specific shape"""
        row, col = coordinate
        if 0 <= row < self.grid_num and 0 <= col < self.grid_num:
            self.shaded_dict[color] = coordinate
            x1 = row * self.cell_size
            y1 = col * self.cell_size
            x2 = x1 + self.cell_size
            y2 = y1 + self.cell_size

            if shape == "square":
                self.canvas.create_rectangle(
                    x1, y1, x2, y2, fill=color, outline="lightgray"
                )

            elif shape == "triangle":
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                size = self.cell_size * 0.4  # controls how big the triangle is

                self.canvas.create_polygon(
                    cx,
                    cy - size,  # top
                    cx - size,
                    cy + size,  # bottom-left
                    cx + size,
                    cy + size,  # bottom-right
                    fill=color,
                    outline="lightgray",
                )

            elif shape == "circle":
                padding = self.cell_size * 0.1
                self.canvas.create_oval(
                    x1 + padding,
                    y1 + padding,
                    x2 - padding,
                    y2 - padding,
                    fill=color,
                    outline="lightgray",
                )

    def game_over_screen(self):
        """
        Function created using ChatGPT prompt in response to creating "game over" display
        """
        # Clear everything drawn on canvas
        self.canvas.delete("all")

        # Optional: disable further updates
        self.canvas.unbind_all("<Key>")

        # Center of the screen
        center_x = (self.grid_num * self.cell_size) // 2
        center_y = (self.grid_num * self.cell_size) // 2

        self.canvas.create_text(
            center_x,
            center_y,
            text="GAME OVER",
            fill="red",
            font=("Helvetica", 36, "bold"),
        )

    def Open_map(self):
        pass
        # self.root.mainloop()


if __name__ == "__main__":
    pass
    # new = Renderer(12,100,0.1)
    # new.Open_map()
