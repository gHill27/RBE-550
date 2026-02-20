import turtle

t = turtle.Turtle()
t.left(90)
for i in range(3):
    for i in range(3):
        t.forward(100)  # length of each side
        t.right(120)  # turn angle for a triangle
    t.left(120)

t.forward(100)
t.left(90)
t.circle(100)

turtle.done()
