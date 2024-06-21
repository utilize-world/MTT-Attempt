import utils

x = 10
y = 20
x1 = x
y1 = y


for t in range(100):
    if t % 10 == 0:
        x1, y1 = utils.randomWalk(x1, y1, 2, 5)
        print(f"x1: {x1}, y1:{y1}")
