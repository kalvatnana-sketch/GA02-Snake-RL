#from game_environment_parallel import Snake
from game_environment import SnakeNumpy as Snake #changed imports because there is no game_environment_parallel

import numpy as np
env = Snake(board_size=10, frames=2, games=3)
s = env.reset()
action = np.array([0, 0, 0], dtype=np.int64)
s2, r, done, info, legal = env.step(action)

'''
done = 0
while(not done):
    # action = np.random.choice([-1, 0, 1], 1)[0]
    # instead of random action, take input from user
    action = int(input('Enter action [-1, 0, 1] : '))
    # print(action)
    s, r, done, info = env.step(action)
    # print(env._snake_direction)
    # for i, x in enumerate(env._snake):
        # print(i, x.row, x.col)
    env.print_game()
'''