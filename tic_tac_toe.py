# Tic tac toe AI trained from scratch [= without dedicated ML frameworks]

import numpy as np
import itertools
import random as rand 

agent = 'x'
opponent = 'o'

all_variations = list(itertools.product(['x', 'o', 'blank'], repeat=9))

valid_boards = [
    v for v in all_variations
    if abs(v.count('x') - v.count('o')) <= 1
]

def check_status(board_tuple):
    board = np.array(board_tuple, dtype=str).reshape(3, 3)
    series_to_check = [
        ((0,0),(0,1),(0,2)), ((1,0),(1,1),(1,2)), ((2,0),(2,1),(2,2)),
        ((0,0),(1,0),(2,0)), ((0,1),(1,1),(2,1)), ((0,2),(1,2),(2,2)),
        ((0,0),(1,1),(2,2)), ((0,2),(1,1),(2,0))
    ]
    for stc in series_to_check:
        values = [board[i, j] for i, j in stc]
        if all(v == agent for v in values):
            return 1.0
        if all(v == opponent for v in values):
            return 0.0
    return 0.5

table = {board: check_status(board) for board in valid_boards}

print(table)

# Now that we have the reward table, we need a training algorithm

"""
Start with matrix [0 = blank]: 
    0 0 0 
    0 0 0 
    0 0 0  
Pick the highest win% valid next move from the table in 9/10 cases
Pick random move in 1/10 cases
If it's 2nd or later move compare the rewards between current and previous move
Update the previous move based on the updated win%
END EPISODE 
NEW EPPISODE (as defined)
continue forever 
"""

episodes = 0
learning_rate = 0.02

for ep in range(episodes):
    round = 1
    current_state = ['blank' for i in range(9)]
    while round != 10:
        blank_indexes = [current_state.index[x] for x in current_state if
                         x == 'blank']
        possible_states = [current_state.replace(current_state[b_i], agent)
                            for b_i in blank_indexes]
        roll = rand.uniform(0,1)
        if roll >= 0.1:
            new_state = max(possible_states, key=lambda p_s: table[p_s])
        else:
            new_state = rand.choice(possible_states)
        table[current_state] += learning_rate*(table[new_state] - table[current_state])
        current_state = new_state
        round += 1

        



