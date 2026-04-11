# Tic tac to model from scratch (apart from standard + numpy)

import numpy as np
import itertools
import random as rand 
from datetime import datetime

agent = 'x'
opponent = 'o'

# Generate boards, filter illegal boards
all_variations = list(itertools.product(['x', 'o', 'blank'], repeat=9))
valid_boards = [
    v for v in all_variations
    if abs(v.count('x') - v.count('o')) <= 1
]

# Initial evaluaton of game states
def check_status(board_tuple, player, adversary):
    board = np.array(board_tuple, dtype=str).reshape(3, 3)
    # Possible winning combinations
    combinations_to_check = [
        ((0,0),(0,1),(0,2)), ((1,0),(1,1),(1,2)), ((2,0),(2,1),(2,2)),
        ((0,0),(1,0),(2,0)), ((0,1),(1,1),(2,1)), ((0,2),(1,2),(2,2)),
        ((0,0),(1,1),(2,2)), ((0,2),(1,1),(2,0))
    ]
    for stc in combinations_to_check:
        values = [board[i, j] for i, j in stc] # Get values
        if all(v == player for v in values):
            return 1.0 # win
        if all(v == adversary for v in values):
            return 0.0 # lose
        return 0.5 # neither

table_agent = {board: check_status(board, agent, opponent) for board in valid_boards}
table_opponent = {board: check_status(board, opponent, agent) for board in valid_boards}
table = {agent: table_agent, opponent: table_opponent}

# We have the policy table -> time for a training loop
"""
1) Start with matrix [0 = blank]: 
    0 0 0 
    0 0 0 
    0 0 0  
2) Pick the highest win% valid next move from the table in 9/10 cases
3) Pick random move in 1/10 cases
4) Compare the rewards between current and previous move
5) Update the previous move based on the updated win%
6) Play till someone wins or the space runs out  
7) NEW EPPISODE (as defined ^)
8) Continue as long as necessary  
"""
# Training parameters
episodes = 5000000
learning_rate = 0.2
exploration_rate = 0.125
# Training globals
round = None
current_state = None

def move(player):
    global round, current_state
    blank_indexes = [i for i, x in enumerate(current_state) if x == 'blank']
    if not blank_indexes:  # Exception (because of round != 1- condition in the training)
        return
    possible_states = [[player if i == b_i else x for i, x in enumerate(current_state)] 
                       for b_i in blank_indexes]
    roll = rand.uniform(0,1)
    if roll >= exploration_rate:
        new_state = max(possible_states, key=lambda p_s: table[player][tuple(p_s)])
    else:
        new_state = rand.choice(possible_states)
    table[player][tuple(current_state)] += learning_rate*(table[player][tuple(new_state)] - table[player][tuple(current_state)])
    current_state = new_state
    round += 1

pre_train_time = (datetime.now())
for ep in range(episodes):
    current_state = ['blank' for i in range(9)]
    round = 1
    while round != 10:
        move(agent)
        if table[agent][tuple(current_state)] == 1:
            break 
        move(opponent)
        if table[opponent][tuple(current_state)] == 1:
            break
    if ep % 10000 == 0:
        print(f'Episode: {ep}/{episodes}')
        print(f'Training time: {datetime.now() - pre_train_time}')
    if ep == episodes - int(98/100*episodes):  # For the end episodes, tone down the learning and exploration rate
        learning_rate = 0.1
        exploration_rate = 0.05
    

def print_board(state):
    symbols = {'x': 'X', 'o': 'O', 'blank': '.'}
    board = [symbols[s] for s in state]
    print(f"\n {board[0]} | {board[1]} | {board[2]}")
    print(f"---|---|---")
    print(f" {board[3]} | {board[4]} | {board[5]}")
    print(f"---|---|---")
    print(f" {board[6]} | {board[7]} | {board[8]}")
    print()

def human_move(state):
    blank_indexes = [i for i, x in enumerate(state) if x == 'blank']
    while True:
        try:
            idx = int(input("Your move (1-9): ")) - 1
            if idx in blank_indexes:
                state[idx] = opponent
                return state
            else:
                print("Invalid move, try again.")
        except ValueError:
            print("Enter a number between 1 and 9.")

def check_winner(state):
    combos = [
        (0,1,2),(3,4,5),(6,7,8),
        (0,3,6),(1,4,7),(2,5,8),
        (0,4,8),(2,4,6)
    ]
    for a,b,c in combos:
        if state[a] == state[b] == state[c] != 'blank':
            return state[a]
    if 'blank' not in state:
        return 'draw'
    return None

print("\nBoard positions:")
print(" 1 | 2 | 3")
print("---|---|---")
print(" 4 | 5 | 6")
print("---|---|---")
print(" 7 | 8 | 9")

current_state = ['blank'] * 9
round = 1
just_played = agent

while True:
    # Agent moves
    move(agent)
    print_board(current_state)
    winner = check_winner(current_state)
    if winner:
        print("Agent wins!" if winner == agent else "Draw!")
        break

    # Human moves
    current_state = human_move(current_state)
    print_board(current_state)
    winner = check_winner(current_state)
    if winner:
        print("You win!" if winner == opponent else "Draw!")
        break




        
