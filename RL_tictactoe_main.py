# Tic tac toe model from scratch [only (standard + numpy)]

import numpy as np
import itertools
import random as rand 
from datetime import datetime

# Our agent and opponent for self play 
agent = 'x'
opponent = 'o'

# Generate boards, filter illegals
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

    """
     If the board state is won give it reward/win% == 1
     If lost == 0
     Else win% ==  0.1 (Pesimistic initial values)
     Why pesimistic? 
     We want the reward value to "flow" as fast as
     possible from the 100% winning stats in the late rounds 
     to the earlier states, but gradually slow down towards the 
     early states, which are naturally more "uncertain" to evaluate
    """
    
    for stc in combinations_to_check:
        values = [board[i, j] for i, j in stc] # Get values
        if all(v == player for v in values):
            return 1.0 # win
        if all(v == adversary for v in values):
            return 0.0 # lose
    return 0.1 # neither

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
episodes = 50000
learning_rate = 0.1
exploration_rate = 0.1

def move(player):
    # Get the possible next moves
    global round, current_state
    blank_indexes = [i for i, x in enumerate(current_state) if x == 'blank']
    possible_states = [[player if i == b_i else x for i, x in enumerate(current_state)] 
                       for b_i in blank_indexes]
    #  Main space search idea: exploitation / exploration tradeoff 
    roll = rand.uniform(0,1)
    if roll >= exploration_rate:
        new_state = max(possible_states, key=lambda p_s: table[player][tuple(p_s)])
    else:
        new_state = rand.choice(possible_states)
    current_state = new_state

pre_train_time = (datetime.now())
for ep in range(episodes):
    if ep % 100000 == 0 and ep > 0:
        learning_rate *= 0.9 # Small learning rate decay 
        print(f'Episode: {ep}/{episodes}')
        print(f'Training time: {datetime.now() - pre_train_time}')
        print(f'Progress: {ep/episodes}%')
    # Initialize board
    current_state = ['blank' for i in range(9)]
    agent_prev_state = None
    opponent_prev_state = None
    
    while True:
        move(agent) # Agent move
        # !Important! : update the last state evaluation AFTER the move of the opponent
        # Initially it was set update instantly after own move, but this leads
        # to updating actions independent of the player
        # since current state is variable is modified by enemy's move
        if agent_prev_state is not None:
            table[agent][tuple(agent_prev_state)] += learning_rate * (
                table[agent][tuple(current_state)] - table[agent][tuple(agent_prev_state)]
            )
        # Save for later
        agent_prev_state = current_state.copy()
        # 3. Check Win/Loss/Draw
        if table[agent][tuple(current_state)] == 1:
            # Punish (without explicit punish, the game stops, and thus 
            # enemy never actually learns from "giving" a win to the opponent)
            if opponent_prev_state is not None:
                table[opponent][tuple(opponent_prev_state)] += learning_rate * (
                    0.0 - table[opponent][tuple(opponent_prev_state)]
                )
            break
        if 'blank' not in current_state:
            break

        move(opponent) # Oppo turn
        if opponent_prev_state is not None:
            table[opponent][tuple(opponent_prev_state)] += learning_rate * (
                table[opponent][tuple(current_state)] - table[opponent][tuple(opponent_prev_state)]
            )  
        opponent_prev_state = current_state.copy()
        if table[opponent][tuple(current_state)] == 1:
            if agent_prev_state is not None:
                table[agent][tuple(agent_prev_state)] += learning_rate * (
                    0.0 - table[agent][tuple(agent_prev_state)]
                )
            break
        if 'blank' not in current_state:
            break
    

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
exploration_rate = 0

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






        
