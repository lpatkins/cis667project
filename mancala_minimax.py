import numpy as np
import random
from mancala_helpers import *

# A simple evaluation function that simply uses the current score.
def simple_evaluate(state):
    return score_in(state)

# TODO
# Implement a better evaluation function that outperforms the simple one.
def better_evaluate(state):
    return max(minimax(state, 13, simple_evaluate)[1])

# depth-limited minimax as covered in lecture

#AI implementation: pruning is now involved with alpha (minimum) and beta (maximum)
def minimax(state, alpha, beta, max_depth, maximizer, evaluate):
    # returns chosen child state, utility

    # base cases
    if game_over(state): return None, score_in(state)
    if max_depth == 0: return None, evaluate(state)

    #Case of maximizing player (best-case scenario):
    if maximizer == True:
        best = -np.inf
        # recursive case
        children = [perform_action(action, state) for action in valid_actions(state)]
        for child in children:
            val = minimax(child, -np.inf, np.inf, max_depth - 1, False, evaluate)
            best = max(best, val)
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        results = [minimax(child, -np.inf, np.inf, max_depth-1, False, evaluate) for child in children]
    
    #Case of minimizing player (worst-case scenario):
    else:
        best = np.inf
        # recursive case
        children = [perform_action(action, state) for action in valid_actions(state)]
        for child in children:
            val = minimax(child, -np.inf, np.inf, max_depth - 1, True, evaluate)
            best = min(best, val)
            alpha = min(beta, best)
            if beta <= alpha:
                break
        results = [minimax(child, -np.inf, np.inf, max_depth-1, True, evaluate) for child in children]


    _, utilities = zip(*results)
    player, board = state
    if player == 0:
        action = np.argmax(utilities)
    if player == 1:
        action = np.argmin(utilities)
    return children[action], utilities[action]

# runs a competitive game between two AIs:
# better_evaluation (as player 0) vs simple_evaluation (as player 1)
def compete(max_depth, verbose=True):
    state = initial_state()
    while not game_over(state):

        player, board = state
        if verbose: print(string_of(board))
        if verbose: print("--- %s's turn --->" % ["Better","Simple"][player])
        state, _ = minimax(state, max_depth, [better_evaluate, simple_evaluate][player])
    
    score = score_in(state)
    player, board = state
    if verbose:
        print(string_of(board))
        print("Final score: %d" % score)
    
    return score
 

if __name__ == "__main__":
    
    #score = compete(max_depth=4, verbose=True)
    testingAI()

# Added for AI Project: represents the baseline/medium-level AI system.
def baselineAI(state):
    act = valid_actions(state)
    if len(act) == 1:
        return 0
    else:
        val = random.randInt(0, len(act) - 1)
        return val

#Five experiments to be run, each with their own variation.
def testingAI():
    print("New game! This time, 10 pits, 4 gems per pit, player 0 begins")
    for i in range(100):
        aiGame(10, 4, [0, 1])
    #Like the first test, only the addition of the pool pit
    print("New game! This time, 10 pits, 4 gems per pit, pool pit exists, player 0 begins")
    for i in range(100):
        aiGame(11, 4, [0, 1])
    #Changed the order of who is going first, so baseline will go first.
    print("New game! This time, 10 pits, 4 gems per pit, player 1 begins")
    for i in range(100):
        aiGame(10, 4, [1, 0])
    print("New game! This time, 22 pits, 4 gems per pit, pool pit exists, player 1 begins")
    #Larger size for both the pit count and gem count per pit.
    for i in range(100):
        aiGame(23, 4, [0, 1])
    print("New game! This time, 22 pits, 9 gems per pit, pool pit exists, player 0 begins")
    #Same standards as the last, just change in order
    for i in range(100):
        aiGame(23, 9, [0, 1])


def aiGame(pits: int, gems: int, players):
    state = initial_state(gems, pits, players)
        while not game_over(state):
            player, board = state
            print(stringof(board))
            if player == 0:
                print("Minimax's turn: ")
                state, _ = minimax(state, -np.inf, np.inf, 4, True, simple_evaluate)
            elif player == 1:
                print("Baseline AI's turn: ")
                base_act = baselineAI(state)
                state = perform_action(base_act, state)
        player, board = state
        print(stringof(board))
        if is_tied(board):
            print("Game over, it is tied.")
        winner = winner_of(board)
        print("Game over! ", winner, " wins!")
        print("Player 0 score: ", board[(pits // 2) - 1])
        if len(board) % 2 == 0:
            print("Player 1 score: ", board[pits - 1])
        else:
            print("Player 1 score: ", board[pits - 2])
