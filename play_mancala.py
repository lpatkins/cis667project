from mancala_helpers import *
from mancala_minimax import minimax, simple_evaluate

def get_user_action(state):
    actions = list(map(str, valid_actions(state)))
    player, board = state
    prompt = "Player %d, choose an action (%s): " % (player, ",".join(actions))
    while True:
        action = input(prompt)
        if action in actions: return int(action)
        print("Invalid action, try again.")

if __name__ == "__main__":

    max_depth = 4
    pits, gems, players = begin_game()
    state = initial_state(gems, pits, players[0])
    while not game_over(state):

        player, board = state
        print(string_of(board))
        if player == 0:
            action = get_user_action(state)
            state = perform_action(action, state)
        elif player == 1: #Easy CPU
            print("--- AI's turn --->")
            state, _ = minimax(state, -np.inf, np.inf, max_depth, False, simple_evaluate)
        elif player == 2: #Medium CPU
            print("--- AI's turn --->")
            state, _ = baselineAI(state)
        else: #Hard CPU
            print("--- AI's turn --->")
            state, _ = minimax(state, -np.inf, np.inf, max_depth, True, simple_evaluate)
        
    player, board = state
    print(string_of(board))
    if is_tied(board):
        print("Game over, it is tied.")
    else:
        winner = winner_of(board)
        print("Game over, player %d wins." % winner)

