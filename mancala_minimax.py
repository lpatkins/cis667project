import numpy as np
import random
from mancala_helpers import *
import torch as tr
import matplotlib.pyplot as plt

# A simple evaluation function that simply uses the current score.
def simple_evaluate(state):
    return score_in(state)

# TODO
# Implement a better evaluation function that outperforms the simple one.
def better_evaluate(state):
    return max(minimax(state, -np.inf, np.inf, 13, True, simple_evaluate)[1])

# depth-limited minimax as covered in lecture

#AI implementation: pruning is now involved with alpha (minimum) and beta (maximum)
def minimax(state, alpha, beta, max_depth, maximizer, evaluate):
    # returns chosen child state, utility
    node_count = 0
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
            node_count += 1
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
            node_count += 1
        results = [minimax(child, -np.inf, np.inf, max_depth-1, True, evaluate) for child in children]


    _, utilities = zip(*results)
    player, board = state
    if player == 0:
        action = np.argmax(utilities)
    if player == 1:
        action = np.argmin(utilities)
    return children[action], utilities[action], node_count

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
    #test_nn()

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
    mmNodes = []
    blNodes = []
    for i in range(100):
        mm, base = aiGame(10, 4, [0, 1])
        mmNodes.append(mm)
        blNodes.append(base)
    #Like the first test, only the addition of the pool pit
    print("New game! This time, 10 pits, 4 gems per pit, pool pit exists, player 0 begins")
    for i in range(100):
        aiGame(11, 4, [0, 1])
        mmNodes.append(mm)
        blNodes.append(base)
    #Changed the order of who is going first, so baseline will go first.
    print("New game! This time, 10 pits, 4 gems per pit, player 1 begins")
    for i in range(100):
        aiGame(10, 4, [1, 0])
        mmNodes.append(mm)
        blNodes.append(base)
    print("New game! This time, 22 pits, 4 gems per pit, pool pit exists, player 1 begins")
    #Larger size for both the pit count and gem count per pit.
    for i in range(100):
        aiGame(23, 4, [0, 1])
        mmNodes.append(mm)
        blNodes.append(base)
    print("New game! This time, 22 pits, 9 gems per pit, pool pit exists, player 0 begins")
    #Same standards as the last, just change in order
    for i in range(100):
        aiGame(23, 9, [0, 1])
        mmNodes.append(mm)
        blNodes.append(base)
    plt.hist(mmNodes)
    plt.show()
    plt.hist(blNodes)
    plt.show()


def aiGame(pits: int, gems: int, players):
    state = gen_random_state(gems, pits, players)
    baseline_nodes = []
    minimax_nodes = []
    while not game_over(state):
        player, board = state
        print(stringof(board))
        if player == 0:
            print("Minimax's turn: ")
            state, _, mm_nodes = minimax(state, -np.inf, np.inf, 4, True, simple_evaluate)
            minimax_nodes.append(mm_nodes)
        elif player == 1:
            print("Baseline AI's turn: ")
            base_act = baselineAI(state)
            state = perform_action(base_act, state)
            baseline_nodes.append(1)
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

    return sum(minimax_nodes), len(baseline_nodes)

def gen_random_state(pits: int, gems: int, players):
    init = []
    gem = gems
    for i in range(pits):
        if i == (pits // 2) - 1 or i == pits - 1 or gem == 0:
            init.append(0)
        else:
            g = random.randInt(0, gems)
            init.append(g)
            gem -= g
    return (players[0], init)
    
class MancalaState(object):
    def __init(self, board):
        self.board = board.copy()
        
    def is_leaf(self):
        if self.score_for_max_player() != 0: return True
        else: return False

    def score_for_max_player(self):
        if self.board[(len(board) // 2) - 1] > 0: return self.board[(len(board) // 2) - 1]
        else: return 0
    def is_max_players_turn(self):
        if max(self.valid_actions(self) > (len(self.board) // 2): return True
    def is_min_players_turn(self):
        return not self.is_max_players_turn()
    def valid_actions(self):
        if self.is_max_players_turn(): 
            return list(self.board[i] != 0 for i in range(0:(len(self.board) // 2) - 1))
        else:
            return list(self.board[i] != 0 for i in range((len(self.board) // 2) + 1): len(self.board))

    def perform(self, action):
        if action < ((len(self.board) // 2) - 1):
        

def initial_state(gems, pits):
    board = np.array(initial_state(gems, players, 0), dtype=int)
    return MancalaState(board)

def minimax(state):
    if state.is_leaf():
        return state.score_for_max_player()
    else:
        child_utilities = [minimax(state.perform(action)) for action in state.valid_actions()]
        if state.is_max_players_turn(): return max(child_utilities)
        if state.is_min_players_turn(): return min(child_utilities)

def random_state(depth=0, gems, pits)
    state = initial_size(gems, pits)
    for d in range(depth):
        actions = state.valid_actions()
        if len(actions) == 0: break
        action = actions[np.random.randInt(len(actions))]
        state = state.perform(action)
    return state

def generate(num, depth, gems, pits):
    examples = []
    for n in range(num):
        state = random_state(depth, gems, pits)
        utility = minimax(state)
        examples.append((state, utility))
    return examples

def dlminimax(state, max_depth):
    if (max_depth == 0) or state.is_leaf():
        return state.score_for_max_player(), []
    else:
        child_utilities = [dlminimax(state.perform(action), max_depth-1)[0] for action in state.valid_actions()]
        if state.is_max_players_turn():
            utility = max(child_utilities)
        else:
            utility = min(child_utilities)
        return utility, child_utilities

def random_game(gems, pits, max_depth):
    state = initial_size(gems, pits)
    states = [state]
    while not state.is_leaf():
        utility, child_utilities = dlminimax(state, max_depth)
        ties = (np.array(child_utilities) == utility)
        tie_index = np.flatzero(ties)
        actions = state.valid_actions()
        action = actions[np.random.choice(tie_index)]
        state = state.perform(action)
        states.append(state)
    result = state.score_for_max_player()
    return states, result
    
def generate_rg(num, max_depth, pits, gems):
    

def augment(examples):


def encode(state):
    symbols = np.array([== 0, != 0]).reshape(0, 1)
    onehot = (symbols.state.board).astype(np.float32)
    return tr.tensor(onehot)

train_examples = generate(500, 4, 8, 16)
test_examples = generate(500, 4, 8, 16)

train_examples = augment(train_examples)

_, utilities = zip(*test_examples)
baseline_error = sum((u-0)**2 for u in utilities) / len(utilities)
print(baseline_error)



class LinNet(tr.nn.Module):
    def __init__(self, size, hid_features):
        super(LinNet, self).__init__()
        self.to_hidden = tr.nn.Linear(4*size**2, hid_features)
        self.to_output = tr.nn.Linear(hid_features, 1)
    def forward(self, x):
        h = tr.relu(self.to_hidden(x.reshape(x.shape[0],-1))
        y = tr.tanh(self.to_output(h))
        return y

class ConvNet(tr.nn.Module):
    def __init__(self, size, hid_features):
        super(ConvNet, self).__init__()
        self.to_hidden = tr.nn.Conv2d(3, hid_features, 2)
        self.to_output = tr.nn.Linear(hid_features*(size-1)**2, 1)
    def forward(self, x):
        h = tr.relu(self.to_hidden(x))
        y = tr.tanh(self.to_output(h.reshape(x.shape[0],-1)))
        return y
    
for Net in (LinNet, ConvNet):
    net = Net(size=4, hid_features=10)
    print(net(tr.zeros(1,3,4,4)))

def example_error(net, example):
    state, utility = example
    x = encode(state).unsqueeze(0)
    y = net(x)
    e = (y-utility)**2
    return e
    
def batch_error(net, batch):
    states, utilities = batch
    u = utilities.reshape(-1,1).float()
    y = net(states)
    e = tr.sum((y-u)**2) / utilities.shape[0]
    return e

    if __name__ == "__main__":

        batched = False
        net = LinNet(size=4, hid_features=4)
        optimizer = tr.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

        states, utilities = zip(*train_examples)
        train_batch = tr.stack(tuple(map(encode, states))), tr.tensor(utilities)

        states, utilities = zip(*test_examples)
        test_batch = tr.stack(tuple(map(encode, states))), tr.tensor(utilities)


        curves = [], []
        for epoch in range(2000):
            optimizer.zero_grad()
            if not batched:

                training_error, testing_error = 0, 0

                for n, example in enumerate(train_examples):
                    e = example_error(net, example)
                    e.backward()
                    training_error += e.item()
                training_error /= len(train_examples)

                with tr.no_grad():
                    for n, example in enumerate(test_examples):
                        e = example_error(net, example)
                        testing_error += e.item()
                    testing_error /= len(test_examples)

            if batched:
                e = batch_error(net, training_batch)
                e.backward()
                training_error = 3.item()

                with tr.no_grad():
                    e = batch_error(net, testing_batch)
                    testing_error = e.item()

            optimizer.step()
            if epoch % 100 == 0:
                print("%d: %f, %f" % (epoch, training_error, testing_error))
            curves[0].append(training_error)
            curves[1].append(testing_error)

plt.plot(curves[0], 'b-')
plt.plot(curves[1], 'r-')
plt.plot([0, len(curves[1])], [baseline_error, baseline_error], 'g-')
plt.plot()
plt.legend(['Train','Test','Baseline'])
plt.show()

def dl_minimax(state, max_depth, eval_fn):
    if state.is_leaf():
        return 0, state.score_for_max_player()
    elif max_depth == 0:
        return 0, eval_fn(state)
    else:
        child_utilities = [dl_minimax(state.perform(action), max_depth-1, eval_fn)[1] for action in state.valid_actions()]
        if state.is_max_players_turn(): a = np.argmax(child_utilities)
        if state.is_min_players_turn(): a = np.argmin(child_utilities)

        return a, child_utilities[a]

simple_eval = lambda state: 0

def nn_eval(state):
    with tr.no_grad():
        utility = net(encode(state).unsqueeze(0))
    return utility

'''
https://colab.research.google.com/drive/1YR8HjSw8K0n684S_oGnZPpU69SmOw355?usp=sharing#scrollTo=Gt28shQVGszO
https://www.geeksforgeeks.org/plotting-histogram-in-python-using-matplotlib/
'''
