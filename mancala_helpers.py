# ********
# This file is individualized for NetID lpatkins.
# ********

# TODO: implement pad(num)
# Return a string representation of num that is always two characters wide.
# If num does not already have two digits, a leading " " is inserted in front.
# This is called "padding".  For example, pad(12) is "12", and pad(1) is " 1".
# You can assume num is either one or two digits long.
def pad(num: int) -> str:
    if num < 10: 
        return " " + str(num) 
    else: 
        return str(num)

# TODO: implement pad_all(nums)
# Return a new list whose elements are padded versions of the elements in nums.
# For example, pad_all([12, 1]) should return ["12", " 1"].
# Your code should create a new list, and not modify the original list.
# You can assume each element of nums is an int with one or two digits.
def pad_all(nums: list) -> list:
    padList = []
    for i in range(len(nums)):
        padList.append(pad(nums[i]))
    return padList

# TODO: implement initial_state()
# Return a (player, board) tuple representing the initial game state
# The initial player is player 0.
# board is list of ints representing the initial mancala board at the start of the game.
# The list element at index p should be the number of gems at position p.

#AI changes: user prompt will now initialize the board.
def initial_state(gems: int, pits: int, player: int) -> tuple:
    init = []
    for i in range(0: (pits // 2) - 1):
        init.append(gems)
    init.append(0) #Player 0's Mancala
    for i in range(0:(pits // 2 - 1)):
        init.append(gems)
    init.append(0) #Player 1's Mancala
    if pits % 2 == 1:
        init.append(0) #Pool pit (odd number of pits)    
    return (player, init)

# TODO: implement game_over(state)
# Return True if the game is over, and False otherwise.
# The game is over once all pits are empty.
# Your code should not modify the board list.
# The built-in functions "any" and "all" may be useful:
#   https://docs.python.org/3/library/functions.html#all
def game_over(state: tuple) -> bool:
    if all(val == 0 for val in state[1][0:(len(state[1] // 2) - 1)]) and all(val == 0 for val in state[1][len(state[1] // 2):len(state[1]) - 1]):
        return True
    else:
        return False

# TODO: implement valid_actions(state)
# state is a (player, board) tuple
# Return a list of all positions on the board where the current player can pick up gems.
# A position is a valid move if it is one of the player's pits and has 1 or more gems in it.
# For example, if all of player's pits are empty, you should return [].
# The positions in the returned list should be ordered from lowest to highest.
# Your code should not modify the board list.

#AI changes: updated version accomodates for the custom board size
def valid_actions(state: tuple) -> list:
    valid = []
    if state[0] == 0:
        for i in range(len(state[1]) // 2):
            if state[1][i] > 0:
                valid.append(i)
    elif state[0] == 1:
        if len(state[1]) % 2 == 0: #No pool
            for i in range((len(state[1]) // 2) - 1: len(state[1]) - 1):
                if state[1][i] > 0:
                    valid.append(i)
        else:
            for i in range((len(state[1]) // 2) - 1: len(state[1]) - 2):
                if state[1][i] > 0:
                    valid.append(i)
    return valid
        

# TODO: implement mancala_of(player)
# Return the numeric position of the given player's mancala.
# Player 0's mancala is on the right and player 1's mancala is on the left.
# You can assume player is either 0 or 1.

#AI changes: now accomodates for the custom board size.
def mancala_of(player: int, board: list) -> int:
    if player == 0:
        return len(board) // 2
    else:
        if len(board) % 2 == 0:
            return len(board) - 1
        else:
            return len(board) - 2

# TODO: implement pits_of(player)
# Return a list of numeric positions corresponding to the given player's pits.
# The positions in the list should be ordered from lowest to highest.
# Player 0's pits are on the bottom and player 1's pits are on the top.
# You can assume player is either 0 or 1.

#AI changes: now accomodates for custom board size
def pits_of(player: int, board: list) -> list:
    pitList = []
    if len(board) % 2 == 0: #No pool pit
        if player == 0:
            for i in range(0: len(board) // 2):
                pitList.append(i)
        else:
            for i in range((len(board) // 2): len(board) - 1):
                pitList.append(i)
    else: #Pool pit present (last value in board)
        if player == 0:
            for i in range(0: (len(board) // 2) - 1):
                pitList.append(i)
        else:
            for i in range((len(board) // 2): len(board) - 2):
                pitList.append(i)
    return pitList 

# TODO: implement player_who_can_do(move)
# Return the player (either 0 or 1) who is allowed to perform the given move.
# The move is allowed if it is the position of one of the player's pits.
# For example, position 2 is one of player 0's pits.
# So player_who_can_do(2) should return 0.
# You can assume that move is a valid position for one of the players.

#AI Changes: now accomodates for custom board size
def player_who_can_do(move: int, board: list) -> int:
    if move <= (len(board) // 2): 
        return 0 
    else: 
        return 1

# TODO: implement opposite_from(position)
# Return the position of the pit that is opposite from the given position.
# Check the pdf instructions for the definition of "opposite".

#AI changes: now accomodates for custom board size.
def opposite_from(position: int, board: list) -> int:
    if len(board) // 2 == 0:
        return len(board) - 2 - position
    else:
        return len(board) - 3 - position

# TODO: implement play_turn(move, board)
# Return the new game state after the given move is performed on the given board.
# The return value should be a tuple (new_player, new_board).
#   new_player should be the player (0 or 1) whose turn it is after the move.
#   new_board should be a list representing the new board state after the move.
#
# Parameters:
#   board is a list representing the current state of the game board before the turn is taken.
#   move is an int representing the position where the current player picks up gems.
# You can assume that move is a valid move for the current player who is taking their turn.
# Check the pdf instructions for the detailed rules of taking a turn.
#
# It may be helpful to use several of the functions you implemented above.
# You will also need control flow such as loops and if-statements.
# Lastly, the % (modulo) operator may be useful:
#  (x % y) returns the remainder of x / y
#  from: https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex

#AI changes: now accomodates for custom board size and inclusion of pool pit if need be.
def play_turn(move: int, board: list) -> tuple:

    # Make a copy of the board before anything else
    # This is important for minimax, so that different nodes do not share the same mutable data
    # Your code should NOT modify the original input board or else bugs may show up elsewhere
    board = list(board)
    move_count = 0
    gems_in_hand = board[move]
    board[move] = 0
    if move < (len(board) // 2) - 1: #indicates that it's player 0's turn
        while gems_in_hand > 0:
            move += 1
            if move == (len(board) - 1) and (len(board) % 2 == 0):
                move = 0
            #Pool pit processing for player 0 (adding gem to pool)
            if len(board) % 2 == 1 and move == (len(board) // 2) + ((len(board) // 2) // 2):
                board[len(board) - 1] += 1 #Pool pit (last index of board)
                gems_in_hand -= 1          #Loses gem, doesn't count for move_count
            board[move] += 1
            gems_in_hand -= 1
            move_count += 1
            #Pool pit processing if gems_in_hand is 0 but satisfies requirement to play pool pit
            if (len(board) % 2 == 1) and (gems_in_hand == 0) and (board[len(board) - 1] > 0) and (move_count >= (((len(board) // 2) - 1) * 1.5)):
                gems_in_hand += board[len(board) - 1]
                board[len(board) - 1] = 0
                move = 0 #Set so next increment, move = 1, would go from there
                move_count = 0 #resets the condition; otherwise, it'll remain player 0's turn for the rest of the game
            #if gems_in_hand == 0 and board[move] > 1:
            #    gems_in_hand += board[move]
            #    board[move] = 0
            #if board[move] > 1:
                #gems_in_hand += board[move]
                #board[move] = 0
        if board[opposite_from(move)] > 0 and board[move] == 1 and move != (len(board) // 2) - 1:
            board[(len(board) // 2) - 1] += board[move]
            board[(len(board) // 2) - 1] += board[opposite_from(move)]
            board[move] = 0
            board[opposite_from(move)] = 0
        if move == (len(board) // 2) - 1:
            next = (0, board)
        else:
            next = (1, board)
        string_of(board)
    else:
        while gems_in_hand > 0:
            move += 1
            if move == (len(board) - 1) and (len(board) % 2 == 0):
                move = 0
            if move == (len(board) // 2) - 1:
                move = len(board) // 2
            board[move] += 1
            gems_in_hand -= 1
            #Pool pit processing:
            if move == ((len(board) // 2) - 1) // 2 and len(board) % 2 == 1:
                board[len(board) - 1] += 1 #Pool pit (last index of board)
                gems_in_hand -= 1          #Loses gem, doesn't count for move_count
            #if gems_in_hand == 0 and board[move] > 1:
            #    gems_in_hand += board[move]
            #    board[move] = 0
            if (len(board) % 2 == 1) and (gems_in_hand == 0) and (board[len(board) - 1] > 0) and (move_count >= (((len(board) // 2) - 1) * 1.5)):
                gems_in_hand += board[len(board) - 1]
                board[len(board) - 1] = 0
                move = (len(board) // 2) - 1 #Set so next increment, move = len(board) // 2, would go from there
        if board[opposite_from(move)] > 0 and board[move] == 1 and ((move != len(board) - 1)):
            board[] += board[move]
            board[len(board)] += board[opposite_from(move)]
            board[move] = 0
            board[opposite_from(move)] = 0
        if ((move == len(board) - 1) and (len(board) // 2 == 0)) or ((move == len(board) - 2) and (len(board) // 2 == 1)):
            next = (1, board)
        else:
            next = (0, board)
        string_of(board)
        next = (0, board)
    return next

# TODO: implement clear_pits(board)
# Return a new list representing the game state after clearing the pits from the board.
# When clearing pits, any gems in a player's pits get moved to that player's mancala.
# Check the pdf instructions for more detail about clearing pits.
def clear_pits(board: list) -> list:
    #Clearing pits without pool pit
    if (len(board) % 2 == 0):
        for i in range(len(board)):
            if i < (len(board) // 2): 
                board[(len(board) // 2)] += board[i]
                board[i] = 0
            elif (i == (len(board) // 2) or i == (len(board) - 1)): 
                board[i] += 0 
            elif i > (len(board) // 2): 
                board[(len(board) - 1)] += board[i]
                board[i] = 0
    #Clearing pits with pool pit (pool doesn't allocate to anyone)
    else:
        for i in range(len(board)):
            if i < (len(board) // 2): 
                board[(len(board) // 2) - 1] += board[i]
                board[i] = 0
            elif (i == (len(board) // 2 - 1) or i == (len(board) - 2)): 
                board[i] += 0 
            elif i > (len(board) // 2): 
                board[len(board) - 2] += board[i]
                board[i] = 0
    return board


# This one is done for you.
# Plays a turn and clears pits if needed.
def perform_action(action, state):
    player, board = state
    new_player, new_board = play_turn(action, board)
    if 0 in [len(valid_actions((0, new_board))), len(valid_actions((1, new_board)))]:
        new_board = clear_pits(new_board)
    return new_player, new_board

# TODO: implement score_in(state)
# state is a (player, board) tuple
# Return the score in the given state.
# The score is the number of gems in player 0's mancala, minus the number of gems in player 1's mancala.
def score_in(state: tuple) -> int:
    if len(board) % 2 == 0:
        return state[1][len(state[1]) // 2] - state[1][len(state[1]) - 1]
    else:
        return state[1][(len(state[1]) // 2) - 1] - state[1][len(state[1]) - 2]

# TODO: implement is_tied(board)
# Return True if the game is tied in the given board state, False otherwise.
# A game is tied if both players have the same number of gems in their mancalas.
# You can assume all pits have already been cleared on the given board.
def is_tied(board: list) -> bool:
    if len(board) % 2 == 0:
        return board[len(board) // 2] == board[len(board) - 1]
    else:
        return board[(len(board) // 2) - 1] == board[len(board) - 2]

# TODO: implement winner_of(board)
# Return the winning player (either 0 or 1) in the given board state.
# The winner is the player with more gems in their mancala.
# You can assume it is not a tied game, and all pits have already been cleared.

#AI changes: now attribute to the length of the board instead of being fixed on one value.
def winner_of(board: list) -> int:
    if len(board) % 2 == 0:
        if board[len(board) // 2] > board[len(board) - 1]:
            return 0
        else:
            return 1
    else:
        if board[(len(board) // 2) - 1] > board[len(board) - 2]:
            return 0
        else:
            return 1

#Added for human to initialize game settings
def begin_game():
    
    while True:
        try:
            pits = int(input("Please enter the number of pits for both players (between 5 and 31): "))
            if pits < 5 or pits > 31:
                print("Invalid number of pits entered.")
            else:
                break
        except:
            print("Invalid")
            continue

    while True:
        try:
            gems = int(input("Please enter the number of gems for each pit (between 1 and half the number of pits): "))
            if gems < 1 or gems > (pits // 2):
                print("Invalid number of gems entered.")
            else:
                break
        except:
            print("Invalid")
            continue
    players = []
    while True:
        try:
            play = int(input("Please enter 1 if you want to play, or 0 if you don't want to play: "))
            if play != 1 or play != 0:
                print("Invalid player entered.")
            else:
                break
        except:
            print("Invalid")
            continue
    player1 = play:
    while True:
        try:
            play2 = int(input("If you want to play against a human opponent, please press 1. If you want to play against a CPU with 'Easy' difficulty. please enter 2. If you want to play against a CPU with 'Medium' difficulty. please enter 3. Or if you want to play against a CPU with 'Hard' difficulty. please enter 4."))
            if play != 1 or play != 2 or play != 3 or play != 4:
                print("Invalid player entered.")
            else:
                break
        except:
            print("Invalid")
            continue
    player2 = play2
    while True:
        try:
            seq = int(input("Do you want to go first? Type 1 if yes, or 0 if no."))
            if seq != 1 or seq != 0:
                print("Invalid player entered.")
            else:
                break
        except:
            print("Invalid")
            continue
    if seq == 1:
        players.append(player1)
        players.append(player2)
    else:
        players.append(player2)
        players.append(player1)
    return pits, gems, players
    
# TODO: implement string_of(board)
# Return a string representation of the given board state for text-based game play.
# The string should have three indented lines of text.
# The first line shows the number of gems in player 1's pits,
# The second line shows the number of gems in each player's mancala,
# And the third line shows the number of gems in player 0's pits.
# The gem numbers should be padded and evenly spaced.
# For example, the string representation of the initial game state is:
# 
#             5  5  5  5  5  5  5
#          0                       0
#             5  5  5  5  5  5  5
# 
# Another example for a different game state with more gems is:
# 
#            12 12 12 12 12 12 12
#          0                       0
#             5  5  5  5  5  5  5
# 
# Excluding the leading comment symbols "# " above, all blank space should match exactly:
#   There are exactly 8 blank spaces before the left (padded) mancala number.
#   There is exactly 1 blank space between each (padded) pit number.
#   The returned string should start and end with new-line characters ("\n")
def string_of(board: list) -> str:
    pad = pad_all(board)
    if len(pad) % 2 == 1:
            print("Gems in Pool pit: ")
            print(pad[len(pad)-1])
    for i in range(len(pad) - 1):
        if (len(pad) - 1 - i == len(pad) - 1) and len(pad) % 2 == 1:
            print("Player 2's Score: ")
            print(pad[len(pad) - 2]) #Accomodates presence of pool pit (2 less)
        elif (len(pad) - 1 - i == len(pad) - 1):
            print("Player 2's Score: ") #No pool pit (1 less)
            print(pad[len(pad) - 1])
        print("Player 2's pits (from closest to mancala to farthest): "
        elif (len(pad) - i > (len(pad) // 2)): #Displays all of Player 2's pits and their values
            print(pad[len(pad) - 1 - i], end=" ")
        elif len(pad) - 1 - i == (len(pad) // 2) - 1: #Displays Player 1's score
            print("Player 1's Score: ")
            print(pad[len(pad) - 1 - i])
        print("Player 1's pits (from closest to mancala to farthest): "
        elif len(pad) - 1 - i < (len(pad) // 2) - 1 and (len(pad) - 1 - i) >= 0:
            print(pad[len(pad) - 1 - i], end=" ")
        
            
        
    #return "\n" + "           " + pad[14] + " " + pad[13] + " " + pad[12] + " " + pad[11] + " " + pad[10] + " " + pad[9] + " " + pad[8] + "\n" + "        " + pad[15] + "                      " + pad[7] + "\n" + "           " + pad[0] + " " + pad[1] + " " + pad[2] + " " + pad[3] + " " + pad[4] + " " + pad[5] + " " + pad[6] + "\n" 

