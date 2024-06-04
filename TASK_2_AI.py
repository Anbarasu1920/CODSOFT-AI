import math

def print_board(board):
    for row in board:
        print(" | ".join(row))
        print("-" * 5)

def check_winner(board, player):
    win_conditions = [
        [board[0][0], board[0][1], board[0][2]],
        [board[1][0], board[1][1], board[1][2]],
        [board[2][0], board[2][1], board[2][2]],
        [board[0][0], board[1][0], board[2][0]],
        [board[0][1], board[1][1], board[2][1]],
        [board[0][2], board[1][2], board[2][2]],
        [board[0][0], board[1][1], board[2][2]],
        [board[2][0], board[1][1], board[0][2]]
    ]
    return [player, player, player] in win_conditions

def check_draw(board):
    return all(cell != " " for row in board for cell in row)

def get_available_moves(board):
    moves = []
    for i in range(3):
        for j in range(3):
            if board[i][j] == " ":
                moves.append((i, j))
    return moves

def minimax(board, depth, is_maximizing, alpha, beta):
    if check_winner(board, "X"):
        return -1
    if check_winner(board, "O"):
        return 1
    if check_draw(board):
        return 0

    if is_maximizing:
        max_eval = -math.inf
        for move in get_available_moves(board):
            board[move[0]][move[1]] = "O"
            eval = minimax(board, depth + 1, False, alpha, beta)
            board[move[0]][move[1]] = " "
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = math.inf
        for move in get_available_moves(board):
            board[move[0]][move[1]] = "X"
            eval = minimax(board, depth + 1, True, alpha, beta)
            board[move[0]][move[1]] = " "
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

def get_best_move(board):
    best_move = None
    best_value = -math.inf
    for move in get_available_moves(board):
        board[move[0]][move[1]] = "O"
        move_value = minimax(board, 0, False, -math.inf, math.inf)
        board[move[0]][move[1]] = " "
        if move_value > best_value:
            best_value = move_value
            best_move = move
    return best_move

def main():
    board = [[" " for _ in range(3)] for _ in range(3)]
    human_player = "X"
    ai_player = "O"
    current_player = "X"

    while True:
        print_board(board)
        if current_player == human_player:
            move = input("Enter your move (row and column): ").split()
            row, col = int(move[0]), int(move[1])
            if board[row][col] == " ":
                board[row][col] = human_player
                if check_winner(board, human_player):
                    print_board(board)
                    print("Congratulations! You win!")
                    break
                current_player = ai_player
        else:
            move = get_best_move(board)
            if move:
                board[move[0]][move[1]] = ai_player
                if check_winner(board, ai_player):
                    print_board(board)
                    print("AI wins! Better luck next time.")
                    break
                current_player = human_player

        if check_draw(board):
            print_board(board)
            print("It's a draw!")
            break


main()
