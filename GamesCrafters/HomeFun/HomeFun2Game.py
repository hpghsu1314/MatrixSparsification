#Tic-Tac-Toe
def DoMove(position, move):
    assert position[move[0]][move[1]] == None, "Position not valid"
    position = tup_to_list(position)
    empty = 0
    for row in position:
        for val in row:
            if val == None:
                empty += 1
    if empty % 2 == 1:
        position[move[0]][move[1]] = "O"
    else:
        position[move[0]][move[1]] = "X"
    position = list_to_tup(position)
    return position


def GenerateMoves(position):
    check_board(position)
    temp = []
    for row in range(len(position)):
        for val in range(len(position[row])):
            if position[row][val] == None:
                temp.append((row, val))
    return temp


def PrimitiveValue(position):
    lose = check_row(position) or check_col(position) or check_left_diag(position) or check_right_diag(position)
    if lose:
        return "Lost"
    else:
        for row in position:
            for value in row:
                if value == None:
                    return "Not Primitive"
        return "Draw"
    

def check_board(position):
    for row in position:
        assert len(row) == len(position), "Board not valid"

def check_row(position):
    for row in position:
        lose = True
        for i in range(len(row)-1):
            if row[i] != row[i+1] or row[i] == None:
                lose = False
                break
        if lose == True:
            return True
    return False

def check_col(position):
    for col in range(len(position[0])):
        lose = True
        for i in range(len(position[col])-1):
            if position[i][col] != position[i+1][col] or position[i][col] == None:
                lose = False
                break
        if lose == True:
            return True
    return False

def check_left_diag(position):
    lose = True
    for i in range(len(position)-1):
        if position[i][i] != position[i+1][i+1] or position[i][i] == None:
            lose = False
            break
    if lose == True:
        return True
    return False
    
def check_right_diag(position):
    lose = True
    for i in range(len(position)-1):
        if position[i][len(position)-i-1] != position[i+1][len(position)-2-i] or position[i][len(position)-i-1] == None:
            lose = False
            break
    if lose == True:
        return True
    return False

def list_to_tup(lst):
    return tuple(tuple(lst[i]) for i in range(len(lst)))

def tup_to_list(tup):
    return list(list(tup[i]) for i in range(len(tup)))

def flip_matrix(matrix):
    matrix = tup_to_list(matrix)
    for i in range(len(matrix)):
        matrix[i] = matrix[i][::-1]
    return list_to_tup(matrix)


init_pos = list_to_tup([[None for __ in range(3)] for __ in range(3)])

def check_symmetry(board):
    result = []
    temp = board[:]
    result.append(temp)
    for row in range(len(board)):
        temp[row] = temp[row::-1]
    result.append(temp)
    result = result + rotation(result[0]) + rotation(result[1])

def rotation(board):
    rotated_boards = []
    temp = [[None for __ in range(len(board))] for __ in range(len(board))]
    for __ in range(3):
        for r in range(len(board)):
            for c in range(len(board)):
                temp[c][len(board)-r-1] = board[r][c]
        rotated_boards.append(temp)
        board = temp[::]
            