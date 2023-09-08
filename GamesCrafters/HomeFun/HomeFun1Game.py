moves = [1,2] #Change this to express the valid/legal moves for each turn

def DoMove(position, move):
    assert position > 0, "Position is primitive"
    assert position - move >= 0 and move in moves, "Move is illegal"
    return position - move


def GenerateMoves(position):
    assert position > 0, "Position is primitive"
    temp = []
    for i in moves:
        if i <= position:
            temp.append(i)
    return temp


def PrimitiveValue(position):
    return "Lost" if position == 0 else "Not Primitive"

init_pos = 10