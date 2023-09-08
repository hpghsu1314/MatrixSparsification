#Solver
import HomeFun2Game as Game

memo = {}

symmetry = True

opposites = {"Win the Game": "Lose the Game", "Draw the Game": "Draw the Game", "Lose the Game": "Win the Game"}
prim_value_chart = {"Lost": "Lose the Game", "Win": "Win the Game", "Draw": "Draw the Game"}
total_prim = {"Win": 0, "Lost": 0, "Draw": 0, "Total" : 0}


def check_res(results):
    win = "Win the Game"
    draw = "Draw the Game"
    lose = "Lose the Game"
    return win if win in results else draw if draw in results else lose

def Solve(position):
    prim_value = Game.PrimitiveValue(position)
    if prim_value != "Not Primitive":
        total_prim[prim_value] += 1
        total_prim["Total"] += 1
        return prim_value_chart[prim_value]
    else:
        legal_moves = Game.GenerateMoves(position)
        result = []
        for move in legal_moves:
            new_pos = Game.DoMove(position, move)
            if new_pos not in memo.keys():
                res = Solve(new_pos)
                memo.update({new_pos: res})
                result.append(opposites[res])
            else:
                pos = new_pos
                result.append(opposites[memo[pos]])
        
        curr_cond = check_res(result)
        memo.update({position: curr_cond})
        return curr_cond

init_pos = Game.init_pos
Solve(init_pos)

draw = 0
win = 0
lose = 0
for i in memo.keys():
    if memo[i] == "Draw the Game":
        draw += 1
    elif memo[i] == "Win the Game":
        win += 1
    else:
        lose += 1

print(f"Lose = {lose} with {total_prim['Lost']} primitive \nWin = {win} with {total_prim['Win']} primitive\nDraw = {draw} with {total_prim['Draw']} primitive")
print(f"Total = {len(memo)} with {total_prim['Total']} primitive")