#Solver
import HomeFun1Game as Game

def Solve(position):
    if Game.PrimitiveValue(position) == "Lost":
        return "Lose the Game"
    else:
        legal_moves = Game.GenerateMoves(position)
        result = []
        for move in legal_moves:
            new_pos = Game.DoMove(position, move)
            if Solve(new_pos) == "Lose the Game":
                result.append("Win the Game")
            else:
                result.append("Lose the Game")
        if "Win the Game" in result:
            return "Win the Game"
        else:
            return "Lose the Game"

tokens = 10 #Change this value to see {token} positions

for token in range(tokens + 1):
    print(f"When {token} tokens, {Solve(token)}")
