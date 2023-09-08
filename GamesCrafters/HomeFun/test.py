import HomeFun2Game as Game
test_cases = [[["O", "O", "O"],
              [None, "X", None],
              [None, "X", None]],
              
              [[None, "X", None],
               ["O", "O", "O"],  
               [None, "X", None]],
              
              [[None, "X", "X"],
               [None, "X", None],
               ["O", "O", "O"]],
              
              [["O", "X", "X"],
               ["O", "X", None],
               ["O", None, "O"]],
              
              [[None, "O", "X"],
               ["X", "O", None],
               ["X", "O", "O"]],
              
              [["X", "X", "O"],
               ["O", "X", "O"],
               ["X", None, "O"]],
              ]


for i in test_cases:
    print(Game.PrimitiveValue(i))