import numpy as np
from itertools import combinations

sol = np.loadtxt("C:/Users/hpghs/Desktop/Research/algorithms/sol.csv", dtype="int", delimiter=",")

linearly_independent = {}
def check_lin_ind():
    print(sol)
    print(sol[[0],:])
    input_size = len(sol[[0],:])
    comb = [i for i in range(len(sol))]
    comb = combinations(comb, input_size)
    solutions = []
    for i in comb:
        try:
            state = linearly_independent[i]
        except KeyError:
            linearly_independent[i] = False
        if linearly_independent[i] == False or np.linalg.det(sol[i,:]) == 0:
            linearly_independent[i] = False
        else:
            solutions.append(sol[i,:])
            return solutions
    return solutions