import z3
import numpy as np

u_arr = np.loadtxt("C:/Users/hpghs/Desktop/Research/algorithms/new_3_4_5_u_arr.csv", dtype="int", delimiter=",")
v_arr = np.loadtxt("C:/Users/hpghs/Desktop/Research/algorithms/v_arr.csv", dtype="int", delimiter=",")
w_arr = np.loadtxt("C:/Users/hpghs/Desktop/Research/algorithms/w_arr.csv", dtype="int", delimiter=",")

array_chosen = u_arr
input_size = len(array_chosen[0])
print(input_size)
def count_arithmetic(matrix):
    arithmetic = 0
    for row in matrix:
        for entry in row:
            if entry != 0:
                arithmetic += 1
        arithmetic -= 1
    return arithmetic   
print(count_arithmetic(array_chosen))