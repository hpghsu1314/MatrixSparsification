from numpy import genfromtxt
import numpy as np
l, m, n = 4, 5, 6


nmk_size = f"{l}{m}{n}"
alg_type = "FlipGraphAlgorithms"

file = open(f"{alg_type}_complexity.csv", "a+")

path = f"C:/Users/hpghs/Desktop/Research/MatrixSparsification/algorithms/{alg_type}"

if alg_type == "DeepMindAlgorithms":
    original_u = genfromtxt(f"{path}/{nmk_size}/{nmk_size}_u_arr.csv", delimiter=",")
    original_v = genfromtxt(f"{path}/{nmk_size}/{nmk_size}_v_arr.csv", delimiter=",")
    original_w = genfromtxt(f"{path}/{nmk_size}/{nmk_size}_w_arr.csv", delimiter=",").transpose()
elif alg_type == "FlipGraphAlgorithms":
    alg = open(f"{path}/{nmk_size}/flips_mod0_{nmk_size}.alg")
    design_matrices = []
    temp = []
    for li in alg:
        line = li.rstrip().split(" ")
        if line[0] == "#":
            design_matrices.append(temp.copy())
            temp = []
        else:
            temp.append([float(x) for x in line])
    design_matrices.append(temp.copy())
    original_u = np.array(design_matrices[1]).T
    original_v = np.array(design_matrices[2]).T
    original_w = np.array(design_matrices[3])

new_u_arr = genfromtxt(f"{path}/{nmk_size}/Optimal/new_{nmk_size}_u_arr.csv", delimiter=",")
new_v_arr = genfromtxt(f"{path}/{nmk_size}/Optimal/new_{nmk_size}_v_arr.csv", delimiter=",")
new_w_arr = genfromtxt(f"{path}/{nmk_size}/Optimal/new_{nmk_size}_w_arr.csv", delimiter=",").transpose()
t = len(new_u_arr)


def countLinOp(matrix):
    count = -1 * len(matrix)
    for row in matrix:
        for entry in row:
            if entry == 1 or entry == -1:
                count += 1
            elif entry != 0:
                count += 2
    return count

uConst, vConst, wConst = countLinOp(new_u_arr), countLinOp(new_v_arr), countLinOp(new_w_arr)


print("Original")
print(countLinOp(original_u) + countLinOp(original_v) + countLinOp(original_w))
original_const = 1 + (countLinOp(original_u)/ (t - (l * m))) + (countLinOp(original_v)/ (t - (m * n))) + (countLinOp(original_w)/ (t - (l * n)))
print(1 + (countLinOp(original_u)/ (t - (l * m))) + (countLinOp(original_v)/ (t - (m * n))) + (countLinOp(original_w)/ (t - (l * n))))

#file.write(f"Original: {countLinOp(original_u) + countLinOp(original_v) + countLinOp(original_w)}\n")

def algComplexity():
    subConst = (uConst / (t - (l * m))) + (vConst / (t - (m * n))) + (wConst / (t - (l * n)))
    leadingConst = (1 + subConst)
    
    print("Outcome")
    print(uConst + vConst + wConst)

    return f"{leadingConst}"


def algComplexityThm6():
    subU = (l*m*n*n + n*l*t + t*t) * uConst
    subV = (l*l*m*n + m*l*t + t*t) * vConst
    subW = (l*m*m*n + m*n*t + t*t) * wConst
    subT = (l*m*n*n - l*m*m*n - m*n*t + n*l*t) * t
    leadingConst = 1 + ((subU + subV + subW + subT) / (t*t*t - l*l*m*m*n*n))
    return f"{leadingConst}"

def algComplexityCorollary7():
    subU = (l*m*n*n + m*n*t + t*t) * uConst
    subV = (l*l*m*n + n*l*t + t*t) * vConst
    subW = (l*m*m*n + m*l*t + t*t) * wConst
    subT = (l*l*m*n - l*m*m*n - m*l*t + n*l*t) * t
    leadingConst = 1 + ((subU + subV + subW + subT) / (t*t*t - l*l*m*m*n*n))
    return f"{leadingConst}"

comp_1 = algComplexity()
comp_2 = algComplexityThm6()
comp_3 = algComplexityCorollary7()
print(comp_1, comp_2, comp_3)

file.write(f"{l}_{m}_{n}: {countLinOp(original_u) + countLinOp(original_v) + countLinOp(original_w)}, {original_const}; {uConst + vConst + wConst}, {comp_1}, {comp_2}, {comp_3}\n")
file.close()