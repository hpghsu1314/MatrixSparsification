from email.errors import CloseBoundaryNotFoundDefect
from numpy import genfromtxt

n, m, k = 3, 5, 9

nmk_size = f"{n}_{m}_{k}"

original_u = genfromtxt(f"C:/Users/hpghs/Desktop/Research/algorithms/{nmk_size}/{nmk_size}_u_arr.csv", delimiter=",").astype(dtype=int)
original_v = genfromtxt(f"C:/Users/hpghs/Desktop/Research/algorithms/{nmk_size}/{nmk_size}_v_arr.csv", delimiter=",").astype(dtype=int)
original_w = genfromtxt(f"C:/Users/hpghs/Desktop/Research/algorithms/{nmk_size}/{nmk_size}_w_arr.csv", delimiter=",").astype(dtype=int).transpose()

t = len(original_u)

def countLinOp(matrix):
    count = -1 * len(matrix)
    for row in matrix:
        for entry in row:
            if entry != 0:
                count += 1
    return count

def countNonSingular(matrix):
    count = 0
    for row in matrix:
        for entry in row:
            if entry != 1 and entry != 0 and entry != -1:
                count += 1
    return count

def algorithmComplexity():
    Uop = countLinOp(original_u) + countNonSingular(original_u)
    Vop = countLinOp(original_v) + countNonSingular(original_v)
    Wop = countLinOp(original_w) + countNonSingular(original_w)

    subU = Uop / (t - (n * m))
    subV = Vop / (t - (m * k))
    subW = Wop / (t - (n * k))
    
    return 1 + subU + subV + subW

print(algorithmComplexity())