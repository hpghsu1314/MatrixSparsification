from numpy import genfromtxt
import numpy as np
n, m, k = 3, 3, 3

file = open(f"DeepMindAlgorithmsUVW.csv", "a+")

nmk_size = f"{n}_{m}_{k}"
alg_type = "DeepMindAlgorithms"

path = f"C:/Users/hpghs/Desktop/Research/MatrixSparsification/algorithms/{alg_type}"

if alg_type == "DeepMindAlgorithms":
    original_u = genfromtxt(f"{path}/{nmk_size}/{nmk_size}_u_arr.csv", delimiter=",")
    original_v = genfromtxt(f"{path}/{nmk_size}/{nmk_size}_v_arr.csv", delimiter=",")
    original_w = genfromtxt(f"{path}/{nmk_size}/{nmk_size}_w_arr.csv", delimiter=",").transpose()
elif alg_type == "FlipGraphAlgorithms":
    alg = open(f"{path}/{nmk_size}/flips_mod0_{nmk_size}.alg")
    design_matrices = []
    temp = []
    for l in alg:
        line = l.rstrip().split(" ")
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



u_phi_inv = genfromtxt(f"{path}/{nmk_size}/Optimal/{nmk_size}_u_arr_phi_inv.csv", delimiter=",")
v_phi_inv = genfromtxt(f"{path}/{nmk_size}/Optimal/{nmk_size}_v_arr_phi_inv.csv", delimiter=",")
w_phi_inv = genfromtxt(f"{path}/{nmk_size}/Optimal/{nmk_size}_w_arr_phi_inv.csv", delimiter=",").transpose()

def countLinOp(matrix):
    count = -1 * len(matrix)
    for row in matrix:
        for entry in row:
            if entry == 1 or entry == -1:
                count += 1
            elif entry != 0:
                count += 2
    return count

#file.write(f"{nmk_size}, {countLinOp(new_u_arr)}, {countLinOp(new_v_arr)}, {countLinOp(new_w_arr)}\n")


print("Original")
print(countLinOp(original_u) + countLinOp(original_v) + countLinOp(original_w))
print(1 + (countLinOp(original_u)/ (t - (n * m))) + (countLinOp(original_v)/ (t - (m * k))) + (countLinOp(original_w)/ (t - (n * k))))

#file.write(f"Original: {countLinOp(original_u) + countLinOp(original_v) + countLinOp(original_w)}\n")

def algComplexity():
    
    uConst, vConst, wConst = countLinOp(new_u_arr), countLinOp(new_v_arr), countLinOp(new_w_arr)
    subConst = (uConst / (t - (n * m))) + (vConst / (t - (m * k))) + (wConst / (t - (n * k)))
    leadingConst = (1 + subConst)
    
    print("Outcome")
    print(uConst + vConst + wConst)
    #file.write(f"Outcome: {uConst + vConst + wConst}\n")
    
    uPhiConst, vPhiConst, wPhiConst = countLinOp(u_phi_inv), countLinOp(v_phi_inv), countLinOp(w_phi_inv)
    uPhiTimeComp = uPhiConst / (n * m)
    vPhiTimeComp = vPhiConst / (m * k)
    wPhiTimeComp = wPhiConst / (k * k)
    
    return f"{leadingConst}(n^log_{n * m * k}({t}^3)) - {subConst}(n^2) + {uPhiTimeComp}*(nm)log_{n*m}(nm) + {vPhiTimeComp}*(mk)log_{m*k}(mk) + {wPhiTimeComp}*(kk)log_{k*k}(kk)"

comp = algComplexity()
print(comp)

#file.write(f"{n}_{m}_{k}: {comp}\n")
