import numpy as np

name = "266"

algorithmFolder = "FlipGraphAlgorithms"
alg = open(f"C:/Users/hpghs/Desktop/Research/MatrixSparsification/algorithms/{algorithmFolder}/{name}/flips_mod0_{name}.alg")
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
u_arr = np.array(design_matrices[1]).T
v_arr = np.array(design_matrices[2]).T
print(v_arr.shape)
w_arr = np.array(design_matrices[3]).T

new_v_arr = np.loadtxt(f"C:/Users/hpghs/Desktop/Research/MatrixSparsification/algorithms/{algorithmFolder}/{name}/Optimal/new_{name}_v_arr.csv", dtype="float", delimiter=",")
phi_v_arr = np.loadtxt(f"C:/Users/hpghs/Desktop/Research/MatrixSparsification/algorithms/{algorithmFolder}/{name}/Optimal/{name}_v_arr_phi_inv.csv", dtype="float", delimiter=",")

print(phi_v_arr.shape)
print(new_v_arr.shape)
print(np.linalg.matrix_rank(phi_v_arr))
print(np.matmul(new_v_arr, phi_v_arr))
print(v_arr)