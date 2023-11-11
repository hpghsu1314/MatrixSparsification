import z3
import numpy as np

name = "3_4_5"

u_arr = np.loadtxt(f"C:/Users/hpghs/Desktop/Research/algorithms/{name}/{name}_u_arr.csv", dtype="int", delimiter=",")
u_arr = np.array(u_arr)

"""
v_arr = np.loadtxt(f"C:/Users/hpghs/Desktop/Research/algorithms/{name}/{name}_v_arr.csv", dtype="int", delimiter=",")
u_arr = np.array(v_arr)
w_arr = np.loadtxt(f"C:/Users/hpghs/Desktop/Research/algorithms/{name}/{name}_w_arr.csv", dtype="int", delimiter=",")
u_arr = np.array(w_arr)
"""

phi_u_arr = np.loadtxt(f"C:/Users/hpghs/Desktop/Research/algorithms/{name}/{name}_u_arr_phi_inv.csv", dtype="int", delimiter=",")
phi_u_arr = np.array(phi_u_arr)

new_u_arr = np.loadtxt(f"C:/Users/hpghs/Desktop/Research/algorithms/{name}/new_{name}_u_arr.csv", dtype="int", delimiter=",")
new_u_arr = np.array(new_u_arr)

def verification(original, new, phi):
    return np.array_equal(original, np.matmul(new, phi))
    
print(verification(u_arr, new_u_arr, phi_u_arr))