import z3
import numpy as np
from itertools import combinations

name = "3_9_11"

u_arr = np.loadtxt(f"C:/Users/hpghs/Desktop/Research/algorithms/{name}/{name}_u_arr.csv", dtype="float", delimiter=",")
v_arr = np.loadtxt(f"C:/Users/hpghs/Desktop/Research/algorithms/{name}/{name}_v_arr.csv", dtype="float", delimiter=",")
w_arr = np.loadtxt(f"C:/Users/hpghs/Desktop/Research/algorithms/{name}/{name}_w_arr.csv", dtype="float", delimiter=",")

array_chosen = v_arr
array_name = "v"

print(array_chosen)

print("=======================")

class function:
    
    def __init__(self, matrix, name, zeros):
        self.matrix = np.array(matrix)
        self.name = name
        self.solver = z3.Solver()
        self.input_size = len(matrix[0])
        self.solution = []
        self.res_vect = []
        self.z3_matrix = {}
        self.zeros = zeros
        
    #create solution vector with z3
    def sol_vect(self):
        self.solution = z3.IntVector("sol", self.input_size)
        for entry in self.solution:
            self.solver.add(z3.Or(entry == 0, entry == 1, entry == -1))
    
    def result_vector(self):
        self.res_vect = z3.IntVector("res", len(self.matrix))
        for entry in self.res_vect:
            self.solver.add(z3.Or(entry == 0, entry == 1, entry == -1))
            
    #result when multiplying solution vector with matrix
    def match_result(self):
        for row in range(len(self.matrix)):
            self.z3_matrix[f"res__{row}"] = z3.simplify(sum(self.matrix[row,entry] * self.solution[entry] for entry in range(self.input_size)))
            self.solver.add(self.z3_matrix[f"res__{row}"] == self.res_vect[row])
    
    def sparsify(self):
        number = sum(self.z3_is_zero(integer) for integer in self.res_vect)
        self.solver.add(number == self.zeros)
    
    def z3_is_zero(self, integer):
        return z3.If(integer != 0, 0, 1)
    
    #adds non-zero vector constraint
    def non_zero_sol(self):
        temp = True
        for entry in self.solution:
            temp = z3.simplify(z3.And(temp, entry == 0), som=True)
        self.solver.add(z3.Not(temp))

    def del_vect(self, vector):
        temp = False
        neg_temp = False
        for entry in range(self.input_size):
            temp = z3.simplify(z3.Or(temp, vector[entry] != self.solution[entry]), som=True)
            neg_temp = z3.simplify(z3.Or(neg_temp, vector[entry] != -1 * self.solution[entry]), som=True)
        self.solver.add(temp, neg_temp)

    def run(self):
        self.sol_vect()
        self.result_vector()
        self.non_zero_sol()
        self.match_result()
        self.sparsify()


def find_vectors(zeros):       
    s = function(array_chosen, array_name, zeros)
    s.run()
    s.solver.check()
    sol_num = 1
    solutions = []
    while s.solver.check() == z3.sat:
        m = s.solver.model()
        nicer = sorted([(d, m[d]) for d in m], key = lambda x: int(x[0].name()[5:]))
        delete_vector = []
        for solution in nicer:
            if "sol" in solution[0].name():
                delete_vector.append(int(solution[1].as_string()))
        s.del_vect(delete_vector)
        if delete_vector not in solutions:
            solutions.append(delete_vector)
        sol_num += 1
    return solutions


found_vectors = {}
num_zeros = len(array_chosen)

def store_vectors(zeros):
    solutions = find_vectors(zeros)
    if solutions == []:
        if zeros == -1:
            print("No More Solutions")
            return
        zeros -= 1
        store_vectors(zeros)
    else:
        found_vectors[zeros] = solutions
    return

store_vectors(num_zeros)

def find_additional_vectors():
    zeros = min(found_vectors.keys())
    store_vectors(zeros - 1)

def gau_elim(vectors):
    matrix = []
    for v in vectors:
        matrix.append(v)
    matrix = np.array(matrix)
    for row in range(len(matrix)):
        pivot = 0
        while pivot != len(matrix[0]) and matrix[row][pivot] == 0:
            pivot += 1
        if pivot != len(matrix[0]):
            for vect in range(len(matrix[row+1:])):
                scale = matrix[row + vect + 1][pivot] / matrix[row][pivot]
                matrix[row + vect + 1] = matrix[row + vect + 1] - scale * matrix[row]
    return matrix


def check_lin_ind(matrix):
    lin_ind_count = 0
    for row in range(len(matrix)):
        if np.any(matrix[row]):
            lin_ind_count += 1
    return lin_ind_count

def add_ind_vect(ref, matrix):
    sol = []
    for row in range(len(ref)):
        if any(ref[row]):
            sol.append(matrix[row])
    return sol

pos_sol = []
pos_sol.extend(found_vectors[min(found_vectors.keys())])
ref = gau_elim(pos_sol)

while check_lin_ind(ref) < len(array_chosen[0]):
    find_additional_vectors()
    key = min(found_vectors.keys())
    pos_sol.extend(found_vectors[key])
    ref = gau_elim(pos_sol)
    print(check_lin_ind(ref))


def count_arithmetic(matrix):
    if array_name == "w":
        matrix = np.array(matrix).T
    arithmetic = 0
    for row in matrix:
        for entry in row:
            if entry != 0:
                arithmetic += 1
        arithmetic -= 1
    return arithmetic    


lin_ind_matrix = add_ind_vect(ref, pos_sol)

final = np.array(lin_ind_matrix).T
np_array_chosen = np.array(array_chosen)
n = np.matmul(np_array_chosen, final)
n_arith = count_arithmetic(n)
original_arith = count_arithmetic(np_array_chosen)
print(f"original: {original_arith}")
print(f"resulting: {n_arith}")
phi_inv = np.linalg.inv(final).astype(int)
np.savetxt(f"{name}_{array_name}_arr_phi_inv.csv", phi_inv, fmt = "%d", delimiter = ",")
np.savetxt(f"new_{name}_{array_name}_arr.csv", n, fmt = "%d", delimiter = ",")

print((array_chosen == np.matmul(n, phi_inv)).all())
