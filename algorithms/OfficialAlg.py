import z3
import numpy as np
from itertools import combinations

name = "2_2_2"

u_arr = np.loadtxt(f"C:/Users/hpghs/Desktop/Research/MatrixSparsification/algorithms/{name}/{name}_u_arr.csv", dtype="float", delimiter=",")
v_arr = np.loadtxt(f"C:/Users/hpghs/Desktop/Research/MatrixSparsification/algorithms/{name}/{name}_v_arr.csv", dtype="float", delimiter=",")
w_arr = np.loadtxt(f"C:/Users/hpghs/Desktop/Research/MatrixSparsification/algorithms/{name}/{name}_w_arr.csv", dtype="float", delimiter=",")

array_chosen = u_arr
array_name = "u"

print(array_chosen)

print("=======================")

class function:
    
    def __init__(self, matrix, name, zeros):
        self.matrix = np.array(matrix)
        self.name = name
        self.solver = z3.Optimize()
        self.input_size = len(matrix[0])
        self.solution = []
        self.res_vect = []
        self.z3_matrix = {}
        self.zeros = zeros
        
    #create solution vector with z3
    def sol_vect(self):
        self.solution = z3.RealVector("sol", self.input_size)
        for entry in self.solution:
            self.solver.add_soft(z3.Or(entry == 0, entry == 1, entry == -1))
    
    def result_vector(self):
        self.res_vect = z3.RealVector("res", len(self.matrix))
        for entry in self.res_vect:
            self.solver.add_soft(z3.Or(entry == 0, entry == 1, entry == -1))
            
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
        
        anchor = 0
        temp = True
        while vector[anchor] == 0:
            anchor += 1
        for idx in range(0, len(vector), 1):
            temp = z3.simplify(z3.And(temp, (self.solution[idx] / self.solution[anchor] == vector[idx] / vector[anchor])))
        self.solver.add(z3.Not(temp))
        
        """
        temp = False
        neg_temp = False
        for entry in range(self.input_size):
            temp = z3.simplify(z3.Or(temp, vector[entry] != self.solution[entry]), som=True)
            neg_temp = z3.simplify(z3.Or(neg_temp, vector[entry] != -1 * self.solution[entry]), som=True)
        self.solver.add(temp, neg_temp)
        """
    
    def stringToFloat(self, fraction):
        numbers = fraction.split("/")
        if len(numbers) == 1:
            return int(numbers[0])
        return int(numbers[0]) / int(numbers[1])
    
    def run(self):
        self.sol_vect()
        self.result_vector()
        self.non_zero_sol()
        self.match_result()
        self.sparsify()


def find_vectors(zeros):       
    print(zeros)
    s = function(array_chosen, array_name, zeros)
    s.run()
    s.solver.check()
    sol_num = 1
    solutions = []
    while s.solver.check() == z3.sat:
        m = s.solver.model()
        results = [(d, m[d]) for d in m]
        nicer = []
        for v in results:
            if "sol" in v[0].name():
                nicer.append(v)
        nicer = sorted(nicer, key = lambda x: int(x[0].name()[5:]))
        delete_vector = []
        for solution in nicer:
            if "sol" in solution[0].name():
                delete_vector.append(s.stringToFloat(solution[1].as_string()))
        s.del_vect(delete_vector)
        print(delete_vector)
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
            exit()
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

def matrixMultVector(matrix, vector):
    solution = []
    for v in matrix:
        solution.append(sum([vector[idx] * v[idx] for idx in range(len(vector))]))
    return solution

def count_arithmetic(matrix):
    if array_name == "w":
        matrix = np.array(matrix).T
    arithmetic = 0
    for row in matrix:
        arithmetic += count_arithmetic_helper(row)
    return arithmetic    

def count_arithmetic_helper(vector):
    count = -1
    for i in vector:
        if i == 1 or i == -1:
            count += 1
        elif i != 0:
            count += 2
    return count

pos_sol = []
pos_sol.extend(found_vectors[min(found_vectors.keys())])
ref = gau_elim(pos_sol)

while check_lin_ind(ref) < len(array_chosen[0]):
    find_additional_vectors()
    key = min(found_vectors.keys())
    pos_sol.extend(found_vectors[key])
    ref = gau_elim(pos_sol)
    print(ref)

#pos_sol = sorted(pos_sol, key=lambda v: count_arithmetic_helper(matrixMultVector(array_chosen, v)))
ref = gau_elim(pos_sol)
lin_ind_matrix = add_ind_vect(ref, pos_sol)

final = np.array(lin_ind_matrix).T
np_array_chosen = np.array(array_chosen)
n = np.matmul(np_array_chosen, final)
n_arith = count_arithmetic(n)
original_arith = count_arithmetic(np_array_chosen)
print(f"original: {original_arith}")
print(f"resulting: {n_arith}")
phi_inv = np.linalg.inv(final).astype(float)
#np.savetxt(f"{name}_{array_name}_arr_phi_inv.csv", phi_inv, fmt = "%d", delimiter = ",")
#np.savetxt(f"new_{name}_{array_name}_arr.csv", n, fmt = "%d", delimiter = ",")

print((array_chosen == np.matmul(n, phi_inv)).all())
