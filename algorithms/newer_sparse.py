import z3
import numpy as np
from itertools import combinations

name = "3_4_5"
u_arr = np.loadtxt(f"C:/Users/hpghs/Desktop/Research/algorithms/{name}/{name}_u_arr.csv", dtype="int", delimiter=",")
v_arr = np.loadtxt(f"C:/Users/hpghs/Desktop/Research/algorithms/{name}/{name}_v_arr.csv", dtype="int", delimiter=",")
#w_arr = np.loadtxt(f"C:/Users/hpghs/Desktop/Research/algorithms/{name}/{name}_w_arr.csv", dtype="int", delimiter=",")

array_chosen = v_arr
array_name = "v"

print(v_arr)
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
        self.solution = z3.RealVector("sol", self.input_size)
    
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

create_solution = False

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
                delete_vector.append(int(solution[1].as_decimal(4)))
        s.del_vect(delete_vector)
        if delete_vector not in solutions:
            solutions.append(delete_vector)
        if s.solver.check() == z3.sat:
            m = s.solver.model()
            nicer = sorted ([(d, m[d]) for d in m], key = lambda x: str(x[0]))
        sol_num += 1
    return solutions

def count_arithmetic(matrix):
    arithmetic = 0
    for row in matrix:
        for entry in row:
            if entry != 0:
                arithmetic += 1
    return arithmetic            

chosen_arr_arith = count_arithmetic(array_chosen)

found_vectors = {}
   
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

num_zeros = len(array_chosen)
store_vectors(num_zeros)

def find_additional_vectors():
    zeros = min(found_vectors.keys())
    store_vectors(zeros - 1)

for i in range(3):
    find_additional_vectors()

solution_num = 0
for i in found_vectors:
    solution_num += len(found_vectors[i])
    print(len(found_vectors[i]))

print(solution_num > len(array_chosen[0]))

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
        
ref_matrix = gau_elim(found_vectors[41]) #change this line later

lin_ind_count = 0
linear_ind_matrix = []
for row in range(len(ref_matrix)):
    if np.any(ref_matrix[row]):
        lin_ind_count += 1
        linear_ind_matrix.append(found_vectors[41][row]) #change this line later

np_lin_ind_matrix = np.array(linear_ind_matrix)

for v in found_vectors[44]:
    linear_ind_matrix.append(v)
for v in found_vectors[43]:
    linear_ind_matrix.append(v)

next_matrix = gau_elim(linear_ind_matrix)
next_lin_ind = 0
for row in range(len(next_matrix)):
    if np.any(next_matrix[row]):
        next_lin_ind += 1

print(next_lin_ind)
"""

key=lambda v: min(n for n in range(len(v)) if v[n] != 0 )


def check_lin_ind(sol):
    input_size = len(sol[0,:])
    comb = [i for i in range(len(sol))]
    comb = combinations(comb, input_size)
    solutions = []
    for i in comb:
        determinant = np.linalg.det(sol[i,:])
        if determinant >= 0.001:
            solutions.append(sol[i,:])
            print(f"found {len(solutions)} solutions")
    return solutions

vectors = found_vectors[41] + found_vectors[42]
solution = check_lin_ind(np.array(vectors))
print(solution)

def check_lin_ind(sol):
    input_size = len(sol[0,:])
    comb = [i for i in range(len(sol))]
    comb = combinations(comb, input_size)
    solutions = []
    for i in comb:
        determinant = np.linalg.det(sol[i,:])
        if determinant >= 0.001:
            n = np.matmul(np.array(u_arr), sol[i,:].T)
            if count_arithmetic(n) < chosen_arr_arith:
                solutions.append(sol[i,:])
                print(f"found {len(solutions)} solutions")
                if len(solutions) > 100:
                    return solutions
    return solutions

num_zero = len(array_chosen)
input_size = len(array_chosen[0])

no_of_sol = len(find_vectors(num_zero))

while no_of_sol < input_size:
    print(f"number of solutions: {no_of_sol}")
    num_zero -= 1
    no_of_sol = len(find_vectors(num_zero))
    if num_zero == 0:
        print("no more solutions")
        break

solutions = find_vectors(num_zero)

while check_lin_ind(np.array(solutions)) == [] and num_zero >= 0:
    num_zero -= 1
    solutions = find_vectors(num_zero)

resultant = check_lin_ind(np.array(solutions))
least = np.inf

for i in resultant:
    n = np.matmul(np.array(u_arr), i.T)
    arith = count_arithmetic(n)
    if arith < least:
        least = arith

for i in resultant:
    n = np.matmul(np.array(u_arr), i.T)
    arith = count_arithmetic(n)
    if arith == least:
        print("Matrix Found")
        print(i)
        np.savetxt(f"{name}_{array_name}_arr_phi_inv.csv", np.linalg.inv(i.T), fmt = "%d", delimiter = ",")
        print("Resulting Matrix")
        print(n)
        np.savetxt(f"new_{name}_{array_name}_arr.csv", n, fmt = "%d", delimiter = ",")
        print("_________________________")
"""