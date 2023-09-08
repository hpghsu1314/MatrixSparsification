import z3
import numpy as np
from itertools import combinations

name = "3_4_5"
u_arr = np.loadtxt(f"C:/Users/hpghs/Desktop/Research/algorithms/{name}/{name}_u_arr.csv", dtype="int", delimiter=",")
v_arr = np.loadtxt(f"C:/Users/hpghs/Desktop/Research/algorithms/{name}/{name}_v_arr.csv", dtype="int", delimiter=",")
#w_arr = np.loadtxt(f"C:/Users/hpghs/Desktop/Research/algorithms/{name}/{name}_w_arr.csv", dtype="int", delimiter=",")

array_chosen = v_arr
array_name = "v"

print(u_arr)
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
        self.solver.add(number >= self.zeros)
    
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
            print(delete_vector)
            solutions.append(delete_vector)
        if s.solver.check() == z3.sat:
            m = s.solver.model()
            nicer = sorted ([(d, m[d]) for d in m], key = lambda x: str(x[0]))
        sol_num += 1
    print(sol_num)
    return solutions

def count_arithmetic(matrix):
    arithmetic = 0
    for row in matrix:
        for entry in row:
            if entry != 0:
                arithmetic += 1
    return arithmetic            

chosen_arr_arith = count_arithmetic(array_chosen)

non_zero_det = {}

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