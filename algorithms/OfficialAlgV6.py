import z3
import numpy as np

name = "456"

algorithmFolder = "FlipGraphAlgorithms"

#For DeepMind Algorithms
if algorithmFolder == "DeepMindAlgorithms":
    u_arr = np.loadtxt(f"C:/Users/hpghs/Desktop/Research/MatrixSparsification/algorithms/{algorithmFolder}/{name}/{name}_u_arr.csv", dtype="float", delimiter=",")
    v_arr = np.loadtxt(f"C:/Users/hpghs/Desktop/Research/MatrixSparsification/algorithms/{algorithmFolder}/{name}/{name}_v_arr.csv", dtype="float", delimiter=",")
    w_arr = np.loadtxt(f"C:/Users/hpghs/Desktop/Research/MatrixSparsification/algorithms/{algorithmFolder}/{name}/{name}_w_arr.csv", dtype="float", delimiter=",")

#For FlipGraph Algorithms
elif algorithmFolder == "FlipGraphAlgorithms":
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
    w_arr = np.array(design_matrices[3]).T
    
array_chosen = v_arr
array_name = "v"

print(array_chosen)
print(name)
print("=======================")


class function:
    def __init__(self, matrix, name, arith):
        self.matrix = np.array(matrix)
        self.name = name
        self.solver = z3.Solver()
        self.input_size = matrix.shape[1]
        self.solution = []
        self.res_vect = []
        self.z3_matrix = {}
        self.arith = arith

    # create solution vector with z3
    def sol_vect(self):
        self.solution = z3.RealVector("sol", self.input_size)

    # create result vector with z3
    def result_vector(self):
        self.res_vect = z3.RealVector("res", self.matrix.shape[0])

    # result when multiplying solution vector with matrix
    def match_result(self):
        for row in range(self.matrix.shape[0]):
            self.z3_matrix[f"res__{row}"] = z3.simplify(sum(self.matrix[row,entry] * self.solution[entry] for entry in range(array_chosen.shape[1])))
            self.solver.add(self.z3_matrix[f"res__{row}"] == self.res_vect[row])

    # sets the number of arithmetic with z3
    def sparsify(self):
        number = sum(z3.If(n != 0, 1, 0) for n in self.res_vect) + sum(z3.If(z3.Or(n == 0, n == 1, n == -1), 0, 1) for n in self.res_vect)
        self.solver.add(number == self.arith)

    # adds non-zero vector constraint
    def non_zero_sol(self):
        self.solver.add(z3.Or([entry == 1 for entry in self.res_vect]))

    # adds linear independency constraint
    def implicit_gau_elim(self, sol, rank):
        if rank != 0:
            constants = np.array([z3.Real(f"c_{i}") for i in range(rank)])
            combinations = constants @ (sol[:rank, :])
            statement = z3.Or([combinations[idx] != self.solution[idx] for idx in range(len(self.solution))])
            self.solver.add(z3.simplify(z3.ForAll(constants, statement)))
            

    def run(self):
        self.sol_vect()
        self.result_vector()
        # self.non_zero_sol()
        # self.match_result()
        self.sparsify()

    def base_run(self):
        self.sol_vect()
        self.result_vector()
        self.non_zero_sol()
        self.match_result()


def z3_to_float(v):
    if v.is_int():
        return float(v.as_long())
    return float(v.as_fraction().numerator) / float(v.as_fraction().denominator)


def find_vectors():
    rank = array_chosen.shape[1]
    solutions = np.zeros((rank, rank))
    base = function(array_chosen, array_name, 0)
    base.base_run()
    assertions = base.solver.assertions()
    i = 0
    for arith in range(1, 2 * array_chosen.shape[0]):
        print(arith)  # delete this line if you don't want to see the arithmetic count
        s = function(array_chosen, array_name, arith)
        s.run()
        s.solver.add(assertions)
        if i:
            s.implicit_gau_elim(solutions, i)
        while s.solver.check() == z3.sat:
            m = s.solver.model()
            nicer = [(d, m[d]) for d in m if "sol" in d.name()]
            nicer = sorted(nicer, key=lambda x: int(x[0].name()[5:]))
            solutions[i] = [z3_to_float(solution[1]) for solution in nicer]
            print(solutions[i])  # delete this line if you don't want to see the vectors
            if np.linalg.matrix_rank(solutions) == rank:
                return solutions
            i += 1
            s.implicit_gau_elim(solutions, i)
            print(s.solver.sexpr())
            print(i)


def count_arithmetic(matrix):
    pm = {1, -1}
    arithmetic = -matrix.shape[1] if array_name == "w" else -matrix.shape[0]
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if matrix[row, col]:
                arithmetic += 2 - (matrix[row, col] in pm)
    return arithmetic

final = find_vectors().T

n = np.matmul(array_chosen, final).astype(np.float16)
print(n)
n_arith = count_arithmetic(n)
original_arith = count_arithmetic(array_chosen)
print(f"original: {original_arith}")
print(f"resulting: {n_arith}")
phi_inv = np.linalg.inv(final).astype(np.float16)

"""
np.savetxt(f"C:/Users/hpghs/Desktop/Research/MatrixSparsification/algorithms/{algorithmFolder}/{name}/Optimal/{name}_{array_name}_arr_phi_inv.csv", phi_inv, fmt = "%d", delimiter = ",")
np.savetxt(f"C:/Users/hpghs/Desktop/Research/MatrixSparsification/algorithms/{algorithmFolder}/{name}/Optimal/new_{name}_{array_name}_arr.csv", n, fmt = "%d", delimiter = ",")
"""
print((array_chosen == np.matmul(n, phi_inv)).all())