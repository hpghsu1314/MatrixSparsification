import gurobipy as gp
import numpy as np

name = "223"

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
        self.solver = gp.Model("MatrixSparsification")
        self.input_size = matrix.shape[1]
        self.solution = []
        self.res_vect = []
        self.arith_vect = []
        self.arith_vect_sing = []
        self.z3_matrix = {}
        self.arith = arith

    def displayModel(self):
        self.solver.display()

    # create solution vector with z3
    def sol_vect(self):
        self.solution = self.solver.addMVar(self.input_size, lb=-float("inf"))

    def result_vector(self):
        self.res_vect = self.solver.addMVar(self.matrix.shape[0], lb=-float("inf"))
        
    def match_result(self):
        self.solver.addConstr((self.matrix @ self.solution) == self.res_vect)
        self.solver.update()
        
    def sparsify(self):
        self.arith_vect = self.solver.addMVar(self.matrix.shape[0], lb=0, ub=1, vtype="B")
        self.arith_vect_sing = self.solver.addMVar(self.matrix.shape[0], lb=0, ub=1, vtype="B")
        for idx in range(self.matrix.shape[0]):
            """
                if self.arith_vect_sing == 1, then self.solution is not singular
                else we have that it is
                if self.arith_vect == 0, then self.solution is zero
                else we have that self.solution isnt zero
        
            """
            
            
        self.solver.setObjective(sum(self.arith_vect) + sum(self.arith_vect_sing))
        self.solver.update()

base = function(array_chosen, array_name, 0)
base.sol_vect()
base.result_vector()
base.match_result()
base.sparsify()
base.displayModel()