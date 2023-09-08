import z3
import numpy as np

num = 0

#upload csv folders for the matrix
u_arr = np.loadtxt("C:/Users/hpghs/Desktop/Research/algorithms/u_arr.csv", dtype="int", delimiter=",")
v_arr = np.loadtxt("C:/Users/hpghs/Desktop/Research/algorithms/v_arr.csv", dtype="int", delimiter=",")
w_arr = np.loadtxt("C:/Users/hpghs/Desktop/Research/algorithms/w_arr.csv", dtype="int", delimiter=",")

"""
#create matrix inputs
u_inputs = [f"a_{i}" for i in range(len(u_arr[0]))]
v_inputs = [f"b_{i}" for i in range(len(v_arr[0]))]

u_input_size = len(u_inputs)
v_input_size = len(v_inputs)

#fucntion to change strings to z3 reals
def inputs_to_reals(inputs):
    for i in range(len(inputs)):
        inputs[i] = z3.Real(inputs[i])
    return inputs

#change inputs to reals
u_inputs, v_inputs = inputs_to_reals(u_inputs), inputs_to_reals(v_inputs)

#define vector multiplied to matrix
def mult_matrix_vect(input, design_matrix):
    output_list = []
    for i in range(len(design_matrix)):
        equation = 0
        for entry in range(len(design_matrix[i])):
            equation = equation + design_matrix[i][entry] * input[entry]
        equation = z3.simplify(equation, som=True)
        output_list.append(equation)
    output_list = np.array(output_list)
    return output_list

u_out = mult_matrix_vect(u_inputs, u_arr)
v_out = mult_matrix_vect(v_inputs, v_arr)

intermediate = u_out * v_out

#creates correct outputs
result = mult_matrix_vect(intermediate, w_arr)
"""

#create z3 solver
solver = z3.Solver()

#based around z3 for finding solutions
class function:
    
    def __init__(self, matrix, name):
        self.matrix = matrix
        self.input_size = len(matrix[0])
        self.name = name
        self.mtn = self.input_size * len(self.matrix)
        self.constants = {}
        self.z3_matrix = {}
        self.memo = {}
        self.np_sol_matrix = []
        self.sol_determinant = 0
        self.solver = z3.Solver()
    
    #convert vectors to usable z3 form, stores in self.z3_matrix dictionary    
    def convert_to_z3_vector(self, vector, name):
        self.z3_matrix[f"{name}"] = z3.IntVector(name, self.input_size)
        for i in range(len(vector)):
            self.solver.add(self.z3_matrix[f"{name}"][i] == vector[i])
    
    def convert_matrix_to_vectors(self):
        for i in range(len(self.matrix)):
            self.convert_to_z3_vector(self.matrix[i], f"{self.name}__{i}")
    
    #create arbitrary phi matrix (solution matrix, it is transposed)
    def create_phi_matrix(self):
        for sol in range(self.input_size):
            self.z3_matrix[f"sol__{sol}"] = z3.IntVector(f"sol__{sol}", self.input_size)
            for entry in range(self.input_size):
                self.solver.add(z3.Or(self.z3_matrix[f"sol__{sol}"][entry] == 0, self.z3_matrix[f"sol__{sol}"][entry] == 1, self.z3_matrix[f"sol__{sol}"][entry] == -1))
        
    #creates variables for result
    def input_res(self):
        for result in range(len(self.matrix)):
            self.z3_matrix[f"res__{result}"] = z3.IntVector(f"res__{result}", self.input_size)
    
    #adds constraints for result
    def find_res(self):
        for sol in range(self.input_size):
            for line in range(len(self.matrix)):
                temp = 0
                for entry in range(self.input_size):
                    temp += self.z3_matrix[f"{self.name}__{line}"][entry] * self.z3_matrix[f"sol__{sol}"][entry]
                temp = z3.simplify(temp, som=True)
                self.solver.add(self.z3_matrix[f"res__{line}"][sol] == temp)
    
    #checks linear independence of solution/phi matrix
    def linear_independence(self, name):
        det = self.det(self.compile_square_np_matrix(name))
        self.sol_determinant = det
        self.solver.add(z3.simplify(det, som=True) != 0)
    
    #calculate determinant with recursion
    def det(self, matrix):
        if len(matrix) == 2:
            return matrix[0,0] * matrix[1,1] - matrix[1,0] * matrix[0,1]
        else:
            temp = 0
            size = len(matrix)
            for entry in range(size):
                new_matrix = [[] for __ in range(size - 1)]
                for y in range(size):
                    for x in range(size):
                        if y != 0 and x != entry:
                            new_matrix[y-1].append(matrix[y, x])
                temp += ((-1) ** entry) * matrix[0,entry] * self.det(np.array(new_matrix))
            return temp
    
    #make a matrix numpy from dictionary
    def compile_square_np_matrix(self, name):
        n = len(self.z3_matrix[f"{name}__0"])
        matrix = []
        for vector in range(n):
            matrix.append(self.z3_matrix[f"{name}__{vector}"])
        np_matrix = np.matrix(matrix)
        if name == "sol":
            self.np_sol_matrix = np_matrix
        return np_matrix
    
    #constraints on sparsification of the resulting matrix with solution/phi matrix with z3(main goal)
    def sparsification(self):
        self.constants["zeros"] = z3.Int("zeros")
        temp = 0
        for y in range(len(self.matrix)):
            for x in self.z3_matrix[f"res__{y}"]:
                temp += self.z3_is_zero(x)
        self.solver.add(temp > self.mtn//2 + 3)
    
    #none of the rows are fully zero
    def non_zero(self):
        for row in range(self.input_size):
            temp = True
            for value in self.z3_matrix[f"sol__{row}"]:
                temp = z3.And(temp, value == 0)
            self.solver.add(z3.Not(temp))
    
    #support function
    def z3_is_zero(self, integer):
        return z3.If(integer != 0, 0, 1)
    
    #find the inverse of solution matrix and make sure that all entries are 1, 0, or -1
    #unnecessary for W array
    def find_inverse(self):
        cofactor_matrix = [[] for __ in range(self.input_size)]
        for vector in range(self.input_size):
            for entry in range(self.input_size):
                temp_matrix = self.np_sol_matrix
                temp_matrix = np.delete(temp_matrix, vector, 0)
                temp_matrix = np.delete(temp_matrix, entry, 1)
                cofactor_matrix[vector].append((((-1)**vector) * ((-1)**entry) * self.det(temp_matrix)) / self.sol_determinant)
        for y in cofactor_matrix:
            for x in y:
                self.solver.add(z3.Or(x == 1, x == 0, x == -1))
    """            
    #linear independence without determinant
    def no_det_lin_ind(self):
        temp = False
        for const in range(self.input_size):
            self.constants[f"a__{const}"] = z3.Real(f"a__{const}")
            temp = z3.simplify(z3.Or(temp, self.constants[f"a__{const}"] != 0), som=True)
        equation = 0
        for row in range(self.input_size):
            for c in range(self.input_size):
    """            
        
        
print(u_arr)
s = function(u_arr, "u")
s.convert_matrix_to_vectors()
s.create_phi_matrix()
s.input_res()
s.find_res()
s.non_zero()
s.linear_independence("sol")
s.sparsification()
s.find_inverse()
print(s.solver.check())
m = s.solver.model()
nicer = sorted ([(d, m[d]) for d in m], key = lambda x: str(x[0]))
print(nicer)