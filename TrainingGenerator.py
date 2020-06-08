import numpy as np 
from util import hiff_fitness
import math
import copy 

def gen_train_sat(N, set_size):
    """
    Generate training set for H-IFF problem. 
    
    return: binary array of size N to train NN
    """
    input = []
    output = []

    if not (math.log2(N)).is_integer():
            raise ValueError("Array size must be power of 2.")
    for k in range(set_size):
        candidate_solution = np.random.randint(2, size = N)
        input.append(candidate_solution)
        solution_fitness = hiff_fitness(candidate_solution)
        for i in range(10 * N):
            index = np.random.randint(N)
            new_candidate_sol = copy.copy(candidate_solution)
            new_candidate_sol[index] = 1 - new_candidate_sol[index] # apply variation 
            new_fitness = hiff_fitness(new_candidate_sol) # check the change 
            if new_fitness >= solution_fitness : 
                candidate_solution = new_candidate_sol
                solution_fitness = new_fitness
        output.append(candidate_solution)

    return input, output
