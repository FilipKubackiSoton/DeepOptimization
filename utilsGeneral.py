import numpy as np
import math
import copy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Reshape
from shallowNet.shallowNet import shallowNet, DenseTranspose
from tensorflow.keras.optimizers import Adam
import shutil
from KnapSack import KnapSack
import os

datasets_directory_name = "saved_datasets"
models_directory_name = "saved_models"
plots_directory_name = "saved_plots"

knapSack = KnapSack("100_5_25_1")
fitness_fun = knapSack.Fitness

def fitness_function(argument):
    knapSack = KnapSack("100_5_25_1")
    fitness_fun = knapSack.Fitness
    return fitness_fun(argument)

def save(*args):
    """
    Pass list of either of models or datasets.
    They will be saved in the respective directories:
    models - models_directory_name
    plots - plots_directory_name
    """


    def save_model(model):
        """
        Save model in the directory saved_model. 
        The model will not be saved if it's alreday saved. 
        
        Parameters: 
            model - TF's model 
        
        """
        if model in save.saved_models:
            print("[INFO]: This model was already saved!!!")
        else:
            save.saved_models.append(model)  # append list of saved models
            save.model_counter += 1  # get number of saved model
            model_name = str(
                "model_" + str(save.model_counter)
            )  # consruct name of the model
            model_dir = os.path.join(models_directory_name, model_name)  # model dir
            model_path = Path(model_dir)  # model path
            # create model dir or if it's empty clean it
            try:
                model_path.rmdir()
            except OSError as e:
                print(f"Error: {model_path} : {e.strerror}")
            model_path.mkdir(exist_ok=True, parents=True)

            # Save the entire model
            model.save(model_dir)
            if os.path.exists(model_path):
                shutil.rmtree(model_path)
            os.makedirs(model_path)
            model.save(model_path)
            print("[INFO]: This model was saved in the directory: ", model_path)


    def save_dataset(dataset):
        """
        Save data set used in training DO Networks. 

        Parameters:
            dataset - dataset we want to save 
        """
        save.dataset_counter += 1
        save.saved_datasets.append(dataset)
        dataset_dir = datasets_directory_name + "/training_dataset_{}.npy".format(save.dataset_counter)
        dataset_path = Path(datasets_directory_name)
        try:
            dataset_path.rmdir()
        except OSError as e:
            print(f"Error: {dataset_path} : {e.strerror}")
        dataset_path.mkdir(exist_ok=True, parents=True)
        if os.path.exists(dataset_path) and save.dataset_counter == 1:
            shutil.rmtree(dataset_path)
            os.makedirs(dataset_path)
        with open(dataset_dir, 'wb') as f:
            np.save(f, dataset)
        print("[INFO]: Dataset was saved in the directory: ", dataset_path)

    model_type = tf.python.keras.engine.training.Model
    dataset_type = np.ndarray
    for instance_to_save in args:      
        if isinstance(instance_to_save, model_type ):
            save_model(instance_to_save)
        elif isinstance(instance_to_save, dataset_type):
            save_dataset(instance_to_save)
        else:
            print(str(instance_to_save), " cannot be saved: it's not an model or a dataset")
        

save.model_counter = 0
save.saved_models = []

save.dataset_counter = 0
save.saved_datasets = []

def load_models(*model_index):
    def get_model(model_index):
        model_dir = models_directory_name + "/model_{}".format(model_index)
        return tf.keras.models.load_model(model_dir)
        
    model_list = []
    for m_i in model_index:
        model_list.append(get_model(m_i))
    return model_list

def load_datasets(*dataset_index):
    def get_dataset(dataset_index):
        dataset_dir = datasets_directory_name + "/training_dataset_{}.npy".format(dataset_index)
        return np.load(dataset_dir)

    datasets_list = []
    for d_i in dataset_index:
        datasets_list.append(get_dataset(d_i))
    return datasets_list
    
def create_plot_path(name):
    return Path(os.path.join(plots_directory_name, name))

def generate_training_sat(N, set_size, debuge_variation=False):
    """
    Generate training set for H-IFF problem. 
    
    return: binary array of size N to train NN
    """
    output = np.ndarray(shape=(set_size, N))

    for k in range(set_size):
        candidate_solution = np.random.randint(2, size=N)
        solution_fitness = fitness_function(candidate_solution)
        for i in range(1 * N):
            index = np.random.randint(N)
            new_candidate_sol = copy.copy(candidate_solution)
            new_candidate_sol[index] = 1 - new_candidate_sol[index]  # apply variation
            new_fitness = fitness_function(new_candidate_sol)  # check the change
            if new_fitness >= solution_fitness:
                candidate_solution = new_candidate_sol
                solution_fitness = new_fitness
            if debuge_variation:
                print("For sample {} - iteration {} : fitness - {}".format(k,i, solution_fitness))
        output[k] = knapSack.SolToTrain(candidate_solution) # convert 0's to -1's
        #####output[k] = candidate_solution
    return output

def rand_bin_array(K, N):
    """
    THe function return random binary string. 
    The string has K - 0's adn N-K - 1's
    """
    arr = np.zeros(N)
    arr[:K] = 1
    np.random.shuffle(arr)
    return arr

def hiff_fitness(array):
    """
    Calculate and return value related to h-iff 
    assignment to the binary string of array. 
    """

    def f(val):
        if val == 1 or val == 0:
            return 1
        else:
            return 0

    def t(left, right):
        if left == 1 and right == 1:
            return 1
        elif left == 0 and right == 0:
            return 0
        else:
            return None

    def val_recursive(array, flor, sum):
        if flor > levels:
            return sum
        arr = []
        power = 2 ** flor
        for i in range(0, 2 ** (levels - flor) - 1, 2):
            arr.append(t(array[i], array[i + 1]))
            sum = sum + (f(array[i]) + f(array[i + 1])) * power
        return val_recursive(arr, flor + 1, sum)

    size = len(array)
    if not (size / 2).is_integer():
        raise ValueError("Array size must be power of 2.")
    levels = int(math.log2(size))
    sum = 0
    return val_recursive(array, 0, sum)