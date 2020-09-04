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
import pickle
import re


class UtilsGeneral:
    def __init__(self, knapSack):
        self.knapSack = knapSack
        self.fitness_function = self.knapSack.Fitness  
        self.datasets_directory_name = "saved_datasets"
        self.models_directory_name = "saved_models"
        self.plots_directory_name = "saved_plots"
        self.model_counter = 0
        self.saved_models = []
        self.dataset_counter = 0
        self.saved_datasets = []


    def save_obj(self, obj, name ):
        with open( name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_obj(self, name ):
        with open( name + '.pkl', 'rb') as f:
            return pickle.load(f)

    def get_numpy_values_from_dictionary(self, directory, encoder_template, weights_template, decoder_template, decoder_weights_template = None, decoder_reverse_order = True, show = True):
        """
        Return dictionaries of - layer_number -> numpy value
        It returns dictionaries of: 
            encoder bias, 
            encoder weights, 
            decoder bias, 
            decoder weights
        
        !!! Be careful with regular expressions: good template is for instance: 
            get_numpy_values_from_dictionary()
                directory = "100_10_25_7\Model_CheckPoints",
                encoder_template = 'Train_BiasEncoder_L(.+)_EvoStep_3', 
                weights_template = 'Train_Weights_L(.+)_TrainLevel_3',
                decoder_template = 'Train_BiasDecoder_L(.+)_EvoStep_3',
                show = True
            )

        that includes the following files: 
            Encoder bias[1]:  (1, 200): Train_BiasEncoder_L1_EvoStep_3
            Encoder bias[2]:  (1, 150): Train_BiasEncoder_L2_EvoStep_3
            Encoder bias[3]:  (1, 112): Train_BiasEncoder_L3_EvoStep_3
            Encoder weights[1]:  (100, 200): Train_Weights_L1_EvoStep_3
            Encoder weights[2]:  (200, 150): Train_Weights_L2_EvoStep_3
            Encoder weights[3]:  (150, 112): Train_Weights_L3_EvoStep_3
            Decoder atributes have been reversed
            Decoder bias[1]:  (1, 150): 'Train_BiasDecoder_L3_EvoStep_3'
            Decoder bias[2]:  (1, 200): 'Train_BiasDecoder_L2_EvoStep_3'
            Decoder bias[3]:  (1, 100): 'Train_BiasDecoder_L1_EvoStep_3'
            Decoder weights are transposed of reversed encoder weights
            Decoder weights[1]:  (112, 150): transpose of - Train_Weights_L3_EvoStep_3
            Decoder weights[2]:  (150, 200): transpose of - Train_Weights_L2_EvoStep_3
            Decoder weights[3]:  (200, 100): transpose of - Train_Weights_L1_EvoStep_3

        Example:
            
        """

        def reverse_dictionary(dic, function=lambda x : x ):
            reverse_dic = {}
            i = max(dic.keys())
            index = 1
            while i>0: 
                reverse_dic[index] = function(dic[i])
                i-=1
                index +=1
            return reverse_dic
        

        pathlist = Path(directory).rglob("*.npy") # list of numpy files
        encoder_bias = {} # initialize dictionary 
        decoder_bias = {} # initialize dictionary 
        encoder_weights = {} # initialize dictionary 
        decoder_weights = {} # initialize dictionary 

        for file in pathlist: # iterate over file in the directory
            if(re.match(encoder_template, file.stem)):
                index = int(re.findall(encoder_template, file.stem)[0])
                encoder_bias[index] = np.load(file)
            
            if(re.match(decoder_template, file.stem)):
                index = int(re.findall(decoder_template, file.stem)[0])
                decoder_bias[index] = np.load(file)

            if(re.match(weights_template, file.stem)):
                index = int(re.findall(weights_template, file.stem)[0])
                encoder_weights[index] = np.load(file)
            
            if(decoder_weights_template != None and re.match(decoder_weights_template, file.stem)):
                index = int(re.findall(decoder_weights_template, file.stem)[0])
                decoder_weights[index] = np.load(file)
            
        if(decoder_reverse_order):
            decoder_bias = reverse_dictionary(decoder_bias)
            if(decoder_weights_template==None):
                decoder_weights = reverse_dictionary(encoder_weights, np.transpose)
            else:
                decoder_weights = reverse_dictionary(encoder_weights)

            
        if(show):
            for k,v in encoder_bias.items():
                print("Encoder bias[{}]: ".format(k), np.shape(encoder_bias[k]))
            
            for k,v in encoder_weights.items():
                print("Encoder weights[{}]: ".format(k), np.shape(encoder_weights[k]))
            
            if(decoder_reverse_order):
                print("Decoder atributes have been reversed")

            for k,v in decoder_bias.items():
                print("Decoder bias[{}]: ".format(k), np.shape(decoder_bias[k]))

            if(decoder_weights_template == None):
                print("Decoder weights are transposed of encoder weights")

            for k,v in decoder_weights.items():
                print("Decoder weights[{}]: ".format(k), np.shape(decoder_weights[k]))

        return  encoder_bias, encoder_weights, decoder_bias, decoder_weights

    def restore_model_from_numpy_dictionaries(self, encoder_bias, encoder_weights, decoder_bias, decoder_weights, show=False):
        
        class NumpyInitializer(tf.keras.initializers.Initializer):
            # custom class converting numpy arrays to tf's initializers 
            # used to initialize both kernel and bias
            def __init__(self, array):
                # convert numpy array into tensor 
                self.array = tf.convert_to_tensor(array.tolist())
                
            def __call__(self, shape, dtype=None):
                # return tensor 
                return self.array 
                
        # CREATE ENCODING PART
        sample_size = np.shape(encoder_weights[1])[0]
        inputs = Input(shape = (sample_size,)) # create first model input layer
        x = inputs 

        for layer_number  in range(1,max(encoder_weights.keys())+1):
            bias_initializer = NumpyInitializer(encoder_bias[layer_number][0]) # create bias initializer for key's layer 
            kernal_initializer = NumpyInitializer(encoder_weights[layer_number]) # create weights initializer for key's layer 
            layer_size = len(encoder_bias[layer_number][0]) # get the size of the laye

            new_layer = tf.keras.layers.Dense( # initialize new Dense layer
                units = layer_size, 
                kernel_initializer=kernal_initializer, 
                bias_initializer = bias_initializer,
                activation="tanh")
            new_layer.trainable = False # optional: I was dealing with pretrained model so I disabled trainable
            x = new_layer(x) # stack layer at the top of the previous layer

        #STACK ABOVE DECODING PART 

        for layer_number  in range(1,max(decoder_weights.keys())+1):
            bias_initializer = NumpyInitializer(decoder_bias[layer_number][0]) # create bias initializer for key's layer 
            kernal_initializer = NumpyInitializer(decoder_weights[layer_number]) # create weights initializer for key's layer 
            layer_size = len(decoder_bias[layer_number][0]) # get the size of the laye

            new_layer = tf.keras.layers.Dense( # initialize new Dense layer
                units = layer_size, 
                kernel_initializer=kernal_initializer, 
                bias_initializer = bias_initializer,
                activation="tanh")
            new_layer.trainable = False # optional: I was dealing with pretrained model so I disabled trainable
            x = new_layer(x) # stack layer at the top of the previous layer


        model = tf.keras.Model(inputs, x) # create tf's model based on the stacked layers 
        model.compile() # compile model 
        if(show):
            model.summary()
        return model

    def restore_model_from_directory(self, directory, encoder_template, weights_template, decoder_template, decoder_weights_template = None, decoder_reverse_order = True, show = False):
        encoder_bias, encoder_weights, decoder_bias, decoder_weights = self.get_numpy_values_from_dictionary(
            directory = directory,
            encoder_template = encoder_template, 
            weights_template = weights_template,
            decoder_template = decoder_template, 
            decoder_weights_template = decoder_weights_template, 
            decoder_reverse_order=decoder_reverse_order, 
            show = show)
        return  self.restore_model_from_numpy_dictionaries(encoder_bias, encoder_weights, decoder_bias, decoder_weights, show)


    def save(self, *args):
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
            if model in self.saved_models:
                print("[INFO]: This model was already saved!!!")
            else:
                self.saved_models.append(model)  # append list of saved models
                self.model_counter += 1  # get number of saved model
                model_name = str(
                    "model_" + str(self.model_counter)
                )  # consruct name of the model
                model_dir = os.path.join(self.models_directory_name, model_name)  # model dir
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
            self.dataset_counter += 1
            self.saved_datasets.append(dataset)
            dataset_dir = self.datasets_directory_name + "/training_dataset_{}.npy".format(self.dataset_counter)
            dataset_path = Path(self.datasets_directory_name)
            try:
                dataset_path.rmdir()
            except OSError as e:
                print(f"Error: {dataset_path} : {e.strerror}")
            dataset_path.mkdir(exist_ok=True, parents=True)
            if os.path.exists(dataset_path) and self.dataset_counter == 1:
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
            
    def load_models(self, *model_index):
        def get_model(model_index):
            model_dir = self.models_directory_name + "/model_{}".format(model_index)
            return tf.keras.models.load_model(model_dir)
            
        model_list = []
        for m_i in model_index:
            model_list.append(get_model(m_i))
        return model_list

    def load_datasets(self, *dataset_index):
        def get_dataset(dataset_index):
            dataset_dir = self.datasets_directory_name + "/training_dataset_{}.npy".format(dataset_index)
            return np.load(dataset_dir)

        datasets_list = []
        for d_i in dataset_index:
            datasets_list.append(get_dataset(d_i))
        return datasets_list
        
    def create_plot_path(self, name):
        "Create path to the plots' directory"
        return Path(os.path.join(self.plots_directory_name, name))

    def flip(self, solution, size=None, index=None):
            """
            Execute changes in the encoded representation of a solution.

            Parameters:
                solution - form of a container representing solution we modify 
                size - size of the solution 
            Oprionals: 
                index (None - random) - index which we would like to change 
            """
            if size == None: 
                size = len(solution)
            if index == None: 
                index = np.random.randint(size)
            solution[index] *= -1
            other_indx = np.random.randint(size)
            solution[other_indx] *= -1
            return

    def flip_and_update(self, current_solution, search, debuge_variation=False, *args):
        """
        Execute search function on the current_solution. 
        Update current_solution ifthe fitness value of 
        the modified solutin is greater, else does no change 
        
        Parameters:
            current_solution - container holding solution we modify 
            search - function searching and modyfing encoded representation 
        Optionals: 
            debug_variation (False) - turning on/off debug mode 
        Return: 
            conteiner, with the same type as current_solution
        """
        size = len(current_solution) # get the length of the solutin 
        rand_index = np.random.randint(size) # pick up a random index 
        new_solution = np.copy(current_solution) # copy current_solution 
        search(new_solution, *args) # execute search function on the copied solutino 
        new_fitness = self.fitness_function(new_solution) # calculate fitness of the new solution
        if new_fitness >= self.fitness_function(current_solution): 
            current_solution = new_solution # update current solutin if fitness is better 
        if debuge_variation: 
            print("New: ", new_fitness)
        return current_solution
        
    def initialize_solution(self, size):
        """
        Return array of (-1)'s
        """
        return np.zeros(size)-1

    def generate_training_set(self , sample_size, set_size, learning_steps_coef = 10, debuge_variation=False):
        """
        Generate training set. 

        Patameters: 
            sample_size - size of a solution 
            set_size - number of elements in the dataset 
        Optionals: 
            laerning_steps_coef (10) - it * size of the solution will give us number serach
            debug_variation (False) - turning on/off debug mode

        return np.ndarray of training elements

        """
        training_set = np.ndarray(shape=(set_size, sample_size)) # initialize container 
        number_of_search = int(sample_size * learning_steps_coef) # get the number of search 
        for i in range(set_size): 
            
            current_solution = self.initialize_solution(sample_size) # initialize new solutin 
            for k in range(number_of_search): # execute search 
                current_solution = self.flip_and_update(current_solution, self.flip, False) #search and update solutin 
            if self.fitness_function(current_solution)>0: # if fitness is below 0 (does not fulfill constrains)
                training_set[i] = current_solution          # reject such sample 
            else:
                i -=1
            if debuge_variation:
                print(i, self.fitness_function(current_solution))

        return training_set

    def rand_bin_array(self, K, N):
        """
        THe function return random binary string. 
        The string has K - 0's adn N-K - 1's
        """
        arr = np.zeros(N)
        arr[:K] = 1
        np.random.shuffle(arr)
        return arr

    def hiff_fitness(self, array):

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
    

"""
def generate_training_sat(N, set_size, debuge_variation=False):
    Generate training set for H-IFF problem. 
    
    return: binary array of size N to train NN
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
    """
