import numpy as np
import math
import copy
import os
import shutil
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Reshape
from shallowNet.shallowNet import shallowNet, DenseTranspose
from tensorflow.keras.optimizers import Adam
from KnapSack import KnapSack
import utilsGeneral as utg
from numpy.ma import masked_array


class UtilsModel:
    def __init__(self, utg):
        self.utg = utg
        self.search = self.utg.flip
        self.knapSack = utg.knapSack
        self.fitness_function = self.knapSack.Fitness

    def train_model(self, trainingSet, epochs = 500, compression = 0.8,  batch_size = 10, reg_cof = (0.0001,0.001), dropout = 0.2, lr = 0.001, validation_split = 0.05, metrics = tf.keras.metrics.RootMeanSquaredError() ):
        modelTmp = shallowNet.build(
            input_shape=self.knapSack.Size, 
            reg_cof= reg_cof, 
            lr = lr, 
            dropout= dropout, 
            compression=compression, 
            metrics = metrics)

        H1 = modelTmp.fit(
            trainingSet, trainingSet, 
            validation_split = 0.1,
            epochs=epochs, 
            batch_size=batch_size, 
            shuffle=True,
            verbose=0)
        return modelTmp, H1

    def code(self, array, encoder, input_size=None, latent_size=None, debuge_variation=False):
        """
        Code solution vector to the latent representation: 
        Parameters: 
            array - numpy ndarray to code 
            encoder - tf's model to encode 
        
        Optional Parameters: 
            input_size - size of the array
            latent_size - size of the latent space 
            debug_variation (False) - activate debug mode 
        """
        if input_size == None:
            input_size = len(array) # if input_size is implicit do not waist time to calcule it
        if latent_size == None:
            latent_size = np.shape(encoder.layers[-1].get_weights()[0])[-1] # if latent_size is implicit do not waist time to calcule it
        encoded_solution = encoder(np.expand_dims(array, axis = 0)).numpy().flatten() # encode array 
        return encoded_solution

    def decod(self, encoded_solution, decoder):

        """
        Decode solution from the latent representation to the input form. 
        Decoded solution is discretized [-1, 1] around 0.  
        
        Parameters: 
            encoded_solution - numpy ndarray to decode 
            encoder - tf's model to encode 
        """
        new_tensor = decoder(encoded_solution.reshape(1,len(encoded_solution))) # decode changed solution 
        output_array_binary = np.where(new_tensor.numpy()[-1] > 0.0, 1, -1)  # binarize decoded tensor around 0.0
        new_fitness = self.fitness_function(output_array_binary) # calculate new fitness
        return output_array_binary, new_fitness  

    def split_model_into_encoder_decoder(self, model, index_to_split = None, show = False):
        """
        Extract encoder and decoder from the model.
        The model is splited around the bottle neck or index_to_split. 

        Parameters:
            model - model
            index_to_split (None) - index of layer around which model will be splited (None - around bottle neck)
            show (False) - show summary of encoder and decoder 
        Returns:
            encoder, decoder of types tensorflow.model
        """
        def create_model_from_list_of_layers(*args):
            model_ = tf.keras.Sequential([*args])
            model_.compile()
            return model_

        if index_to_split == None:
            index_to_split = np.argmin([x.output.shape[-1] for x in model.layers])+1

        encoder = create_model_from_list_of_layers(*model.layers[:index_to_split])
        decoder_layers = [tf.keras.layers.InputLayer(input_shape = (len(model.layers[index_to_split-1].get_weights()[1]),))]
        decoder_layers += model.layers[index_to_split:]
        decoder = create_model_from_list_of_layers(*decoder_layers)
        
        if show:
            print("---------------------------- ENCODER ----------------------------")
            encoder.summary()
            print("\n---------------------------- DECODER ------------------------")
            decoder.summary()
        
        return encoder, decoder
        
    def code_flip_decode(self, array, encoder, decoder, search = None, input_size=None, latent_size=None,  debuge_variation=False):
        """
        Apply random bit flip in the latent space. 
        encode -> flip -> decode 

        Parameters: 
            array - sample binary array which will be decoded->searched->decoded 
            encoder - encoder reducing dimensionality
            decoder - decoder retrieving values from the latent space 
            search - function to change encoded representation
            input_size - size of the input (faster calculations)
            latetnt_size - size of the laten (faster calculations)
            debuge_variation - show info useful fo debuging 

        Returns: 
            output_array_binary - binarized array efter going through encoder->decoder
            new_fitness - fitness of the ourput_array_binary 
        """
        if input_size == None:
            input_size = len(array) # if input_size is implicit do not waist time to calcule it
        if latent_size == None:
            latent_size = np.shape(encoder.layers[-1].get_weights()[0])[-1] # if latent_size is implicit do not waist time to calcule it
        encoded_solution = encoder(np.expand_dims(array, axis = 0)).numpy().flatten() # encode array 
        if search == None:
            self.search(encoded_solution, latent_size) # modify encoded representation using default search function
        else:
            search(encoded_solution, latent_size) # modify encoded representation using passed function 
        new_tensor = decoder(encoded_solution.reshape(1,latent_size)) # decode changed solution 
        output_array_binary = np.where(new_tensor.numpy()[-1] > 0.0, 1, -1)  # binarize decoded tensor around 0.0
        new_fitness = self.fitness_function(output_array_binary) # calculate new fitness
        
        if debuge_variation: # show info for debuging 
            print(
                "Input fitness: ",
                self.fitness_function(array),
                ", Decoded fitness: ",
                self.fitness_function(output_array_binary),
            )
            print("Input: ", array)
            print("Encoded: ", encoded_solution)
            print("Decoded: ", new_tensor.numpy())
            print("Decoder binary: ", output_array_binary, "\n")

        return output_array_binary, new_fitness

    def transfer_sample_latent_flip(self, array, encoder, decoder, search=None, learning_steps_coef = 10, normalization_factor = 1, debuge_variation=False):
        """
        Execute search function in the latent space for 10 * size_of_sample times
        Update the sample if the fitness after the flip and decoding improve 

        Parameters:
            array - sample array to encode->search->decode
            encoder - model encodeing a solution 
            decoder - model decodeing a solution 
        Optionals:
            search (self.search) - function executing search in encoded solution 
            laerning_steps_coef (10) - it * size of the array will give us number serach
            normalization_factor (1) - factor which normalize all results 
            debuge_variation (False) - variable indicating debug mode 

        Returns:
            array - improved initial sample with greater fitness   
        """
        N = np.shape(array)[-1] # get the length of the arra y
        current_fitness = self.fitness_function(array) # calculare current solution fitness 
        progress_holder = [] # initialize list hodling changes 
        learning_steps = N * learning_steps_coef # calculate number of steps to transfer sample 
        if search == None: 
            search = self.search # if search function is not specified get the default one 

        for i in range(learning_steps):
            output_array, new_fitness = self.code_flip_decode( # move soultion through encoder -> decoder,
                array, encoder, decoder, search= search         # modifying encoded representation 
            )
            progress_holder.append(new_fitness) # append container holding history of modyfication 
            if new_fitness >= current_fitness:  # compare flip with current  fitness
                current_fitness = new_fitness
                array = output_array
            if debuge_variation:
                print("Current fitness: ", current_fitness)
        return array, np.divide(progress_holder, normalization_factor)

    def generate_enhanced_training_set(self, model, initial_training_set, search = None, dataset_split = 1.0):
        """
        Generate training set based on the transfer_sample_latent_flip method,
        which enhance quality of the samples

        Parameters:
            model - used to encode and decoder solutions  
            initial_training_set - training set on which latent space modification happens 
        Optinals:    
            search (self.serach) - function modyfing latent representation 
            dataset_split (1.0) - fraction indicating number of elements in the new dataset
                                    with respect to the size of the initial_training_set 

        Returns:
            imporoved_training_set of type (numpy array)  
        """
        print("[INFO]: Generating new enhanced data set")
        assert dataset_split <=1.0 and dataset_split > 0, "dataset_split must be between (0,1]" # validate dataset_split value 
        encoder, decoder = self.split_model_into_encoder_decoder(model) # split the model into encoder and decoder 
        new_trainig_set = [] # initialize holder for new training set 
        if search == None: 
            search = self.search # if search function is not specified get the default one 
        if dataset_split == 1.0:
            for array in initial_training_set: # execute for every element form the initial_training_set
                new_trainig_set.append(self.transfer_sample_latent_flip(
                    array = array, 
                    encoder = encoder, 
                    decoder = decoder, 
                    search = search)[0])
        else: 
            new_dataset_size = int(np.shape(initial_training_set)[0] * dataset_split)
            for i in range(new_dataset_size): # execute for a fraction of elements form the initial_training_set
                new_trainig_set.append(self.transfer_sample_latent_flip(
                    array = initial_training_set[i], 
                    encoder = encoder, 
                    decoder = decoder, 
                    search = search)[0])

        return np.asarray(new_trainig_set, dtype=np.float32)

    def add_layer_to_model(self, model, compression=0.8, dropout=0.2, reg_cof=0.001,lr = 0.001, show_summary=False):
        """
        Add new layer to the middle of the model. 

        Parameters:
            model - model to which we would like to add new layer 

        Optional parameters:
            compression (default 0.8) - level of compression compared to the latent space of the model 
            droupout (default 0.2) - dropout of the drop layer before new latent layer 
            reg_cof (default 0.001) - reguralization coefficient for new latent space 
            show_summary (default False) - variable to show new model structure
        """
        encoder_old, decoder_old = self.split_model_into_encoder_decoder(
            model
        )  # split the old model into encoder and decoder
        latent_size = encoder_old.layers[-1].output.shape[-1]  # calculate new latent size
        input_shape = model.input_shape[-1]  # extract input shape
        inputs = Input(shape=(input_shape,))  # explicitly define input layer

        new_latent_layer = Dense(  # new latent layer
            int(latent_size * compression),
            activation="tanh",
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            bias_initializer=tf.keras.initializers.Zeros(),
            kernel_regularizer=tf.keras.regularizers.l1(reg_cof),
        )
        new_decoding_layer = DenseTranspose(
            dense=new_latent_layer
        )  # create dependant "diverging" layer
        new_dropout_layer = Dropout(dropout)

        x = encoder_old.layers[1](inputs)  # add first layers: input -> encoder.layers[1]
        for e in encoder_old.layers[2:]:  # add model encoder
            x = e(x)
        x = new_dropout_layer(x) # add dropout layer
        x = new_latent_layer(x)  # add latent laver
        x = new_decoding_layer(x)  # add transition "diverging" layer
        for d in decoder_old.layers[1:]:  # add model decoder
            x = d(x)

        new_model = tf.keras.Model(inputs, x)  # construct model
        opt = Adam(lr=lr)  # set up optimizer
        new_model.compile(loss="mse", optimizer=opt)  # compile model
        if show_summary:
            new_model.summary()
            
        return new_model

    def generate_trajectory_plot(self, encoder, decoder, array, target_size=10, learning_steps=30,normalization_factor = 1):
        #normalization_factor = self.fitness_function(np.ones((np.shape(array)[-1],)))
        trajectory_samples = []
        modified_data_set = np.ndarray(shape=(target_size, np.shape(array)[-1]))
        for k in range(target_size):
            current_array = array[k]
            current_fitness = self.fitness_function(current_array)
            current_target_trajectory = []
            current_target_trajectory.append(current_fitness / normalization_factor)
            for i in range(learning_steps - 1):
                output_array, new_fitness = self.code_flip_decode(
                    current_array, encoder, decoder, search= self.search
                )
                if new_fitness >= current_fitness:
                    current_fitness = new_fitness
                    current_array = output_array
                    current_target_trajectory.append(new_fitness / normalization_factor)
                else:
                    current_target_trajectory.append(current_target_trajectory[-1])

            modified_data_set[k] = current_array
            trajectory_samples.append(current_target_trajectory)
        return modified_data_set, np.asarray(trajectory_samples)

    def correlation_matrix(self, model, activation=1, background_activation =-1, threshold = None):
            """
            Return "corellation" matrix. It compares encoded representation comming from 
            different single bits activation (sample with i-th bit set to activation's value and 
            all others set to back_ground's value). "Correlation" is calculated as RMSE with 
            some threshodl value to alliviate noise influence. 

            Parameters: 
                model - tf's model to evaluetion  
                activation (1) - activation of a single bit 
                background_activation (-1) - activation of the remaining bits
                threshodle (None) - threshodl set to std, else threshold might be passed manually 
            Returns: 
                Corelation matrix plot 
            """
            def latent_activation(model):
                encoder, decoder = self.split_model_into_encoder_decoder(model)
                size = encoder.input_shape[-1]
                res = []
                for i in range(size):
                    arr = np.zeros(size) + background_activation
                    arr[i] = activation
                    res.append(self.code(arr, encoder))
                return res

            def correlation(array1, array2, threshold = threshold):
                arr = array1-array2
                if threshold == None:
                    threshold = np.std(arr)
                arr=np.where(((arr < threshold) & (arr > -threshold)), 0, arr)
                return np.sqrt((arr**2).mean())
        
        

            encoded_bit_representation = latent_activation(model)
            sample_numbers = np.shape(encoded_bit_representation)[0]
            conv = []
            for i in range(sample_numbers):
                tmp = []
                for j in range(sample_numbers):
                    tmp.append(
                        correlation(
                            encoded_bit_representation[i], 
                            encoded_bit_representation[j]
                            ))
                conv.append(np.asarray(tmp))
            return np.asarray(conv)

    def weights_matrix_sorting(self, model, index_to_split = None, activation=1, background_activation = -1, column_sort = True, unit_sort=True, index_sort = False, log_conversion = False):
        """
        Transofrm weight matrix for the sake of visibility. 

        Parameters: 
            model - tf's model from which weight matrix will be displayed 
            model_split_index (None) - 
        
            activation (1) - activation of a single bit 
            background_activation (-1) - activation of the remaining bits
            column_sort (True) -  sort weight by columns (True), sort weights by rows (False)
            unit_sort (True) - sort by the magnitude (absolut value) of weights (true), else sorth by signs (false)
            index_sort (False) - shuffle columns according to the probability of encoded neuron actiavtion (True)
            log_conversion (False) - convert values of weights to discrete log values (True), work on the linear scale (False)
        
        Returns:
            if unit_sort = True:
                return hidden_nodes_activation_distribution, weights_matrix, sorted_weights_matrix
            if unit_sort = False:
                return hidden_nodes_activation_distribution, weights_matrix, 
                negative_weights_negative_mask, positive_weights_negative_mask ,
                sorted_negative_weights_matrix, sorted_positive_weights_matrix
        """
        def extract_weight_matrix(weights):
            if type(weights) == type(np.array([[1],[1]])) and len(np.shape(weights))==2:
                return weights
            for w in weights:
                if(len(np.shape(w))==2):
                    return w
        def value(x):
            if log_conversion:
                if x==0 or x ==-0:
                    return 0
                return round(math.log(abs(x),10))
            else:
                if(x<0):
                    return -x
                return x

        def shuffle_columns_weights_matrix(decoder, pos): 
            #matrix  = decoder.layers[1].get_weights()[1]
            matrix  = extract_weight_matrix(encoder.layers[-1].get_weights())
            
            if np.shape(matrix)[0]<np.shape(matrix)[1]:
                matrix = matrix.transpose()
            mat = np.full_like(matrix, 0)
            index = 0
            for i in pos:
                mat[:, index] = matrix[:, i]
                index +=1
            return mat

        def matrix_row_sort(x):
            def pushZerosToEnd(arr): 
                count = 0 # Count of non-zero elements 
                n = len(arr)
                for i in range(n): 
                    if arr[i] != 0: 
                        arr[count] = arr[i] 
                        count+=1
                while count < n: 
                    arr[count] = 0
                    count += 1
                return arr
            
            def pushZerosToBegining(arr): 
                count = 0 # Count of non-zero elements 
                tmp = []
                count_zeros =0
                for i in arr.tolist(): 
                    if i != 0: 
                        tmp.append(i)
                    else:
                        count_zeros+=1
                return np.concatenate((np.zeros(count_zeros), np.asarray(tmp)), axis = None)

            pos = []
            neg = []

            if unit_sort:
                for i in x:
                    if i>0:
                        pos.append(value(i))
                    else:
                        pos.append(value(-i))
                return np.sort(pos)

            else:
                for i in x:
                    if i>0:
                        pos.append(value(i)) 
                        neg.append(0)
                    else:
                        neg.append(value(-i))
                        pos.append(0)
                return pushZerosToEnd(-np.sort(-np.asarray(neg)[::-1])), pushZerosToBegining(np.sort(np.asarray(pos)))            
            
            
        encoder, decoder = self.split_model_into_encoder_decoder(model, index_to_split)
        size = encoder.layers[0].input_shape[-1]
        latent_size = np.shape(encoder.layers[-1].get_weights()[0])[-1]
        res = np.zeros(latent_size)
        glob_pos = {}
        for i in range(size):
            arr = np.zeros(size) + background_activation
            arr[i] = activation
            res += np.where(self.code(arr, encoder) > 0.0, 0, 1)/size
        
        for i in range(latent_size):
            glob_pos[i] = res[i]

        if index_sort:
            pos = {k : v for k, v in sorted(glob_pos.items(), key = lambda item : item[1])}
            res.sort() # sort bits accorg
        else:
            pos = np.arange(latent_size)

        if column_sort:        
            weights_matrix = shuffle_columns_weights_matrix(decoder, pos).transpose()
        else:
            weights_matrix = shuffle_columns_weights_matrix(decoder, pos) 


        if unit_sort:
            sorted_weights_matrix = copy.copy(weights_matrix)
            for i in range(np.shape(sorted_weights_matrix)[0]):
                sorted_weights_matrix[i] = matrix_row_sort(sorted_weights_matrix[i])
            if column_sort:
                weights_matrix = weights_matrix.transpose()
                sorted_weights_matrix = sorted_weights_matrix.transpose()

            return res, weights_matrix, sorted_weights_matrix

        else:
            sorted_weights_matrix_neg = copy.copy(weights_matrix)
            sorted_weights_matrix_pos = copy.copy(weights_matrix)

            for i in range(np.shape(sorted_weights_matrix_pos)[0]):
                sorted_weights_matrix_neg[i], sorted_weights_matrix_pos[i] = matrix_row_sort(sorted_weights_matrix_neg[i])

            if column_sort:
                weights_matrix =weights_matrix.transpose()
                neg_mask = masked_array(sorted_weights_matrix_neg.transpose(), sorted_weights_matrix_neg.transpose()==0)
                pos_mask = masked_array(sorted_weights_matrix_pos.transpose(), sorted_weights_matrix_pos.transpose()==0)
            else: 
                neg_mask = masked_array(sorted_weights_matrix_neg, sorted_weights_matrix_neg==0)
                pos_mask = masked_array(sorted_weights_matrix_pos, sorted_weights_matrix_pos==0)

            return res, weights_matrix, neg_mask, pos_mask ,sorted_weights_matrix_neg, sorted_weights_matrix_pos

