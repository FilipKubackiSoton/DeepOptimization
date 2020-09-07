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
        The model is splited around the bottle neck. 

        Parameters:
            model - model
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


"""
    def split_model_into_encoder_decoder(self, model, show_summary=False):

        if show_summary:
            print("[INFO]: Extracting encoder and decoder from the model")
            print(type(model))
        layer_to_split = model.layers[0]
        index_to_split = 0
        # the code here might be simpler stoping at the denseTranspose type, but its more robust
        for i in range(len(model.layers)):
            if model.layers[i].output.shape[-1] <= layer_to_split.output.shape[-1]:
                layer_to_split = model.layers[i]
                index_to_split = i
        index_to_split += 1

        inputs_encoder = model.inputs
        x = inputs_encoder
        for new_layer_encoder in model.layers[1:index_to_split]:
            x = new_layer_encoder(x)
        encoder_ = tf.keras.Model(inputs_encoder, x)

        latent_shape = encoder_.layers[-1].output.shape[-1]
        inputs_decoder = Input(shape=latent_shape)
        y = inputs_decoder
        for new_layer_decoder in model.layers[index_to_split:]:
            y = new_layer_decoder(y)
        decoder_ = tf.keras.Model(inputs_decoder, y)
        if show_summary:
            print("---------------------------- ENCODER ----------------------------")
            encoder_.summary()
            print("\n---------------------------- DECODER ------------------------")
            decoder_.summary()
        return encoder_, decoder_
"""




"""
def code_flip_decode(array, encoder, decoder, debuge_variation=False):
    
    Apply random bit flip in the latent space. 
    encode -> flip - > decode 

    Parameters: 
        array - array representing binary array 
        encoder - encoder reducing dimensionality
        decoder - decoder retrieving values from the latent space 
        debuge_variation - show info useful fo debuging 

    Returns: 
        output_tensor, output_array_binary, new_fitness
    
    N = np.shape(array)[-1]
    new_array = (
        encoder(tf.expand_dims(array, 0))[-1].numpy().flatten()
    )  # encode a sample
    # new_array_binary = np.where(new_array>0, 1, 0) # binarize latent representation
    index = np.random.randint(np.shape(new_array)[-1])  # choose random index to flip
    new_array_fliped = copy.copy(new_array)  # create copy of the encoded array
    new_array_fliped[index] *= -1  # apply flip
    changed_tensor = tf.convert_to_tensor(
        tf.expand_dims(new_array_fliped, 0)
    )  # create new tensor
    new_tensor = decoder(
        changed_tensor
    )  # decode the sample with the change from the latent spaece
    output_array = new_tensor.numpy()[-1]  # extraxt simple 1D array from tensor
    output_array_binary = np.where(
        new_tensor.numpy()[-1] > 0.0, 1, -1
    )  # binarize decoded tensor around 0.0
    new_fitness = self.fitness_function(
        output_array_binary
    )  # calculate transformed tensor fitness
    output_tensor = tf.convert_to_tensor(
        output_array_binary.reshape((1, N)), dtype=tf.float32
    )  # save output tensor
    if debuge_variation:
        print(
            "Input fitness: ",
            self.fitness_function(array),
            ", Decoded fitness: ",
            self.fitness_function(output_array_binary),
        )
        print("Input: ", array)
        print("Encoded: ", new_array)
        print("Encoded fliped, index: ", index, " : ", new_array_fliped)
        print("Decoded: ", output_array)
        print("Decoder binary: ", output_array_binary, "\n")
    #output_array_binary = knapSack.SolToTrain(output_array_binary)
    return output_tensor, output_array_binary, new_fitness
"""


