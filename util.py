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

import os


def save_model(model):
    """
    Save model in the directory saved_model. 
    The model will not be saved if it's alreday saved. 
    
    Parameters: 
        model - TF's model 
    
    """
    if model in save_model.saved_models:
        print("[INFO]: This model was already saved!!!")
    else:
        save_model.saved_models.append(model)  # append list of saved models
        save_model.counter += 1  # get number of saved model
        model_name = str(
            "model" + str(save_model.counter)
        )  # consruct name of the model
        model_dir = os.path.join("saved_model", model_name)  # model dir
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


save_model.counter = 0
save_model.saved_models = []


def create_plot_path(name):
    return Path(os.path.join("plots", name))


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


def generate_training_sat(N, set_size):
    """
    Generate training set for H-IFF problem. 
    
    return: binary array of size N to train NN
    """
    input = np.ndarray(shape=(set_size, N))
    output = np.ndarray(shape=(set_size, N))

    if not (math.log2(N)).is_integer():
        raise ValueError("Array size must be power of 2.")
    for k in range(set_size):
        candidate_solution = np.random.randint(2, size=N)
        input[k] = candidate_solution
        solution_fitness = hiff_fitness(candidate_solution)
        for i in range(10 * N):
            index = np.random.randint(N)
            new_candidate_sol = copy.copy(candidate_solution)
            new_candidate_sol[index] = 1 - new_candidate_sol[index]  # apply variation
            new_fitness = hiff_fitness(new_candidate_sol)  # check the change
            if new_fitness >= solution_fitness:
                candidate_solution = new_candidate_sol
                solution_fitness = new_fitness
        output[k] = candidate_solution

    return input, output


def split_model_into_encoder_decoder(model, show_summary=False):
    """
    Extract encoder and decoder from the model.
    The model is splited around the bottle neck. 

    Parameters:
        model - model
        show_summary = False - show summary of encoder and decoder 

    Returns:
        encoder, decoder
    """
    print("[INFO]: Extracting encoder and decoder from the model")
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


def code_flip_decode(array, encoder, decoder, debuge_variation=False):
    """
    Apply random bit flip in the latent space. 
    encode -> flip - > decode 

    Parameters: 
        array - array representing binary array 
        encoder - encoder reducing dimensionality
        decoder - decoder retrieving values from the latent space 
        debuge_variation - show info useful fo debuging 

    Returns: 
        output_tensor, output_array_binary, new_fitness
    """
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
        new_tensor.numpy()[-1] > 0.5, 1, 0
    )  # binarize decoded tensor around 0.5
    new_fitness = hiff_fitness(
        output_array_binary
    )  # calculate transformed tensor fitness
    output_tensor = tf.convert_to_tensor(
        output_array_binary.reshape((1, N)), dtype=tf.float32
    )  # save output tensor
    if debuge_variation:
        print(
            "Input fitness: ",
            hiff_fitness(array),
            ", Decoded fitness: ",
            hiff_fitness(output_array_binary),
        )
        print("Input: ", array)
        print("Encoded: ", new_array)
        print("Encoded fliped, index: ", index, " : ", new_array_fliped)
        print("Decoded: ", output_array)
        print("Decoder binary: ", output_array_binary, "\n")
    return output_tensor, output_array_binary, new_fitness


def transfer_sample_latent_flip(array, encoder, decoder):
    """
    Execute random bit flip in the latent space for 10 * size_of_sample, and
    update the sample if the fitness after the flip and decoding has improved

    Parameters:
        array - sample to encode->flip->decoder 
        encoder - encoder reducing dimensionality
        decoder - decoder retrieving values from the latent space     

    Returns:
        array - improved initial sample with greater fitness   
    """
    N = np.shape(array)[-1]
    current_fitness = hiff_fitness(array)
    progress_holder = []
    normalization_factor = hiff_fitness(np.ones((N,)))

    for i in range(10 * N):
        output_tensor, output_array, new_fitness = code_flip_decode(
            array, encoder, decoder
        )
        progress_holder.append(new_fitness)
        if new_fitness >= current_fitness:  # compare flip with current  fitness
            current_fitness = new_fitness
            array = output_array
    return array, np.divide(progress_holder, normalization_factor)


#
def generate_new_training_set(initial_training_set, encoder, decoder):
    """
    Generate training set based on the transfer_sample_latent_flip method,
    which enhance quality of the samples

    Parameters: 
        initial_training_set - training set on which latent space modification happens 
        encoder - encoder reducing dimensionality
        decoder - decoder retrieving values from the latent space     

    Returns:
        imporoved_training_set (numpy array)  
    """
    print("[INFO]: Generating new enhanced data set")
    new_trainig_set = []
    N = np.shape(initial_training_set)[-1]
    for array in initial_training_set:
        new_trainig_set.append(transfer_sample_latent_flip(array, encoder, decoder)[0])
    return np.asarray(new_trainig_set, dtype=np.float32)


def add_layer_to_model(
    model, compression=0.8, dropout=0.2, reg_cof=0.001, show_summary=False
):
    """
    To do: 
     - add activation parameter
     - add stddev param
     - add initializer param 
    Add new layer to the middle of the model.

    Parameters:
        model - model to which we would like to add new layer 

    Optional parameters:
        compression (default 0.8) - level of compression compared to the latent space of the model 
        droupout (default 0.2) - dropout of the drop layer before new latent layer 
        reg_cof (default 0.001) - reguralization coefficient for new latent space 
        show_summary (default False) - variable to show new model structure
    """
    encoder_old, decoder_old = split_model_into_encoder_decoder(
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

    x = encoder_old.layers[1](inputs)  # add first layers: input -> encoder.layers[1]
    for e in encoder_old.layers[2:]:  # add model encoder
        x = e(x)

    x = new_latent_layer(x)  # add latent laver
    x = new_decoding_layer(x)  # add transition "diverging" layer
    for d in decoder_old.layers[1:]:  # add model decoder
        x = d(x)

    new_model = tf.keras.Model(inputs, x)  # construct model
    opt = Adam(lr=0.01)  # set up optimizer
    new_model.compile(loss="mse", optimizer=opt)  # compile model
    if show_summary:
        new_model.summary()
    return new_model


def generate_trajectory_plot(
    encoder, decoder, array, target_size=10, learning_steps=30,
):
    normalization_factor = hiff_fitness(np.ones((np.shape(array)[-1],)))
    trajectory_samples = []
    modified_data_set = np.ndarray(shape=(target_size, np.shape(array)[-1]))
    for k in range(target_size):
        current_array = array[k]
        current_fitness = hiff_fitness(current_array)
        current_target_trajectory = []
        current_target_trajectory.append(current_fitness / normalization_factor)
        for i in range(learning_steps - 1):
            output_tensor, output_array, new_fitness = code_flip_decode(
                current_array, encoder, decoder
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
Everything below is for improvement
"""


def generate_evol_plot(N=32, path="solution_development_plot.png", learning_steps=50):
    candidate_solution = np.random.randint(2, size=N)
    sol_evol = []
    sol_evol.append(candidate_solution)
    max_fitt = 0
    for i in range(learning_steps):
        index = np.random.randint(N)
        current_fit = hiff_fitness(candidate_solution)
        index = np.random.randint(N)
        new_candidate_sol = copy.copy(candidate_solution)
        new_candidate_sol[index] = 1 - new_candidate_sol[index]
        new_fit = hiff_fitness(new_candidate_sol)
        if new_fit >= current_fit:
            candidate_solution = new_candidate_sol

        sol_evol.append(candidate_solution)

    tmp = np.asarray(sol_evol)
    plt.figure()
    plt.imshow(tmp, interpolation="nearest", cmap=cm.Greys_r)
    plt.title("Solution Development at Evolution Step 1")
    plt.xlabel("Solution variable")
    plt.ylabel("Development Step")
    plt.colorbar()
    plt.savefig(path)


def generate_sol_plot(
    N=32, target_size=10, path="trajectory_plot.png", learning_steps=70
):
    X = np.arange(learning_steps)
    normalization_factor = hiff_fitness(np.ones((N,)))
    plt.figure()
    plt.title("Example Solution Trajectory at Evolution Step 1")
    for k in range(target_size):
        candidate_solution = np.random.randint(2, size=N)
        solution_fitness = hiff_fitness(candidate_solution)
        current_target_trajectory = [solution_fitness / normalization_factor]
        for i in range(learning_steps - 1):
            index = np.random.randint(N)
            new_candidate_sol = copy.copy(candidate_solution)
            new_candidate_sol[index] = 1 - new_candidate_sol[index]  # apply variation
            new_fitness = hiff_fitness(new_candidate_sol)  # check the change
            if new_fitness >= solution_fitness:
                candidate_solution = new_candidate_sol
                solution_fitness = new_fitness
                current_target_trajectory.append(new_fitness / normalization_factor)
            else:
                current_target_trajectory.append(current_target_trajectory[-1])
        plt.plot(X, current_target_trajectory)
    plt.xlabel("learning step")
    plt.ylabel("fitness \ max_fitness")
    plt.savefig(path)
