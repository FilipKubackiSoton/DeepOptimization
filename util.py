import numpy as np 
import math 
import copy
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import tensorflow as tf
from pathlib import Path
import os

def create_plot_path(name):
    return Path(os.path.join("plots",name))

def rand_bin_array(K, N):
    """
    THe function return random binary string. 
    The string has K - 0's adn N-K - 1's
    """
    arr = np.zeros(N)
    arr[:K]  = 1
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
        for i in range(0,2**(levels - flor)-1,2):
            arr.append(t(array[i], array[i+1]))
            sum = sum + (f(array[i]) + f(array[i+1]))* power
        return val_recursive(arr, flor + 1, sum)

    size = len(array)
    if not (size/2).is_integer():
        raise ValueError("Array size must be power of 2.")
    levels = int(math.log2(size))
    sum = 0
    return val_recursive(array, 0,  sum)
        

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
        candidate_solution = np.random.randint(2, size = N)
        input[k]=candidate_solution
        solution_fitness = hiff_fitness(candidate_solution)
        for i in range(10 * N):
            index = np.random.randint(N)
            new_candidate_sol = copy.copy(candidate_solution)
            new_candidate_sol[index] = 1 - new_candidate_sol[index] # apply variation 
            new_fitness = hiff_fitness(new_candidate_sol) # check the change 
            if new_fitness >= solution_fitness : 
                candidate_solution = new_candidate_sol
                solution_fitness = new_fitness
        output[k]=candidate_solution

    return input, output

def plot_model_loss(model_fit, plot_name, epochs ):
    """
    Plot model loss to epochs

    Parameters: 
        model_fit - result of the model training 
        plot_name - name of the saving file 
    """
    # construct a plot that plots and saves the training history
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, epochs), model_fit.history["loss"], label="train_loss")
    plt.title("Training Loss and Accuracy\n")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    path = create_plot_path(plot_name)
    plt.savefig(path)
    print("[INFO]: Loss plot was saved in the directory: ", path)

def plot_weights_mode(model, plot_name):
    """
    Plot weight matrix of encoder and decoder

    Parameters: 
        model - model on which we are working 
        plot_name - name of the saving file 
    """
    fig, axs = plt.subplots(1,2)
    fig.suptitle('Weights matrix encoder/decoder', fontsize=16)
    fig.subplots_adjust(hspace =0.7)
    pcm = axs[0].imshow(model.get_weights()[0], interpolation='nearest', cmap=cm.Greys_r)
    axs[0].set_title("Encoder")
    axs[0].set_ylabel("Visible Node #")
    axs[0].set_xlabel("Hidden Node #")

    pcm=axs[1].imshow(model.get_weights()[0], interpolation='nearest', cmap=cm.Greys_r)
    axs[1].set_title("Decoder")
    axs[1].set_ylabel("Visible Node #")
    axs[1].set_xlabel("Hidden Node #")
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(pcm, cax=cbar_ax)
    path = create_plot_path(plot_name)
    plt.savefig(path)
    print("[INFO]: Weight plot was saved in the directory: ", path)

def plot_latent_acitvation(model, plot_name, validation_set_size = 50):
    """
    Plot latent activation 

    Parameters: 
        model - model on which we are working 
        plot_name - name of the saving file 
    """
    # generate the val set 
    print("[INFO] generating validating dataset...")
    (valX, valY) = generate_training_sat(32, validation_set_size)

    features_list = [layer.output for layer in model.layers[:4]]
    new_model = tf.keras.Model(inputs = model.input, outputs = features_list)
    predict = new_model.predict(valX)
    N = np.arange(0, len(predict[3][0]))

    plt.figure()
    for i in range(20):
        index = np.random.randint(len(predict[3][0]))
        plt.plot(N, predict[3][index], 'o',color = 'black')
    plt.title("L1 activation")
    plt.xlabel("Node #")
    plt.ylabel("Activation value")
    path = create_plot_path(plot_name)
    plt.savefig(path)
    print("[INFO]: Latent activation plot was saved in the directory: ", path)

def extract_encoder_and_decoder(model):
    """
    Still to improve - poor model split
    Extract encoder and decoder from the model (simple version to improve)

    Parameters:
        model - model

    Returns:
        encoder, decoder
    """
    print("[INFO]: Extracting encoder and decoder from the model")
    encoder = tf.keras.Model(inputs = model.input, outputs = [layer.output for layer in model.layers[:3]]) # extract layers from the model and stack them to form an encoder 
    features_list_decoder = model.layers[3]# [layer.output for layer in model.layers[3]]
    inputs = tf.keras.layers.Input(encoder.layers[-1].output_shape[-1])
    features_list_decoder(inputs)
    decoder = tf.keras.Model( inputs = inputs ,outputs = features_list_decoder(inputs)) # extract layers from the model and stack them to form an decoder 
    print("-----------------ENCODER-----------------\n")
    encoder.summary()
    print("\n-----------------DECODER-----------------\n")
    decoder.summary()
    return encoder, decoder

def code_flip_decode(array, encoder, decoder, debuge_variation = False):
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
    new_array = encoder(tf.expand_dims(array,0))[-1].numpy().flatten() #encode a sample 
    #new_array_binary = np.where(new_array>0, 1, 0) # binarize latent representation 
    index = np.random.randint(np.shape(new_array)[-1]) #choose random index to flip 
    new_array_fliped = copy.copy(new_array) # create copy of the encoded array 
    new_array_fliped[index] = 1-new_array_fliped[index] # apply flip  
    changed_tensor = tf.convert_to_tensor(tf.expand_dims(new_array_fliped,0)) #create new tensor 
    new_tensor = decoder(changed_tensor) # decode the sample with the change from the latent spaece
    output_array = new_tensor.numpy()[-1] # extraxt simple 1D array from tensor 
    output_array_binary = np.where(new_tensor.numpy()[-1]>0.5, 1, 0) # binarize decoded tensor around 0.5
    new_fitness = hiff_fitness(output_array_binary) # calculate transformed tensor fitness
    output_tensor = tf.convert_to_tensor(output_array_binary.reshape((1,N)), dtype = tf.float32) # save output tensor
    if debuge_variation:
        print("Input fitness: ", hiff_fitness(array), ", Decoded fitness: ", hiff_fitness(output_array_binary))
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
    for i in range(10 *N):
        output_tensor, output_array, new_fitness = code_flip_decode(array, encoder, decoder)
        if new_fitness >= current_fitness: # compare flip with current  fitness 
            current_fitness = new_fitness
            array = output_array
    return array

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
        new_trainig_set.append(transfer_sample_latent_flip(array, encoder, decoder))
    return np.asarray(new_trainig_set, dtype = np.float32)

def plot_evolution_model(encoder, decoder, array, plot_name, learning_steps = 50):
    """
    Still to improve
    Generate and save evolution plot of the model. 

    Parameters: 
        encoder - encoder reducing dimensionality
        decoder - decoder retrieving values from the latent space 
        array - set of sample from which one will be choosen and evaluated
        plot_name - name of the saving plot 
    
    Optional parameters: 
        learning_steps - number of steps of sample evaluation 
    """
    N = np.shape(array)[-1] # size of the array 
    index = np.random.randint(N) #choose random index to flip 
    candidate_solution = array[index]# pick up random sample 
    sol_evol = [] # list to store steps of evolution 
    sol_evol.append(candidate_solution)
    current_fittnes = hiff_fitness(candidate_solution)
    for i in range(learning_steps-1):
        new_candidate_sol = copy.copy(candidate_solution)
        output_tensor, output_array, new_fitness = code_flip_decode(new_candidate_sol, encoder, decoder)
        if new_fitness >= current_fittnes:
            candidate_solution = output_array 
            current_fittnes = new_fitness
        sol_evol.append(candidate_solution)

    tmp = np.array(sol_evol)
    plt.figure()
    plt.imshow(tmp, interpolation='nearest', cmap=cm.Greys_r)
    plt.title("Solution Development at Evolution Step 1")
    plt.xlabel("Solution variable")
    plt.ylabel("Development Step")
    plt.colorbar()
    path = create_plot_path(plot_name)
    plt.savefig(path)
    

def plot_trajectory_evolution(encoder, decoder, array, plot_name, 
                                target_size = 10, learning_steps = 30):  
    """
    Still to improve
    Generate and save trajectory plot of the model. 

    Parameters: 
        encoder - encoder reducing dimensionality
        decoder - decoder retrieving values from the latent space 
        array - set of sample from which  will be choosen and evaluated
        plot_name - name of the saving plot 

    Optional parameters: 
        target_size - number of tracking samples - default 10
        learning_steps - number of steps in evaluation - default 30
        
    """  
    X = np.arange(learning_steps)
    normalization_factor = hiff_fitness(np.ones((np.shape(array)[-1],)))
    trajectory_samples = []
    plt.figure()
    plt.title("Example Solution Trajectory 2 at Evolution Step 1")
    for k in range(target_size):
        current_array = array[k]
        current_fitness = hiff_fitness(current_array)
        current_target_trajectory = []
        current_target_trajectory.append(current_fitness/normalization_factor)
        for i in range(learning_steps-1):
            output_tensor, output_array, new_fitness = code_flip_decode(current_array, encoder, decoder)
            if new_fitness >= current_fitness:
                current_fitness = new_fitness
                current_array = output_array
                current_target_trajectory.append(new_fitness/normalization_factor)
            else:
                current_target_trajectory.append(current_target_trajectory[-1])
        trajectory_samples.append(current_target_trajectory)
        plt.plot(X, np.asarray(current_target_trajectory))
    plt.xlabel("learning step")
    plt.ylabel("fitness \ max_fitness")
    path = create_plot_path(plot_name)
    plt.savefig(path)
        


"""
Everything below is for improvement
"""


def generate_evol_plot(N=32, path = "solution_development_plot.png", learning_steps = 50):
    candidate_solution = np.random.randint(2, size = N)
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
    plt.imshow(tmp, interpolation='nearest', cmap=cm.Greys_r)
    plt.title("Solution Development at Evolution Step 1")
    plt.xlabel("Solution variable")
    plt.ylabel("Development Step")
    plt.colorbar()
    plt.savefig(path)

def generate_sol_plot(N=32, target_size = 10 ,path = "trajectory_plot.png", learning_steps = 70):    
    X = np.arange(learning_steps)
    normalization_factor = hiff_fitness(np.ones((N,)))
    plt.figure()
    plt.title("Example Solution Trajectory at Evolution Step 1")
    for k in range(target_size):
        candidate_solution = np.random.randint(2, size = N)
        solution_fitness = hiff_fitness(candidate_solution)
        current_target_trajectory = [solution_fitness/normalization_factor]
        for i in range(learning_steps-1):
            index = np.random.randint(N)
            new_candidate_sol = copy.copy(candidate_solution)
            new_candidate_sol[index] = 1 - new_candidate_sol[index] # apply variation 
            new_fitness = hiff_fitness(new_candidate_sol) # check the change 
            if new_fitness >= solution_fitness : 
                candidate_solution = new_candidate_sol
                solution_fitness = new_fitness
                current_target_trajectory.append(new_fitness/normalization_factor)
            else:
                current_target_trajectory.append(current_target_trajectory[-1])
        plt.plot(X, current_target_trajectory)
    plt.xlabel("learning step")
    plt.ylabel("fitness \ max_fitness")
    plt.savefig(path)
    
