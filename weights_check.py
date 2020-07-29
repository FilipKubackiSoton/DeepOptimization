from utilsGeneral import UtilsGeneral
from utilsModel import UtilsModel
from utilsPlot import UtilsPlot
from KnapSack import KnapSack
from shallowNet.shallowNet import shallowNet, DenseTranspose
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib import transforms
import math
import scipy.stats as stats
import collections
import matplotlib.cm as cm
import tensorflow as tf
import matplotlib.gridspec as gridspec

knapSack = KnapSack("100_5_25_1")
utg = UtilsGeneral(knapSack)
utm = UtilsModel(utg)
utp = UtilsPlot(utilsGeneral = utg, utilsModel = utm)
fitness_function = knapSack.Fitness

tmptrain = utg.generate_training_set(100, 10)
utg.save(tmptrain)

def code(array, encoder, input_size=None, latent_size=None, debuge_variation=False):
    if input_size == None:
        input_size = len(array) # if input_size is implicit do not waist time to calcule it
    if latent_size == None:
        latent_size = np.shape(encoder.layers[-1].get_weights()[0])[-1] # if latent_size is implicit do not waist time to calcule it
    encoded_solution = encoder(np.expand_dims(array, axis = 0)).numpy().flatten() # encode array 
    return encoded_solution

def decod(encoded_solution, decoder, latent_size, output_size):
    new_tensor = decoder(encoded_solution.reshape(1,latent_size)) # decode changed solution 
    output_array_binary = np.where(new_tensor.numpy()[-1] > 0.0, 1, -1)  # binarize decoded tensor around 0.0
    new_fitness = fitness_function(output_array_binary) # calculate new fitness
    return output_array_binary, new_fitness

def weights_matrix_sorting(model, activation=1, background_activation = -1):
    def shuffle_columns_weights_matrix(decoder, pos):
        #matrix  = decoder.layers[1].get_weights()[1]
        matrix  = decoder.layers[1].get_weights()
        mat = np.full_like(matrix, 0)
        index = 0
        for i in pos:
            mat[:, index] = matrix[:, i]
            index +=1
        return mat

    encoder, decoder = utm.split_model_into_encoder_decoder(model)
    size = encoder.layers[0].input_shape[0][-1]
    latent_size = np.shape(encoder.layers[-1].get_weights()[0])[-1]
    res = np.zeros(latent_size)
    glob_pos = {}
    for i in range(size):
        arr = np.zeros(size) + background_activation
        arr[i] = activation
        res += np.where(code(arr, encoder) < 0.0, 0, 1)/size
    
    for i in range(latent_size):
        glob_pos[i] = res[i]

    pos = {k : v for k, v in sorted(glob_pos.items(), key = lambda item : item[1])}
    res.sort() # sort bits accorg
    
    weights_matrix = shuffle_columns_weights_matrix(decoder, pos)
    sorted_weights_matrix = copy.copy(weights_matrix)
   
    for i in range(np.shape(sorted_weights_matrix)[0]):
        sorted_weights_matrix[i] = np.sort(sorted_weights_matrix[i])

    return res, weights_matrix, sorted_weights_matrix
    

#e, d = utm.split_model_into_encoder_decoder(model3)
def latent_activation(model, title, xl, yl, activation=1, background_activation =-1):
    bit_activation, weights_matrix, sorted_weights_matrix = weights_matrix_sorting(model, activation, background_activation)

    title = title + " - activation: " +str(activation)+" background: "+ str(background_activation)

    fig, axes = plt.subplots(nrows=1, ncols=3, constrained_layout = True)
    fig.suptitle(title, fontsize=16)
    axes[0].bar(np.arange(len(bit_activation)), bit_activation, align = "center", alpha = 0.5)
    axes[0].set_title("Bit Activation")
    axes[0].set_ylabel(yl)
    axes[0].set_xlabel(xl)
    axes[1].imshow(weights_matrix, interpolation='nearest', cmap=cm.Greys_r)
    axes[1].set_title("Weights\n Matrix")
    axes[1].set_ylabel("Visible")
    axes[1].set_xlabel("Hidden")
    im =axes[2].imshow(sorted_weights_matrix, interpolation='nearest', cmap=cm.Greys_r)
    axes[2].set_title("Weights Matrix \nSorted")
    axes[2].set_ylabel("Visible")
    axes[2].set_xlabel("Hidden")
    return
    """
_ = latent_activation(model3, "Bit Activation: model3", "Bit index", "Probability of Activation")
pos = latent_activation(model3, "Bit Activation: model3", "Bit index", "Probability of Activation", background_activation=-1)
"""
#tmp = latent_activation(modelTmp, "Bit Activation: modelTmp", "Bit index", "Probability of Activation", background_activation=-1)

compression = 0.8
epochs = 500
batch_size = 20
reg_cof = (0.0001,0.001)
dropout = 0.2
lr = 0.001

def train_model(trainingSet, epochs = 500, compression = 0.8,  batch_size = 20, reg_cof = (0.001,0.01), dropout = 0.2, lr = 0.001, validation_split = 0.05, metrics = tf.keras.metrics.RootMeanSquaredError() ):
    modelTmp = shallowNet.build(
        input_shape=knapSack.Size, 
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

def test_model_reg_cof(trainingSet, order_variation, epochs = 500, compression = 0.8, batch_size = 20, reg_cof = [0.01, 0.1], dropout = 0.2, lr = 0.001, validation_split = 0.05, metrics = tf.keras.metrics.RootMeanSquaredError()):
    param_name = "com:"+str(compression)+" dro: "+str(dropout)+ "lr: "+str(round(math.log(lr)))
    fig_sorted, axes_sorted = plt.subplots(nrows=order_variation, ncols=order_variation)
    fig_sorted.suptitle('Weights Matrices Sorted'+param_name, fontsize=16)
    fig_sorted.tight_layout(pad=1.0)
    fig_sorted.subplots_adjust(wspace=None, hspace=None)
    fig_index, axes_index = plt.subplots(nrows=order_variation, ncols=order_variation, constrained_layout = False)
    fig_index.suptitle("Index Activation"+param_name, fontsize = 16)
    
    for i in range(order_variation):

        for j in range(order_variation):
            model, model_history = train_model(
                trainingSet, epochs, 
                compression, batch_size, [reg_cof[0] / 10**i, reg_cof[1] / 10**j], 
                dropout, lr, validation_split, metrics)

            bit_activation, weights_matrix, sorted_weights_matrix = weights_matrix_sorting(model)
            axes_sorted[i][j].imshow(sorted_weights_matrix)
            axes_index[i][j].bar(np.arange(len(bit_activation)), bit_activation)
            axes_sorted[i][j].xaxis.set_ticks([])
            if j == i:
                axes_sorted[j][0].set_ylabel("L2: {}".format(round(math.log(reg_cof[1] / 10**j,10))),fontsize=10)
                axes_index[j][0].set_ylabel("L2: {}".format(round(math.log(reg_cof[1] / 10**j,10))),fontsize=10)
            

                 
        axes_sorted[order_variation-1][i].set_xlabel("L1: {}".format(round(math.log(reg_cof[0] / 10**i,10)-1)),fontsize=10) 
        axes_index[order_variation-1][i].set_xlabel("L1: {}".format(round(math.log(reg_cof[0] / 10**i,10)-1)),fontsize=10) 

        #plt.setp(axes_sorted[:,i], ylabel = "L1 {}".format(int(math.log(reg_cof[0] / 10**i,10)-1)))
    
    fig_sorted.savefig(utg.create_plot_path("Weights Sorted2 "+param_name+" .png"))
    fig_index.savefig(utg.create_plot_path("Index Sorted2"+param_name+".png"))
    plt.show()

test_model_reg_cof(tmptrain, 3,epochs=1)