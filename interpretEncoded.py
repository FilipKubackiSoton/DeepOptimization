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
from numpy.ma import masked_array
from matplotlib import pyplot, transforms

knapSack = KnapSack("100_5_25_1")

set1 = utg.load_datasets(1)
model1, model2, model3, model4, model5, model6= utg.load_models(1,2,3,4,5,6)

class EncodedAnalyzer():
    def __init__(self, utilsGeneral, utilsModel, utilsPlot = None):
        self.utg = utilsGeneral
        self.utm = utilsModel
        self.utp = utilsPlot
        self.fitness_function = self.utg.fitness_function

    def code(self, array, encoder, input_size=None, latent_size=None, debuge_variation=False):
        if input_size == None:
            input_size = len(array) # if input_size is implicit do not waist time to calcule it
        if latent_size == None:
            latent_size = np.shape(encoder.layers[-1].get_weights()[0])[-1] # if latent_size is implicit do not waist time to calcule it
        encoded_solution = encoder(np.expand_dims(array, axis = 0)).numpy().flatten() # encode array 
        return encoded_solution

    def decod(self, encoded_solution, decoder, latent_size, output_size):
        new_tensor = decoder(encoded_solution.reshape(1,latent_size)) # decode changed solution 
        output_array_binary = np.where(new_tensor.numpy()[-1] > 0.0, 1, -1)  # binarize decoded tensor around 0.0
        new_fitness = self.fitness_function(output_array_binary) # calculate new fitness
        return output_array_binary, new_fitness

    def weights_matrix_sorting(self, model, activation=1, background_activation = -1, column_sort = True, unit_sort=True, index_sort = False, log_conversion = False):
        
        def value(x):
            if log_conversion:
                return round(math.log(x,10))
            else:
        
                return x

        def shuffle_columns_weights_matrix(decoder, pos): 
            matrix  = decoder.layers[1].get_weights()[1]
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
            
            
        encoder, decoder = self.utm.split_model_into_encoder_decoder(model)
        size = encoder.layers[0].input_shape[0][-1]
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


    def latent_activation(self, model, title, activation=1, background_activation =-1, column_sort = True, unit_sort=True, index_sort = False, log_conversion = True):
        bit_activation, weights_matrix, *masks  = self.weights_matrix_sorting(model, activation, background_activation,column_sort = column_sort, unit_sort = unit_sort, index_sort = index_sort, log_conversion = log_conversion)

        title = title + " - activation: " +str(activation)+" background: "+ str(background_activation) + "\n "
        if column_sort:
            title += "column sort, "
        else:
            title += "row sort, "
        if index_sort:
            title += "index_sort"
        else:
            title += "no index sort"
        if log_conversion:
            title += " log scale"
        else:
            title += " linear scale"
        
        if index_sort:
            fig, axes = plt.subplots(nrows=1, ncols=3, constrained_layout = True)
        else:
            fig, axes = plt.subplots(nrows=1, ncols=2, constrained_layout = True)
        fig.suptitle(title, fontsize=16)
        if unit_sort:
            pa = axes[0].imshow(masks[0],interpolation='nearest',cmap=cm.Greys_r)
            cba = fig.colorbar(pa, ax = axes[0], location = "left")
            cba.set_label('Magnitude of Absolute Activation')
            axes[0].set_title("Unit Sort")
            axes[0].set_xlabel("Hidden")
        else:
            pa = axes[0].imshow(masks[0],interpolation='nearest',cmap=cm.Blues)
            cba = fig.colorbar(pa, ax = axes[0], location = "left")
            pb = axes[0].imshow(masks[1],interpolation='nearest',cmap=cm.Reds)
            cbb = fig.colorbar(pb, ax = axes[0], location = "right")
            cba.set_label('Negative')
            cbb.set_label('Positive')
            axes[0].set_title("Split Sort")
            axes[0].set_xlabel("Hidden")
        axes[0].set_aspect('auto')

        
        axes[1].imshow(weights_matrix, interpolation='nearest', cmap=cm.Greys_r)
        axes[1].set_title("Weights")
        axes[1].set_ylabel("Visible")
        axes[1].set_xlabel("Hidden")
        axes[1].set_aspect('auto')
        
        if index_sort:
            axes[2].bar(np.arange(len(bit_activation)), bit_activation, align = "center", alpha = 0.5)
            axes[2].set_title("Bit Activation")
            axes[2].set_ylabel("Activation Probability")
            axes[2].set_xlabel("Bit index")
            axes[2].set_aspect('auto')

        plt.show()
        return





