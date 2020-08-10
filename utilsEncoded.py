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
from pathlib import Path
import shutil
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten
from decimal import Decimal

class UtilsEncoded:
    def __init__(self, utilsGeneral, utilsModel):
        self.utg = utilsGeneral
        self.utm = utilsModel
        self.fitness_function = self.utg.fitness_function
        self.search = self.utg.flip

    def weights_matrix_sorting(self, model, activation=1, background_activation = -1, column_sort = True, unit_sort=True, index_sort = False, log_conversion = False):
        """
        Transofrm weight matrix for the sake of visibility. 

        Parameters: 
            model - tf's model from which weight matrix will be displayed 
        
        Optional Parameters:
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
            matrix  = extract_weight_matrix(decoder.layers[1].get_weights())
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
            res += np.where(self.utm.code(arr, encoder) > 0.0, 0, 1)/size
        
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


    def plot(self, ax, model, plot_configuration):

        def convert_plot_option(opt):
            dic = {"column_sort": False, "unit_sort":False, "index_sort": False, "log_conversion": False}
            for i in opt: 
                if i == 'c':
                    dic["column_sort"] = True
                if i == 'i':
                    dic["index_sort"] = True
                if i == "u":
                    dic["unit_sort"] = True
                if i == 'l':
                    dic["log_conversion"] = True
            return dic

        kwargs = convert_plot_option(plot_configuration)

        bit_activation, weights_matrix, *masks  = self.weights_matrix_sorting(model, **kwargs)
        
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])

        if plot_configuration.__contains__('n'):
            ax.imshow(weights_matrix, interpolation='nearest',cmap=cm.Greys)
            return

        if kwargs.__contains__('unit_sort') and not kwargs['unit_sort']:
            pa = ax.imshow(masks[0],interpolation='nearest',cmap=cm.Blues)
            pb = ax.imshow(masks[1],interpolation='nearest',cmap=cm.Reds)    

        ax.imshow(masks[0], interpolation='nearest',cmap=cm.Greys)

    def hyper_params_tuner(self, training_function, visualize_function, vis_config=['n', "cl"], loss_funciton=None, correlation_function = None, **kwargs):
        """
        Hyperparameters tuning and ploting different weight matrxi from models. 
        Up to now it can handel two parameters tuning. 

        Params: 
            training_funciton - function to train model return model, training_history 
            visualize_funciton - function to show (plot) different models features: plot(ax, model, **kwargs)
            vis_config (["n", "cl"]) - list of configurations to plot: 
                c - column sort | if no c - row sort
                i - index sort | if no i - not index sort 
                u - unit sort | if no u - split sort 
                l - logarythmic conversion | if no l - linear scale
                n - no modified matrix (execute not modification of the matrix - pure weight matrix) 
                Example: 
                "ciul" - column sort + index_sort + unit sort + logarythmic conversion 
                "l" - raw sort + no index sort + split sort + logarythmic conversion 
                "n" - raw weight matrix

            Important !!!!    
            **kwargs - list of arguments with names to execute training_function.
                Important!!! - we can pass many arguments to one parameter as it 
                would be examied in the hyper_parameters tunning. 

                Example:  trainingSet = tmptrain, epochs = 1, compression = 0.8, 
                        batch_size = 20, reg_cof = [[0.001,0.01],[0.0001,0.001],[0.00001,0.0001]], 
                        dropout = 0.2, lr = [0.001, 0.0001], validation_split = 0.05
                
                It will check and plot configuration of every reg_cof with every lr. 

                Plot is arranged in the following manner: 
                    The bigges parameter variety is stretched horizontally
                    The second bigges parameter variety is streched vertically. 
                Executing the **kwargs as in the example result in the gird plot with 
                reg_cof examined horizontally and lr examined vertically

        Params still to add: 
            loss_function - plot model's loss history 
            correltation_function - plot "correlation matrix" 

            
        """        

        def get_param_title(**kwargs):
            title = ""
            for k, v in kwargs.items():
                if k == "trainingSet":
                    continue
                title += str(k[0:3])+" "+str(v)+", "
            return title

        def scientific_notation(x):
            if type(x)== list:
                tmp = []
                for i in x: 
                    tmp.append('%.0E' % Decimal(i))
                return tmp
            return '%.0E' % Decimal(x)

        dic={}
        for key, value in kwargs.items():
                dic[key] = value
        var_params = {} 
        con_params = {}
        for k in sorted(dic, key = lambda k : 1 if(type(dic[k])==int or type(dic[k])==float) else len(dic[k])): 
            if(k =="trainingSet" or not type(dic[k]) == list):
                con_params[k]=dic[k]
            else: 
                var_params[k]= dic[k]
        size_arr = []
        for k, x in var_params.items():
            size_arr.append(len(x))
        
        for plot_option in vis_config:
            fig, axes = plt.subplots(nrows=size_arr[-2], ncols=size_arr[-1])
            fig.suptitle('Weights Matrices '+plot_option+' \n'+get_param_title(**con_params), fontsize=16)
            fig.subplots_adjust(hspace = .2, wspace = .2)
            pos = copy.copy(var_params)
            
            for minor_ax_key, minor_ax_value in pos.items(): 
                pos.pop(minor_ax_key)
                row = 0

                for minor_param in minor_ax_value: 
                    for main_ax_key, main_ax_value in pos.items():
                        col = 0
                        for main_param in main_ax_value: 
                            con_params[main_ax_key] =main_param
                            con_params[minor_ax_key] = minor_param
                            print(row, " - ", col, " - ", main_param, " - ", minor_param)
                            
                            model, model_history = training_function(**con_params)
                            
                            visualize_function(axes[row][col], model, plot_option)     
                            if col == 0: 
                                axes[row][col].set_ylabel(scientific_notation(minor_param))
                            axes[row][col].set_xlabel(scientific_notation(main_param))
                            axes[row][col].set_aspect('auto')

                            col +=1
                        con_params.pop(main_ax_key)
                    row +=1
                    con_params.pop(minor_ax_key)           
                break
            fig.text(0.5, 0.04, list(var_params.keys())[1], ha='center', fontsize = 16)
            fig.text(0.04, 0.5,  list(var_params.keys())[0], va='center', rotation='vertical', fontsize = 16)
            fig.savefig(self.utg.create_plot_path("Plot {} for {} while {}.png".format(plot_option, get_param_title(**var_params), get_param_title(**con_params))))
            print("saving ", get_param_title(**var_params), get_param_title(**con_params))
            