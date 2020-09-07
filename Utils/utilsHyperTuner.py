from Utils.utilsGeneral import UtilsGeneral
from Utils.utilsModel import UtilsModel
from Utils.utilsPlot import UtilsPlot
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
from decimal import Decimal

class UtilsHyperTuner:
    def __init__(self, utilsGeneral, utilsModel):
        self.utg = utilsGeneral
        self.utm = utilsModel
        self.fitness_function = self.utg.fitness_function
        self.search = self.utg.flip

    def train_model(self, trainingSet, epochs = 500, compression = 0.8,  batch_size = 10, reg_cof = (0.0001,0.001), dropout = 0.2, lr = 0.001, validation_split = 0.05, metrics = tf.keras.metrics.RootMeanSquaredError() ):
        modelTmp = shallowNet.build(
            input_shape=self.utg.knapSack.Size, 
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

        bit_activation, weights_matrix, *masks  = self.utm.weights_matrix_sorting(model, **kwargs)
        
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])

        if plot_configuration.__contains__('n'):
            ax.imshow(weights_matrix, interpolation='nearest',cmap=cm.Greys)
            return

        if kwargs.__contains__('unit_sort') and not kwargs['unit_sort']:
            pa = ax.imshow(masks[0],interpolation='nearest',cmap=cm.Blues)
            pb = ax.imshow(masks[1],interpolation='nearest',cmap=cm.Reds)    

        ax.imshow(masks[0], interpolation='nearest',cmap=cm.Greys)

    def hyper_params_tuner(self, training_function = None, visualize_function = None, vis_config=['n', "cl"], loss_funciton=None, correlation_function = None, show = True, **kwargs):
        """
        Hyperparameters tuning and ploting different weight matrxi from models. 
        Up to now it can handel two parameters tuning. 

        Parameters: 
            training_funciton - function to train/modify tf's model: return model, training_history 
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
                would be examied in the hyper_parameters tunning proccess. 

                Example:  trainingSet = tmptrain, epochs = 1, compression = 0.8, 
                        batch_size = 20, reg_cof = [[0.001,0.01],[0.0001,0.001],[0.00001,0.0001]], 
                        dropout = 0.2, lr = [0.001, 0.0001], validation_split = 0.05
                
                It will check and plot configuration of every reg_cof with every lr. 

                Plot is arranged in the following manner: 
                    The bigges parameter variety is stretched horizontally
                    The second bigges parameter variety is streched vertically. 
                Executing the **kwargs as in the example result in the gird plot with 
                reg_cof examined horizontally and lr examined vertically           
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

        if training_function == None: 
            training_function = self.train_model
        if visualize_function == None: 
            visualize_function = self.plot

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
            if show: 
                plt.show()