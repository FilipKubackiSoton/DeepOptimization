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

class UtilsGeneral:
    def __init__(self):
        self.datasets_directory_name = "saved_datasets"
        self.models_directory_name = "saved_models"
        self.plots_directory_name = "saved_plots"
        self.model_counter = 0
        self.saved_models = []
        self.dataset_counter = 0
        self.saved_datasets = []

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
