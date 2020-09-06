from utilsGeneral import UtilsGeneral
from utilsModel import UtilsModel
from utilsPlot import UtilsPlot
from utilsEncoded import UtilsEncoded
from utilsGenome import UtilsGenome
from KnapSack.KnapSack import KnapSack
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
import collections
from mpl_toolkits.axes_grid1 import make_axes_locatable
import random 
import pickle
import re

knapSack = KnapSack("100_10_25_7")
utg = UtilsGeneral(knapSack)
utm = UtilsModel(utg)
utp = UtilsPlot(utilsGeneral = utg, utilsModel = utm)
ute = UtilsEncoded(utilsGeneral = utg, utilsModel = utm)

sample_set = np.load("100_10_25_7\TrainData\TrainingData_Layer_1_Evo1.npy")
model6 = utg.restore_model_from_directory("100_10_25_7\Model_CheckPoints", encoder_template = 'Train_BiasEncoder_L(.+)_EvoStep_6', weights_template = 'Train_Weights_L(.+)_TrainLevel_6' ,decoder_template = 'Train_BiasDecoder_L(.+)_EvoStep_6' ,show = False)
utgen6 = UtilsGenome(utg, utm, sample_set, model6)


################################ PLOT TRAJECTORY EVOLUTION ################################
#utp.plot_trajectory_evolution(100, learning_steps = 100, sample_number=10, model = model6)

################################ PLOT LATENT ACTIVATION ################################
#utp.plot_latent_acitvation(model = model6, sample_set = sample_set)

################################ PLOT MODEL EVOLUTION ################################
utp.plot_evolution_model(model = model6, sample_set = sample_set)



