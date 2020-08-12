from utilsGeneral import UtilsGeneral
from utilsModel import UtilsModel
from utilsPlot import UtilsPlot
from utilsEncoded import UtilsEncoded
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
from tqdm import tqdm


knapSack = KnapSack("100_5_25_1")
utg = UtilsGeneral(knapSack)
utm = UtilsModel(utg)
utp = UtilsPlot(utilsGeneral = utg, utilsModel = utm)
ute = UtilsEncoded(utilsGeneral = utg, utilsModel = utm)
fitness_function = knapSack.Fitness
m = utg.restore_model_from_numpy("100_5_25_1Knapsack_Layer1\\100_5_25_1Knapsack")


tmptrain, tmptrain2 = utg.load_datasets(1, 2)

set1 = utg.load_datasets(1)
model1, model2, model3, model4, model5, model6= utg.load_models(1,2, 3,4,5,6)

def cor_matrix(pos):
    l = len(pos)
    mat = np.zeros((l, 80, 100))
    for z in range(l):
        sample = pos[z]
        for x, v in sample.items():
            for y in v:
                mat[z][x][y] = 1
    return mat

def sample_change_distribution(e, d, sample, checks_per_sample):
    def non_zero_index_pos(arr):
        lis = []
        for i in range(len(arr)):
            if arr[i]!=0:
                lis.append(i)
        return lis
    pos = []
    fit = []
    for c in range(checks_per_sample):
        
        tmp = copy.copy(sample)
        tmp[c] *=-1
        encode = utm.code(tmp,e)
        decode_ref, decode_fit_ref = utm.decod(encode, d, len(encode), len(tmp))
        latent_size = len(encode)
        sample_size = len(tmp)
        dic_fit = {}
        dic_pos = {}
        for i in range(latent_size):
            encode_tmp = copy.copy(encode)
            encode_tmp[i] *=-1
            decode_tmp, decode_fit_tmp = utm.decod(encode_tmp, d, latent_size, sample_size)
            dic_fit[i] = decode_fit_tmp - decode_fit_ref
            dic_pos[i] = non_zero_index_pos(decode_tmp - tmp)
        pos.append(dic_pos)
        fit.append(dic_fit)
    return pos, fit


def foo(model, training_set, sample_numbers, checks_per_sample):

    mat=[]
    e,d = utm.split_model_into_encoder_decoder(model)

    for s in tqdm(range(sample_numbers)): 
        print(s) 
        pos, fit = sample_change_distribution(e,d, training_set[s], checks_per_sample)
        mat.append(cor_matrix(pos).mean(axis = 0))
    return np.asarray(mat), fit

mat, fit = foo(m, tmptrain2, 10, 10)

plt.figure()
plt.imshow(mat.mean(axis = 0),cmap=cm.Greys, interpolation='nearest')
plt.xlabel("visible")
plt.ylabel("hidden")
plt.title("Bit chaning map - averaged samples")
plt.colorbar()
plt.savefig("vishid.png")


plt.figure()
plt.imshow(mat.mean(axis = 1), cmap=cm.Greys, interpolation='nearest')
plt.xlabel("visible")
plt.ylabel("sample")
plt.title("Visible bits activation - averaged hidden")
plt.colorbar()
plt.savefig("vissam.png")


plt.figure()
plt.imshow(mat.mean(axis = 2), cmap=cm.Greys, interpolation='nearest')
plt.xlabel("hidden")
plt.ylabel("sample")
plt.title("Hidden bits activation - averaged visible")
plt.colorbar()
plt.savefig("hidsam.png")
plt.show()


