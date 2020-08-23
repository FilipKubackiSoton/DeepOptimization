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
import collections
from mpl_toolkits.axes_grid1 import make_axes_locatable
import random 

knapSack = KnapSack("100_5_25_1")
utg = UtilsGeneral(knapSack)
utm = UtilsModel(utg)
utp = UtilsPlot(utilsGeneral = utg, utilsModel = utm)
ute = UtilsEncoded(utilsGeneral = utg, utilsModel = utm)
fitness_function = knapSack.Fitness
m = utg.restore_model_from_numpy("100_5_25_1Knapsack_Layer1\\100_5_25_1Knapsack")

tmptrain1, tmptrain = utg.load_datasets(1, 2)
set1 = utg.load_datasets(1)
model1, model2, model3, model4, model5, model6= utg.load_models(1,2, 3,4,5,6)

def sample_change(e, d, sample):
    """
    Execute change in encoded space tracking changes in decode 
    form and in fitness. 
    """
    def changed_index_pos(arr):
        """
        Find position of changed bits in decoded representation. 
        Minus sign means addition of elements and positive signe
        means substraction
        """
        lis = []
        for i in range(len(arr)):
            if arr[i]==2:
                lis.append(i)
            if arr[i]==-2:
                lis.append(-i)
        return lis

    pos = []
    fit = []    
    encode = utm.code(sample,e)
    decode_ref, decode_fit_ref = utm.decod(encode, d, len(encode), len(sample))
    latent_size = len(encode)
    sample_size = len(sample)
    dic_fit = {}
    dic_pos = {}
    for i in range(latent_size):
        encode_tmp = copy.copy(encode)
        encode_tmp[i] *=-1
        decode_tmp, decode_fit_tmp = utm.decod(encode_tmp, d, latent_size, sample_size)
        dic_fit[i] = decode_fit_tmp - decode_fit_ref
        dic_pos[i] = changed_index_pos(decode_tmp - decode_ref)
    return dic_pos, dic_fit

def encoded_actions(p):

    def is_swap(arr):
        # check if bit change is a swap
        pos = False
        neg = False
        for i in arr: 
            if i > 0:
                pos = True
            if i<0: 
                neg = True
            if pos and neg: 
                return True
        return False
            
    single_add = {}
    swap = {}
    group = {}
    for k,v in p.items(): # split bit changes into separate groups
        if len(v) == 0: 
            continue
        if len(v)==1:
            single_add[k] = v
            continue
        if is_swap(v):
            swap[k] = v
        else:
            group[k] = v
    return swap, single_add, group 


def get_matrix_representation(dic, visible_size, hidden_size):
    """
    Convert bit chages into matrix to visualize
    """
    res = []
    for i in range(hidden_size):
        tmp = np.zeros(visible_size)
        if dic.__contains__(i):
            for k in dic[i]:
                if k<0:
                    tmp[-k] = -1
                else:
                    tmp[k] = 1
        res.append(tmp)

    return np.asarray(res)


def show_actions_in_encoded_space(model, sample, show= True):
    """
    Visualize 
    """
    e,d = utm.split_model_into_encoder_decoder(model)
    visible_size = e.layers[0].input_shape[-1][-1]
    hidden_size = d.layers[0].input_shape[-1][-1]
    p, f = sample_change(e,d, sample)

    swap,single_add, group = encoded_actions(p)
    if show:
        fig, axes = plt.subplots(nrows=1, ncols=3, constrained_layout = True, figsize=(15,15))
        pc = axes[0].imshow(get_matrix_representation(swap, visible_size, hidden_size),interpolation='nearest',cmap=cm.Greys_r)
        axes[0].set_title("Swap")
        axes[0].set_xlabel("Visible change")
        axes[0].set_ylabel("Hidden change")

        axes[1].imshow(get_matrix_representation(single_add, visible_size, hidden_size),interpolation='nearest',cmap=cm.Greys_r)
        axes[1].set_title("Single addition")
        axes[1].set_xlabel("Visible change")
        axes[1].set_ylabel("Hidden change")

        axes[2].imshow(get_matrix_representation(group, visible_size, hidden_size),interpolation='nearest',cmap=cm.Greys_r)
        axes[2].set_title("'Grouping'")
        axes[2].set_xlabel("Visible change")
        axes[2].set_ylabel("Hidden change")
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(pc, cax=cax, ticks = [1,0,-1], label = "addition      ---      substraction")  
        plt.show()
    return p, f,swap ,single_add, group


def execute_encoded_changes(sample_set, size_set=500):
    swap = {}
    single_add = {}
    group = {}
    words = {}
    res = []
    for i in range(size_set):
        tmp = show_actions_in_encoded_space(m, sample_set[i], False)
        swap[i] = tmp[2]
        single_add[i] = tmp[3]
        group[i] = tmp[4]
        res.append(get_matrix_representation(tmp[2], 100,80))
        words[i] = tmptrain[i]
    return swap, single_add, group, words, res
swap, single_add, group, words, res = execute_encoded_changes(tmptrain, 2000)

def hidden_vissible_genom_samples_map(action_dic):
    res = {}
    for sn, hid in action_dic.items():
        for hid_change, vis_changes in hid.items():
            if not res.__contains__(hid_change):
                res[hid_change] = {}
            for vis_change in vis_changes:
                if not res[hid_change].__contains__(abs(vis_change)):
                    res[hid_change][abs(vis_change)] = []
                res[hid_change][abs(vis_change)].append(sn)
    return res
res = hidden_vissible_genom_samples_map(single_add)



def extract_genome(sample_set, args, show = False):

    def AllPositivesToZero(sol):
        ConvertSol = np.copy(sol)
        return  np.where(sol > 0.0, 1, 0)
        
    def AllNegativesToZero(sol):
        ConvertSol = np.copy(sol)
        return np.where(sol > 0.0, 0, 1)    
    res_pos = []
    res_neg = []
    sample_size = len(sample_set[0])
    final_pos = np.zeros(sample_size)
    final_neg = np.zeros(sample_size)
    for i in args:
        res_pos.append(AllNegativesToZero(sample_set[i]))
        res_neg.append(AllPositivesToZero(sample_set[i]))
    final_pos = np.mean(np.asarray(res_pos), axis = 0)
    final_neg = np.mean(np.asarray(res_neg), axis = 0)
    mean = np.mean(final_pos)
    
    if show:
        plt.figure()
        base = plt.gca().transData
        rot = transforms.Affine2D().rotate_deg(90)
        plt.bar(np.arange(len(final_pos)),-final_pos, color='blue', alpha = 0.7, label = "-p",transform = rot + base)
        plt.bar(range(len(final_neg)), -final_neg, bottom = -np.array(final_pos), color = "red", alpha = 0.7, label = "+p", transform = rot + base)
        plt.axvline(x = mean, color = "black", linestyle = '--')
        plt.title("Extracted genom")
        plt.xlabel("Probability")
        plt.ylabel("Bit positions")
        plt.legend()
        plt.show()
    return final_pos, final_neg

def get_genome_from_distribution(distribution, coef=2,):
    std = np.std(distribution)
    mean = np.mean(distribution)
    if mean + coef*std >=1.0: 
        tmp = 0.98
    else:
        tmp = mean + coef*std
    dist = np.where(distribution > tmp, -1, np.where(distribution < mean  - coef*std, 1, 0))
    ratio_neg = len(distribution) * mean /np.count_nonzero(dist == -1)
    ratio_pos = len(distribution) * (1-mean) /np.count_nonzero(dist == 1)
    mean = ratio_neg / (ratio_pos + ratio_neg)
    return dist



def check_genom(genom, hidden, visible,len_coef=5, number_of_samples=100, show = False):
    counter = 0
    num = 0 
    while num < number_of_samples:
        arr = get_sol_based_on_genom(genom, len_coef)
        num +=1
        dic = show_actions_in_encoded_space(m, arr, False)[3]
        if dic.__contains__(hidden) and abs(dic[hidden][0]) == visible:
            counter +=1
        if show and num % 10==0:
            print(num, "%")
        
    return counter/number_of_samples

def search_genom(solution, flip_list):
    index_one = random.choice(flip_list)
    index_two = random.choice(flip_list)
    solution[index_one] *=-1
    solution[index_two] *=-1
    return solution



def flip_and_update(current_solution, search, debuge_variation=False, *args):
    """
    Execute search function on the current_solution. 
    Update current_solution ifthe fitness value of 
    the modified solutin is greater, else does no change 
    
    Parameters:
        current_solution - container holding solution we modify 
        search - function searching and modyfing encoded representation 
    Optionals: 
        debug_variation (False) - turning on/off debug mode 
    Return: 
        conteiner, with the same type as current_solution
    """
    size = len(current_solution) # get the length of the solutin 
    rand_index = np.random.randint(size) # pick up a random index 
    new_solution = np.copy(current_solution) # copy current_solution 
    search(new_solution, *args) # execute search function on the copied solutino 
    new_fitness = fitness_function(new_solution, ) # calculate fitness of the new solution
    if new_fitness >= fitness_function(current_solution): 
        current_solution = new_solution # update current solutin if fitness is better 
    if debuge_variation: 
        print("New: ", new_fitness)
    return current_solution


def get_sol_based_on_genom(genom, len_cof=10):
    sample_length = len(genom)
    genom_pos = []
    for i in range(sample_length): 
        if genom[i] != 0:
            genom_pos.append(i)
    flip_list = np.delete(np.arange(sample_length), genom_pos)
    current_solution = np.where(genom == 0, -1, genom)

    for i in range(len_cof * sample_length): 
        current_solution = flip_and_update(current_solution, search_genom,False, flip_list)

    return current_solution


def get_genom_dist(res, hidden_size, visible_size, len_coef = 5, number_of_samples = 100):
    result_acc = []
    result_genom = []
    result_len = []
    for h in range(hidden_size):
        res_row = []
        res_genom = []
        res_len = []
        print(h/hidden_size*100, "%")
        for v in range(visible_size):
            if res.__contains__(h) and res[h].__contains__(v):
                n,p = extract_genome(tmptrain, res[h][v], False)
                genom = get_genome_from_distribution(n, 2)
                res_len.append(len(res[h][v]))
                res_genom.append(genom)
                res_row.append(check_genom(genom, h, v, len_coef=len_coef, number_of_samples=100))
            else: 
                res_row.append(-1)
                res_len.append(0)
                res_genom.append(np.asarray([0]))

        result_acc.append(np.asarray(res_row))
        result_len.append(np.asarray(res_len))
        result_genom.append(np.asarray(res_genom))
    return np.asarray(result_acc), np.asarray(result_len), np.asarray(result_genom)


tmp = get_genom_dist(res, 80, 100,10,300)
np.save("acc.npy",tmp[0])
np.save("len.npy",tmp[1])
np.save("gen.npy",tmp[2])
