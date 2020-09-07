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
import pickle


class UtilsGenome:
    def __init__(self, utg, utm, training_set = None, model= None):
        """
        It's example how to check genome for swap action class: 

            swap = utg.load_obj("swap") # load swap action dictionary  !!! or !!! get_map_of_actions_based_on_samples(training_set, number_of_checked_samples)

            genomeUtils = Genome(utg, utm, tmptrain, m) # initialize class

            resswap = genomeUtils.get_map_hidden_visible_samples(swap) # transform swap dictionary 

            a,l,g = genomeUtils.get_genom_performence_distribution(resswap, 0, std_coef=2, sample_length_filter=2) # check genoms for swap actions

            utg.save_obj(a, "accswap") # save genomes' accuracy 
            utg.save_obj(a, "lenswap") # save genomes' lenght
            utg.save_obj(a, "genswap") # save genomes
        """
    
        self.model = model
        self.utg = utg
        self.utm = utm
        self.fitness_function = self.utg.knapSack.Fitness
        self.training_set = training_set

    def sample_change(self, e, d, sample):
        """
        Execute change in encoded space tracking changes in decode 
        form and in fitness. 
        Parmameters: 
            e - tf's encoder 
            d - tf's decoder 
            sample - knap Sack solution
        Returns
            python dictionary {hidden change : visible change} ( positive index value - exclusion, negative index value - inclusion),
            python dictionary {hidden change : fitness change}
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
        encode = self.utm.code(sample,e)
        decode_ref, decode_fit_ref = self.utm.decod(encode, d)
        latent_size = len(encode)
        sample_size = len(sample)
        dic_fit = {}
        dic_pos = {}
        for i in range(latent_size):
            encode_tmp = copy.copy(encode)
            encode_tmp[i] *=-1
            decode_tmp, decode_fit_tmp = self.utm.decod(encode_tmp, d)
            dic_fit[i] = decode_fit_tmp - decode_fit_ref
            dic_pos[i] = changed_index_pos(decode_tmp - decode_ref)
        return dic_pos, dic_fit

    def classify_actions(self, p):
        """
        Clasifie actions into distinctive categories: (swap, single_add, group)
        Parameters: 
            p - python dictionary {hidden change : visible change} ( positive index value - exclusion, negative index value - inclusion)
        Returns: 
            swap action dinctionary {hidden change : visible change},
            single_add action dinctionary {hidden change : visible change},
            group action dinctionary {hidden change : visible change}
        """

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

    def get_matrix_representation(self, dic, visible_size, hidden_size):
        """
        Convert action dictionary into matrix from to visualize
        Parameters: 
            dic - action dictionary {hidden change : visible change}
            visible_size - int 
            hidden_size - int
        Returns: 
            np matrix of change
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
    
    def get_actions_in_encoded_space(self, sample, model = None, show= False, index_to_split = None, title = "-"):
        """
        Extract actions from hidden to visible space (what hidden bit change makes vissible bits change). 
        Paramters: 
            model - tf model 
            sample - knap sack solution which changes will be examined
            show (True) - show changes from this sample and model
        Return: 
            dictionary {hidden change : visible change},
            dictionary {hidden change : fintess change}, 
            dictionary actions belonging to swap {hidden_change : visible change}, 
            dictionary actions belonging to single add {hidden_change : visible change}, 
            dictionary actions belonging to group {hidden_change : visible change}, 
        """
        if model == None: 
            model = self.model

        e,d = self.utm.split_model_into_encoder_decoder(
            model = model, 
            index_to_split = index_to_split
            )
        
        visible_size = e.layers[0].input_shape[-1]#[-1]
        hidden_size = d.layers[0].input_shape[-1]#[-1]

        p, f = self.sample_change(e,d, sample)

        swap,single_add, group = self.classify_actions(p)
        if show:
            
            fig, axes = plt.subplots(nrows=1, ncols=3, constrained_layout = False, figsize=(15,15))  
            pc = axes[0].imshow(self.get_matrix_representation(swap, visible_size, hidden_size),interpolation='nearest',cmap=cm.Greys_r)
            axes[0].set_title("Swap")
            axes[0].set_xlabel("Visible change")
            axes[0].set_ylabel("Hidden change")

            axes[1].imshow(self.get_matrix_representation(single_add, visible_size, hidden_size),interpolation='nearest',cmap=cm.Greys_r)
            axes[1].set_title("Single addition")
            axes[1].set_xlabel("Visible change")
            axes[1].set_ylabel("Hidden change")

            axes[2].imshow(self.get_matrix_representation(group, visible_size, hidden_size),interpolation='nearest',cmap=cm.Greys_r)
            axes[2].set_title("'Grouping'")
            axes[2].set_xlabel("Visible change")
            axes[2].set_ylabel("Hidden change")
            divider = make_axes_locatable(plt.gca())
            cax = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(pc, cax=cax, ticks = [1,0,-1], label = "addition      - {} -      substraction".format(title))  
            plt.show()
        return p, f,swap ,single_add, group


    def get_map_of_actions_based_on_samples(self, model = None, sample_set=None, size_set=500, index_to_split = None):
        """
        Get map of: sample index --> action in action's class (swap, single_add, group)
        Parameters: 
            sample_set - training set 
            size_set (500) - number of considered samples 
        Return: 
            swap {sample index : {hiden bit change : visible bit change}}
            single_add {sample index : {hiden bit change : visible bit change}}
            group {sample index : {hiden bit change : visible bit change}}
        """
        if sample_set.all() == None:
            sample_set = self.training_set
        if model == None: 
            model = self.model
        swap = {}
        single_add = {}
        group = {}
        words = {}
        res = []
        for i in range(size_set):
            tmp = self.get_actions_in_encoded_space(
                sample = sample_set[i],
                model =model,
                index_to_split= index_to_split, 
                show = False)
            swap[i] = tmp[2]
            single_add[i] = tmp[3]
            group[i] = tmp[4]
            res.append(self.get_matrix_representation(tmp[2], 100,80))
            words[i] = self.training_set[i]
        return swap, single_add, group

    def hash_function(self, array):
        # return hash code
        return str(array)

    def check_hash_function(self, array):
        # return alternative hash code
        return str(list(np.asarray(array)*(-1)))

    def dehash(self, hash_code):
        # dehash hash code
        return eval(hash_code)

    def get_map_hidden_visible_samples(self, action_dic, hash_f= None, check_hash_f= None):
        """
        Get map of changes to samples where such changes occure.
        Parameters: 
            action_dic - python dictionary {sample_index: {hidden bit change : visible bit change}}
            hash_f - hashing function (as list are not hashable)
            hash_f - check_hash_f (hashing function which check potential alternative form)
        Returns: 
            {hidden bit change: {visible bit change : samples where such change occure}}
        """
        if hash_f == None: 
            hash_f = self.hash_function
        if check_hash_f == None: 
            check_hash_f = self.check_hash_function

        res = {}
        for sn, hid in action_dic.items():
            for hid_change, vis_changes in hid.items():

                if not res.__contains__(hid_change):
                    res[hid_change] = {}

                if not res[hid_change].__contains__(hash_f(vis_changes)) and  not res[hid_change].__contains__(check_hash_f(vis_changes)):
                        res[hid_change][hash_f(vis_changes)] = []

                if res[hid_change].__contains__(hash_f(vis_changes)):   
                    res[hid_change][hash_f(vis_changes)].append(sn)
                if res[hid_change].__contains__(check_hash_f(vis_changes)):   
                    res[hid_change][check_hash_f(vis_changes)].append(sn)

        return res

    def get_bits_probability_distribution(self, sample_set, args, show = False):
        """
        Calculate bits distribution based on given samples. 
        Parameters: 
            sample_set - traning set 
            args - list of indeces referring to the sample in sample_set
        Return: 
            np array of distribution of negative bits
        Optional: 
            show (False) - show graphicaly distributon
        """

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
        std = np.std(final_pos)
        
        if show:
            plt.figure()
            base = plt.gca().transData
            rot = transforms.Affine2D().rotate_deg(90)
            plt.bar(np.arange(len(final_pos)),-final_pos, color='blue', alpha = 0.7, label = "-p",transform = rot + base)
            plt.bar(range(len(final_neg)), -final_neg, bottom = -np.array(final_pos), color = "red", alpha = 0.7, label = "+p", transform = rot + base)
            plt.axvline(x = mean, color = "black", linestyle = '-')
            plt.axvline(x = mean - 2* std , color = "black", linestyle = '--')
            plt.axvline(x = 0.98 , color = "black", linestyle = '--')

            plt.title("Bits distribution")
            plt.xlabel("Probability")
            plt.ylabel("Bit positions")
            plt.legend()
            plt.show()
        return final_pos

    def get_genome_from_distribution(self, distribution, coef=2,upper_limit = .98):
        """
        Extract genome from "negative" distribution. 
        Parameters: 
            distribution - distribution of negative values in solutions 
        Optional:
            coef (2) - coefficient standing next to std extracted from distribution
        Return 
            genome - np.array with -1 and 1 if they fall out of 2 * std and 0 else
        """
        std = np.std(distribution)
        mean = np.mean(distribution)
        if mean + coef*std >=1.0: 
            tmp = upper_limit
        else:
            tmp = mean + coef*std
        dist = np.where(distribution > tmp, -1, np.where(distribution < mean  - coef*std, 1, 0))
        return dist



    def check_genom(self, genom, hidden, visible, action, dehash = None, check_hash_f = None,len_coef=5, number_of_samples=100, show = False, model = None, index_to_split = None):
        """
        Check genome performence. 
        Paramaeters: 
            genom - np.array of genome (array of -1's, 0's, and 1's)
            hidden - indicator of hidden change specific for genome 
            visible - indicator of visible change specific for genome 
            action - choosing class of actino (0 - swap, 1 - single add, 2 - group)
            dehash - dehasihng function 
            check_hash_f - hashing function checking alternative hash valu e
            len_coef (5) - coefficient defines number of steps in solution generation (num_steps = len(sol) * len_coef)
            number_of_samples - number of samples based on which genom performence is calculated 
            show (False) - debuge variable
        Returns:
            performance of genome (ratio of samples with expected action to number_of_samples)
        """
        if model == None: 
            m = self.model
        else: 
            m = model
        if dehash == None:
            dehash = self.dehash
        if check_hash_f == None: 
            check_hash_f = self.check_hash_function

        counter = 0
        num = 0 
        const = int(number_of_samples/10)
        while num < number_of_samples:
            arr = self.get_sol_based_on_genom(genom, len_coef, False)
            num +=1
            dic = self.get_actions_in_encoded_space(
                sample = arr, 
                model = m, 
                show = False,
                index_to_split = index_to_split)[2+action]
            if dic.__contains__(hidden) and (dic[hidden]==dehash(visible) or dic[hidden] == dehash(self.check_hash_function(dehash(visible)))):
                counter+=1
            if show and num % const==0:
                print(num/number_of_samples * 100, "%")

        print(counter/number_of_samples)

        return counter/number_of_samples

    def get_sol_based_on_genom(self, genom, len_cof=5, show = False):
        """
        Gradually improve solution based on given genome:
        Parameters: 
            genom - np.array of genome (array of -1's, 0's, and 1's)
            len_coef (5) - coefficient defines number of steps in solution generation (num_steps = len(sol) * len_coef)
            show (False) - debuge variable
        Returns: 
            improved solution based on genome
        """
        sample_length = len(genom)
        genom_pos = []
        for i in range(sample_length): 
            if genom[i] != 0:
                genom_pos.append(i)
        flip_list = np.delete(np.arange(sample_length), genom_pos)
        current_solution = np.where(genom == 0, -1, genom)

        for i in range(len_cof * sample_length): 
            current_solution = self.flip_and_update(current_solution, self.search_genom,False, flip_list)
        if show: 
            print(self.fitness_function(current_solution))

        return current_solution

    def flip_and_update(self, current_solution, search, debuge_variation=False, *args):
        """
        !!! It can be changed in utils general: still do compose into

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
        new_fitness = self.fitness_function(new_solution, ) # calculate fitness of the new solution
        if new_fitness >= self.fitness_function(current_solution): 
            current_solution = new_solution # update current solutin if fitness is better 
        if debuge_variation: 
            print("New: ", new_fitness)
        return current_solution

    def search_genom(self, solution, flip_list):
        """
        Perform random bit pair flips based on the list of potential indexes  
        Parameters: 
            solution - solution to knap Sack Problem 
            flip_list - list of indexes from solution which are not included in a genome
        Return 
            modyfied solution
        """
        index_one = random.choice(flip_list)
        index_two = random.choice(flip_list)
        solution[index_one] *=-1
        solution[index_two] *=-1
        return solution

    def get_genom_performence_distribution(self, res, action, dehash_function = None, change_hash_function = None, len_coef = 5, number_of_samples = 200, sample_length_filter=10, std_coef = 2):
        """
        Return performence of genomes extracted from training set. 

        Parameters: 
            res -  python dictionary {hidden bit change: {visible bit change : samples where such change occure}} (e.g. for swap or group)
            action - integer from [0,1,2] (0 - swap, 1 - single add, 2 - group)
            dehash_function - dehashing funciton 
            change_hash_function - hash function checking alternative hahs value
            sample_length_filter (10) - minimal number of samples which must share particular action 
            std_coef (2) - coeficient based on which genom is extracted (bits from bits distribution are in genome if: mean - std_coef*std > bit_p || mean + std_coef*std < bit_p)
            len_coef (5) - coefficient defines number of steps in solution generation (num_steps = len(sol) * len_coef)
            number_of_samples (200) - number of samples based on which genom performence is calculated 
            show (False) - debuge variable
        Returns: 
            accuracy dictionary {hidden change : {visible change : accuracy of extracted genome from such action}},
            length dictionary {hidden change : {visible change :number of samples sharing action}},
            genome dictionary {hidden change : {visible change : extracted genome for such action}}
        """
        if dehash_function == None: 
            dehash_function = self.dehash
        if change_hash_function == None: 
            change_hash_function = self.check_hash_function

        result_acc = {}
        result_genom = {}
        result_len = {}
        progress_index = 0
        hid_len = len(res)
        for hid_change, visible_change in res.items():

            if not result_acc.__contains__(hid_change):
                result_acc[hid_change] = {}
            if not result_genom.__contains__(hid_change):
                result_genom[hid_change] = {}
            if not result_len.__contains__(hid_change):
                result_len[hid_change] = {}

            print(progress_index/ hid_len * 100, "%")
            progress_index +=1

            for vis_change, sample in visible_change.items():
                print(hid_change, " ", vis_change, " ", len(sample))

                if len(sample)>sample_length_filter:
                    
                    n= self.get_bits_probability_distribution(self.training_set, res[hid_change][vis_change], False)
                    genom = self.get_genome_from_distribution(n, std_coef)
                    result_len[hid_change][vis_change]= len(sample)
                    result_genom[hid_change][vis_change] = genom
                    result_acc[hid_change][vis_change] = self.check_genom(genom, hid_change, vis_change, action, self.dehash, self.check_hash_function, len_coef=len_coef, number_of_samples=number_of_samples)
                
        return result_acc, result_len, result_genom


"""
knapSack = KnapSack("100_5_25_1")
utg = UtilsGeneral(knapSack)
utm = UtilsModel(utg)
utp = UtilsPlot(utilsGeneral = utg, utilsModel = utm)
ute = UtilsEncoded(utilsGeneral = utg, utilsModel = utm)
fitness_function = knapSack.Fitness

tmptrain1, tmptrain = utg.load_datasets(1, 2)

model = utg.restore_model_from_directory("100_10_25_7\Model_CheckPoints",encoder_template = 'Train_BiasEncoder_L(.+)_EvoStep_6', weights_template = 'Train_Weights_L(.+)_TrainLevel_6' ,decoder_template = 'Train_BiasDecoder_L(.+)_EvoStep_6' ,show = True)

utgen = UtilsGenome(utg, utm, tmptrain, model)
swap = utgen.get_map_of_actions_based_on_samples(model, None, 100)
""" 
"""

knapSack = KnapSack("100_5_25_1")
utg = UtilsGeneral(knapSack)
utm = UtilsModel(utg)
utp = UtilsPlot(utilsGeneral = utg, utilsModel = utm)
ute = UtilsEncoded(utilsGeneral = utg, utilsModel = utm)
m = utg.restore_model_from_directory(
    "100_10_25_7\Model_CheckPoints",
    encoder_template = 'Train_BiasEncoder_L(.+)_EvoStep_6', 
    weights_template = 'Train_Weights_L(.+)_TrainLevel_6',
    decoder_template = 'Train_BiasDecoder_L(.+)_EvoStep_6',
    show = True)

tmptrain1, tmptrain = utg.load_datasets(1, 2)

genomeUtils = UtilsGenome(utg, utm, tmptrain, m)
swap = utg.load_obj("swap")

resswap = genomeUtils.get_map_hidden_visible_samples(swap)

a,l,g = genomeUtils.get_genom_performence_distribution(resswap, 0, std_coef=2, sample_length_filter=2)

utg.save_obj(a, "accswap")
utg.save_obj(a, "lenswap")
utg.save_obj(a, "genswap")

"""