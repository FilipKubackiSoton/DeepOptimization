from Utils.utilsGeneral import UtilsGeneral
from Utils.utilsModel import UtilsModel
from Utils.utilsPlot import UtilsPlot
from Utils.utilsGenome import UtilsGenome
from KnapSack.KnapSack import KnapSack
from shallowNet.shallowNet import shallowNet, DenseTranspose
import numpy as np


knapSack = KnapSack("100_5_25_1")
utg = UtilsGeneral(knapSack)
utm = UtilsModel(utg)
utp = UtilsPlot(utilsGeneral = utg, utilsModel = utm)
fitness_function = knapSack.Fitness

set1 = np.load("100_10_25_7\TrainData\TrainingData_Layer_1_Evo1.npy") #loading dataset 1
set2 = np.load("100_10_25_7\TrainData\TrainingData_Layer_2_Evo2.npy") #loading dataset 2
set3 = np.load("100_10_25_7\TrainData\TrainingData_Layer_3_Evo3.npy") #loading dataset 3
set4 = np.load("100_10_25_7\TrainData\TrainingData_Layer_4_Evo4.npy") #loading dataset 4
set5 = np.load("100_10_25_7\TrainData\TrainingData_Layer_5_Evo5.npy") #loading dataset 5
set6 = np.load("100_10_25_7\TrainData\TrainingData_Layer_6_Evo6.npy") #loading dataset 6


################################ LOADING MODELS FROM NUMPY FILES ################################
print("\n !!![Creating model 5]")
model5 = utg.restore_model_from_directory(
    directory = "100_10_25_7\Model_CheckPoints",
    encoder_template = 'Train_BiasEncoder_L(.+)_EvoStep_5', 
    weights_template = 'Train_Weights_L(.+)_TrainLevel_5',
    decoder_template = 'Train_BiasDecoder_L(.+)_EvoStep_5',
    show = True) # create model from numpy files and show list of files from which it was created

utgen5 = UtilsGenome(utg, utm, set5, model5) # build an instance of the genomeUtils based on the set5 and model5 (both parameters are optional)

print("\n !!![Creating model 6]")
model6 = utg.restore_model_from_directory(
    directory = "100_10_25_7\Model_CheckPoints",
    encoder_template = 'Train_BiasEncoder_L(.+)_EvoStep_6', 
    weights_template = 'Train_Weights_L(.+)_TrainLevel_6',
    decoder_template = 'Train_BiasDecoder_L(.+)_EvoStep_6',
    show = True) # create model from numpy files and show list of files from which it was created

utgen6 = UtilsGenome(utg, utm, set6, model6) # build an instance of the genomeUtils based on the set5 and model5 (both parameters are optional)

################################ VISUALIZING CHANGES IN ENCODED SPACE ################################
"""
print("\n !!![Examin sample = set[1]]")
utgen6.get_actions_in_encoded_space(set1[1], show = True, title = "model6") #examin sample set1[1]
print("\n !!![Examin sample = set[2]]")
utgen6.get_actions_in_encoded_space(set1[2], show = True, title = "model6") #examin sample set1[2]
"""
################################ GENOME EXTRACTION AND CHECK  ################################
"""
print("\n !!![Creating map of actions]")
swap, single_add, grouping = utgen6.get_map_of_actions_based_on_samples( 
    model = None, 
    sample_set = set1, 
    size_set = 100) # get dictionariess of thre class of actions: swap, single add, and group

result_swap = utgen6.get_map_hidden_visible_samples(swap) # tranfrom swap directory into hidden change : vissible change: list of samples witch such change

print("\n !!![Checking genome quality]")
accuracy, length, genomes = utgen6.get_genom_performence_distribution(result_swap, 0, sample_length_filter=5) # extract genome from every vissible change and check genome accuracy 

utg.save_obj(accuracy, "accuracy") # save accuracy dictionary 
utg.save_obj(length, "length") # save length dictionary 
utg.save_obj(genomes, "genomes")# save genome dictionary 
"""





