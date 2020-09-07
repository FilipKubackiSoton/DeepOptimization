from  Utils.utilsGeneral import UtilsGeneral
from  Utils.utilsModel import UtilsModel
from  Utils.utilsPlot import UtilsPlot
from  Utils.utilsHyperTuner import UtilsHyperTuner
from  Utils.utilsGenome import UtilsGenome
from KnapSack.KnapSack import KnapSack
import numpy as np


knapSack = KnapSack("100_10_25_7")
utg = UtilsGeneral(knapSack)
utm = UtilsModel(utg)
utp = UtilsPlot(utilsGeneral = utg, utilsModel = utm)
uth = UtilsHyperTuner(utilsGeneral = utg, utilsModel = utm)


sample_set = np.load("100_10_25_7\TrainData\TrainingData_Layer_1_Evo1.npy")
uth.hyper_params_tuner(trainingSet = sample_set, epochs = 1, compression = 0.8, batch_size = 20,
 reg_cof = [[0.001,0.01],[0.0001,0.001],[0.00001,0.0001]],
  dropout = 0.2 , lr = [0.001, 0.0001], validation_split = 0.05)