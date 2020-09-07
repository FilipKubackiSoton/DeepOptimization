from utilsGeneral import UtilsGeneral
from utilsModel import UtilsModel
from utilsPlot import UtilsPlot
from utilsEncoded import UtilsEncoded
from utilsGenome import UtilsGenome
from KnapSack.KnapSack import KnapSack
import numpy as np


knapSack = KnapSack("100_10_25_7")
utg = UtilsGeneral(knapSack)
utm = UtilsModel(utg)
utp = UtilsPlot(utilsGeneral = utg, utilsModel = utm)
ute = UtilsEncoded(utilsGeneral = utg, utilsModel = utm)


sample_set = np.load("100_10_25_7\TrainData\TrainingData_Layer_1_Evo1.npy")
ute.hyper_params_tuner(trainingSet = sample_set, epochs = 1, compression = 0.8, batch_size = 20,
 reg_cof = [[0.001,0.01],[0.0001,0.001],[0.00001,0.0001]],
  dropout = 0.2 , lr = [0.001, 0.0001], validation_split = 0.05)