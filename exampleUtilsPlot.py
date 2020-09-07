from  Utils.utilsGeneral import UtilsGeneral
from  Utils.utilsModel import UtilsModel
from  Utils.utilsPlot import UtilsPlot
from  Utils.utilsHyperTuner import UtilsHyperTuner
from  Utils.utilsGenome import UtilsGenome
from KnapSack.KnapSack import KnapSack
from shallowNet.shallowNet import shallowNet, DenseTranspose
import numpy as np

knapSack = KnapSack("100_10_25_7")
utg = UtilsGeneral(knapSack)
utm = UtilsModel(utg)
utp = UtilsPlot(utilsGeneral = utg, utilsModel = utm)

sample_set = np.load("100_10_25_7\TrainData\TrainingData_Layer_1_Evo1.npy")
model6 = utg.restore_model_from_directory("100_10_25_7\Model_CheckPoints", encoder_template = 'Train_BiasEncoder_L(.+)_EvoStep_6', weights_template = 'Train_Weights_L(.+)_TrainLevel_6' ,decoder_template = 'Train_BiasDecoder_L(.+)_EvoStep_6' ,show = True)
utgen6 = UtilsGenome(utg, utm, sample_set, model6)


################################ PLOT TRAJECTORY EVOLUTION ################################
#utp.plot_trajectory_evolution(100, learning_steps = 100, sample_number=10, model = model6)

################################ PLOT LATENT ACTIVATION ################################
#utp.plot_latent_acitvation(model = model6, sample_set = sample_set)

################################ PLOT MODEL EVOLUTION ################################
#utp.plot_evolution_model(model = model6, sample_set = sample_set)

################################ PLOT FITNESS DEVELOPMENT PHASE ################################
#utp.plot_fitness_development_phase(model6, sample_set)

################################ PLOT WEIGHTS IN MODEL ################################
#utp.plot_weights_model(model6, 1,2,4,5, show = True, decoder = True)

################################ PLOT CORRELATION MATRIX ################################
#utp.plot_correlation_matrix(model6)

################################ PLOT SET PROBABILITY AND VALUES ################################
#utp.plot_set_probability_and_values(sample_set, True)

################################ PLOT LATENT ACTIVATION DISTRIBUTION ################################
#utp.plot_latent_activation_distribution(sample_set, model6, probability = True, sort = True, plot_name= "todeletefinal.png")

################################ PLOT LATENT ACTIVATION  ################################
#utp.plot_latent_activation(model6, title = "modelTmp", background_activation=-1, unit_sort = False, index_sort=True, column_sort=True, log_conversion = True)
#utp.plot_latent_activation(model6, title = "modelTmp", background_activation=-1, unit_sort = True, index_sort=True, column_sort=True, log_conversion = True)
#utp.plot_latent_activation(model6, title = "modelTmp", background_activation=-1, unit_sort = False, index_sort=False, column_sort=True, log_conversion = False)
