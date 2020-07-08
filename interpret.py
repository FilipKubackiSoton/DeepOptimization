import util as ut
import plots as pt
from shallowNet.shallowNet import shallowNet, DenseTranspose
import numpy as np
from KnapSack import KnapSack

knapSack = KnapSack("100_5_25_1")

model1 = ut.load_model(model_index = 1) # load model 1
model2 = ut.load_model(model_index = 2) # load model 1
model3 = ut.load_model(model_index = 3) # load model 1

train1 = ut.load_dataset(dataset_index = 1) # load data set 1
train2 = ut.load_dataset(dataset_index = 2) # load data set 2
"""

pt.plot_evolution_model(model1, train1, "evolution_plot_model_1_train1.png")
pt.plot_evolution_model(model2, train1, "evolution_plot_model_2_train1.png")
pt.plot_evolution_model(model3, train1, "evolution_plot_model_3_train1.png")
"""
pt.plot_weights_model(model1, "weights_plot_model_1.png")
pt.plot_weights_model(model2, "weights_plot_model_2.png")
pt.plot_weights_model(model3, "weights_plot_model_3.png")

pt.plot_trajectory_evolution(model1, train1,"trajectory_plot_evolution_model1" )
pt.plot_trajectory_evolution(model2, train1,"trajectory_plot_evolution_model1" )
pt.plot_trajectory_evolution(model3, train1,"trajectory_plot_evolution_model3" )

progress_set_evidence1 = ut.transfer_sample_latent_flip(model1, train1[0])[-1]
progress_set_evidence2 = ut.transfer_sample_latent_flip(model2, train1[0])[-1]
pt.plot_fitness_development_phase(progress_set_evidence1, plot_name="fitness_plot_development_model_1.png")
pt.plot_fitness_development_phase(progress_set_evidence2, plot_name="fitness_plot_development_model_2.png")
