import plots as pt
from shallowNet.shallowNet import shallowNet, DenseTranspose
import numpy as np
from KnapSack import KnapSack
import utilsModel as utm
import utilsGeneral as utg

knapSack = KnapSack("100_5_25_1")

model1, model2, model3 = utg.load_models(1,2,3)
train1, train2 = utg.load_datasets(1,2)

pt.plot_evolution_model(model1, train1, "evolution_plot_model_1_train1.png")
pt.plot_evolution_model(model2, train1, "evolution_plot_model_2_train1.png")
pt.plot_evolution_model(model3, train1, "evolution_plot_model_3_train1.png")

pt.plot_weights_model(model1, "weights_plot_model_1.png")
pt.plot_weights_model(model2, "weights_plot_model_2.png")
pt.plot_weights_model(model3, "weights_plot_model_3.png")

pt.plot_trajectory_evolution(model1, train1,"trajectory_plot_evolution_model1" )
pt.plot_trajectory_evolution(model2, train1,"trajectory_plot_evolution_model2" )
pt.plot_trajectory_evolution(model3, train1,"trajectory_plot_evolution_model3" )

pt.plot_fitness_development_phase(model1, train1, plot_name="fitness_plot_development_model_1.png")
pt.plot_fitness_development_phase(model2, train1, plot_name="fitness_plot_development_model_2.png")
