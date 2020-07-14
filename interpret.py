from KnapSack import KnapSack
from pathlib import Path
import os 
import numpy as np
from utilsGeneral import UtilsGeneral
from utilsModel import UtilsModel
from utilsPlot import UtilsPlot
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-k", "--knap", type=str, default="100_5_25_1", help="knap sack problem name")
args = vars(ap.parse_args())

knapSack = KnapSack(args["knap"])
utg = UtilsGeneral(knapSack)
utm = UtilsModel(utg)
utp = UtilsPlot(utilsGeneral = utg, utilsModel = utm)

model1, model2, model3 = utg.load_models(1,2,3)
train1, train2 = utg.load_datasets(1,2)


utp.plot_evolution_model(model1, train1, "evolution_plot_model_1_train1.png")
utp.plot_evolution_model(model2, train1, "evolution_plot_model_2_train1.png")
utp.plot_evolution_model(model3, train1, "evolution_plot_model_3_train1.png")

utp.plot_weights_model(model1, "weights_plot_model_1.png")
utp.plot_weights_model(model2, "weights_plot_model_2.png")
utp.plot_weights_model(model3, "weights_plot_model_3.png")
"""
utp.plot_trajectory_evolution(knapSack.Size, plot_name = "trajectory_plot_evolution_no_model.png" )
utp.plot_trajectory_evolution(knapSack.Size, model = model1, plot_name = "trajectory_plot_evolution_model1.png" )
utp.plot_trajectory_evolution(knapSack.Size, model = model2, plot_name = "trajectory_plot_evolution_model2.png" )
utp.plot_trajectory_evolution(knapSack.Size, model = model3, plot_name = "trajectory_plot_evolution_model3.png" )

utp.plot_fitness_development_phase(model1, train1, plot_name="fitness_plot_development_model_1.png")
utp.plot_fitness_development_phase(model2, train1, plot_name="fitness_plot_development_model_2.png")
utp.plot_fitness_development_phase(model3, train1, plot_name="fitness_plot_development_model_3.png")
"""
