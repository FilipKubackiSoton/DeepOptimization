from KnapSack import KnapSack
from pathlib import Path
import os 
import numpy as np
import utilsModel as utm
import utilsGeneral as utg
from shallowNet.shallowNet import shallowNet
import plots as pt
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-k", "--knap", type=str, default="100_5_25_1", help="knap sack problem name")
ap.add_argument("-c", "--comp", type=float, default=0.8, help="compression coefficient")
ap.add_argument("-s", "--size", type=int, default=200, help="siz of the training set")
ap.add_argument("-e", "--epochs", type=int, default=200, help="number of epochs")
ap.add_argument("-b", "--batch", type=int, default=10, help="batch size")
ap.add_argument("-r", "--reg", type=float, default=0.0005, help="regularization coefficient")
ap.add_argument("-d", "--drop", type=float, default=0.2, help="drop out coefficient")
ap.add_argument("-l", "--lern", type=float, default=0.003, help="leraning rate")
args = vars(ap.parse_args())

knapSack = KnapSack(args["knap"])
compression = args["comp"]
train_size = args["size"]
epochs = args["epochs"]
batch_size = args["batch"]
reg_cof = args["reg"]
dropout = args["drop"]
lr = args["lern"]

trainY1 = utg.generate_training_sat(knapSack.Size, train_size) # generate training set 

model1 = shallowNet.build(
    input_shape=knapSack.Size, 
    reg_cof= reg_cof, 
    lr = lr, 
    dropout= dropout, 
    compression=compression
)

H1 = model1.fit(
    trainY1, trainY1, 
    epochs=epochs, 
    batch_size=batch_size, 
    shuffle=True,
    verbose=0)

pt.plot_model_loss(H1, "loss_plot_model1.png", epochs) # save plot in 
pt.plot_evolution_model(model1, trainY1, "tp_del.png")


trainY2 = utm.generate_enhanced_training_set(
    model = model1,
    initial_training_set = trainY1 
)  # generate enhanced training set2

model2 = shallowNet.build(
    input_shape=knapSack.Size, 
    reg_cof= reg_cof, 
    lr = lr, 
    dropout= dropout, 
    compression=compression
    )  # build model2 based on the enhanced data set: trainY2

H2 = model2.fit(
    trainY2, trainY2, 
    epochs=epochs, 
    batch_size=batch_size, 
    verbose=0, 
    shuffle=True
    ) # train new shallow model 
pt.plot_model_loss(H2, "loss_plot_model2.png", epochs) # save loss plot 

utm.add_layer_to_model(model2, show_summary=True) # add layer to the enhanced shallow model 

H3 = model2.fit(
    trainY2, trainY2, 
    epochs=epochs, 
    batch_size=batch_size, 
    verbose=1, 
    shuffle=True
    )# train new deep model

pt.plot_model_loss(H3, "loss_plot_deep_model2.png", epochs)# save loss of deep model

# Saving results
utg.save(trainY1, trainY2, model1, model2)
