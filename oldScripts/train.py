from KnapSack import KnapSack
from pathlib import Path
import os 
import numpy as np
from utilsGeneral import UtilsGeneral
from utilsModel import UtilsModel
from utilsPlot import UtilsPlot
from shallowNet.shallowNet import shallowNet
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-k", "--knap", type=str, default="100_5_25_1", help="knap sack problem name")
ap.add_argument("-c", "--comp", type=float, default=0.8, help="compression coefficient")
ap.add_argument("-s", "--size", type=int, default=200, help="siz of the training set")
ap.add_argument("-e", "--epochs", type=int, default=200, help="number of epochs")
ap.add_argument("-b", "--batch", type=int, default=20, help="batch size")
ap.add_argument("-r", "--reg", type=float, default=0.0002, help="regularization coefficient")
ap.add_argument("-d", "--drop", type=float, default=0.2, help="drop out coefficient")
ap.add_argument("-l", "--lern", type=float, default=0.002, help="leraning rate")
args = vars(ap.parse_args())

knapSack = KnapSack(args["knap"])
compression = args["comp"]
train_size = args["size"]
epochs = args["epochs"]
batch_size = args["batch"]
reg_cof = args["reg"]
dropout = args["drop"]
lr = args["lern"]

utg = UtilsGeneral(knapSack)
utm = UtilsModel(utg)
utp = UtilsPlot(utilsGeneral = utg, utilsModel = utm)

#trainY1 = utg.generate_training_set(knapSack.Size,train_size)
#utg.save(trainY1)
trainY1, trainY2 = utg.load_datasets(1,2)

model1 = shallowNet.build(
    input_shape=knapSack.Size, 
    reg_cof= reg_cof, 
    lr = lr, 
    dropout= dropout, 
    compression=compression
)

H1 = model1.fit(
    trainY1, trainY1, 
    validation_split = 0.05,
    epochs=epochs, 
    batch_size=batch_size, 
    shuffle=True,
    verbose=0)
utp.plot_model_loss(H1, "loss_plot_model_1.png", epochs)

#trainY2 = utm.generate_enhanced_training_set(model1, trainY1)
#utg.save(trainY2)

model2 = shallowNet.build(
    input_shape=knapSack.Size, 
    reg_cof= reg_cof, 
    lr = lr, 
    dropout= dropout, 
    compression=compression
)

H2 = model2.fit(
    trainY2, trainY2, 
    validation_split = 0.05,
    epochs=epochs, 
    batch_size=batch_size, 
    shuffle=True,
    verbose=0)
utp.plot_model_loss(H2, "loss_plot_model_2.png", epochs)

model3 = utm.add_layer_to_model(model2, compression, dropout, reg_cof,lr)

H3 = model3.fit(
    trainY2, trainY2, 
    validation_split = 0.05,
    epochs=epochs, 
    batch_size=batch_size, 
    shuffle=True,
    verbose=0)
utp.plot_model_loss(H3, "loss_plot_model_3.png", epochs)

utg.save(model1, model2, model3)

#trainY2 = utm.generate_enhanced_training_set(model1, trainY1)
#utg.save(trainY2)
