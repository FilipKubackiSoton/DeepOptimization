from utilsGeneral import UtilsGeneral
from utilsModel import UtilsModel
from utilsPlot import UtilsPlot
from KnapSack import KnapSack
from shallowNet.shallowNet import shallowNet, DenseTranspose
import numpy as np
import copy
import matplotlib.pyplot as plt

knapSack = KnapSack("100_5_25_1")
utg = UtilsGeneral(knapSack)
utm = UtilsModel(utg)
utp = UtilsPlot(utilsGeneral = utg, utilsModel = utm)
fitness_function = knapSack.Fitness
train_size = 200
compression = 0.8
epochs = 200
batch_size = 20
reg_cof = 0.0002
dropout = 0.2
lr = 0.002

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

trainY2 = utm.generate_enhanced_training_set(model1, trainY1)
utg.save(trainY2)

