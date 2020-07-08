from KnapSack import KnapSack
from pathlib import Path
import os 
import numpy as np
import util as ut
from shallowNet.shallowNet import shallowNet
import plots as pt

knapSack = KnapSack("100_5_25_1")

epochs = 200
train_size = 200
batch_size = 10

trainY1 = ut.generate_training_sat(knapSack.Size, train_size) # generate training set 
ut.save_dataset(trainY1) # save training set 

model1 = shallowNet.build(
    input_shape=knapSack.Size, 
    reg_cof= 0.0005, 
    lr = 0.003, 
    dropout= 0.2, 
    compression=0.8
)

H1 = model1.fit(
    trainY1, trainY1, 
    epochs=epochs, 
    batch_size=batch_size, 
    shuffle=True,
    verbose=0)

pt.plot_model_loss(H1, "loss_plot_model1.png", epochs) # save plot in 
pt.plot_evolution_model(model1, trainY1, "tp_del.png")


trainY2 = ut.generate_enhanced_training_set(
    model = model1,
    initial_training_set = trainY1 
)  # generate enhanced training set2

ut.save_dataset(trainY2) # save enhanced data set 
model2 = shallowNet.build(
    input_shape=knapSack.Size, 
    reg_cof= 0.0005, 
    lr = 0.003, 
    dropout= 0.2, 
    compression=0.8
    )  # build model2 based on the enhanced data set: trainY2

H2 = model2.fit(
    trainY2, trainY2, 
    epochs=epochs, 
    batch_size=batch_size, 
    verbose=0, 
    shuffle=True
    ) # train new shallow model 
ut.save_model(model2) # save new shallow model 
pt.plot_model_loss(H2, "loss_plot_model2.png", epochs) # save loss plot 

model3 = ut.add_layer_to_model(model2, show_summary=True) # add layer to the enhanced shallow model 

H3 = model3.fit(
    trainY2, trainY2, 
    epochs=epochs, 
    batch_size=batch_size, 
    verbose=1, 
    shuffle=True
    )# train new deep model
ut.save_model(model3) # save deep model
pt.plot_model_loss(H3, "loss_plot_model3.png", epochs)# save loss of deep model
