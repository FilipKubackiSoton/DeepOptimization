import matplotlib

matplotlib.use("Agg")
from shallowNet.shallowNet import shallowNet
import util as ut
import plots as pt
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import tensorflow as tf
import os
import shutil
import matplotlib.cm as cm
from matplotlib.pyplot import imshow
from pathlib import Path
from tensorflow.keras.optimizers import Adam

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument(
    "-i",
    "--input",
    type=int,
    default=32,
    help="# dimension of the string passed into the net",
)
ap.add_argument(
    "-c", "--comp", type=float, default=0.8, help=" compression coefficient"
)
ap.add_argument("-s", "--size", type=int, default=100, help="siz of the training set")
ap.add_argument("-e", "--epochs", type=int, default=40, help="number of epochs")
ap.add_argument("-b", "--batch", type=int, default=4, help="batch size")
ap.add_argument(
    "-a",
    "--actplt",
    type=str,
    default="activationL1.png",
    help="name of the plot of the activation",
)
ap.add_argument(
    "-r", "--reg", type=float, default=0.001, help="regularization coefficient"
)
ap.add_argument("-d", "--drop", type=float, default=1.0, help="drop out coefficient ")
ap.add_argument(
    "-p",
    "--aplot",
    type=str,
    default="activationL1.png",
    help="path to output activation plot file",
)
ap.add_argument(
    "-w",
    "--wplot",
    type=str,
    default="weightsEnDe.png",
    help="path to output weights plot file",
)
args = vars(ap.parse_args())

input_size = args["input"]
compression = args["comp"]
set_size = args["size"]
epochs = args["epochs"]
batch_size = args["batch"]

print(
    "input_size: ",
    input_size,
    " compression: ",
    compression,
    " set_size: ",
    set_size,
    " epochs: ",
    epochs,
    " batch: ",
    batch_size,
)
# generate the trainig set
print("[INFO] generating trainnig dataset...")
(trainX, trainY) = ut.generate_training_sat(input_size, set_size)

# generate the test set
print("[INFO] generating testing dataset...")
(testX, testY) = ut.generate_training_sat(input_size, int(set_size / 10))


# build and compile model
model1 = shallowNet.build(
    input_shape=input_size, compression=compression, reg_cof=args["reg"]
)

# train the model
H1 = model1.fit(trainY, trainY, epochs=epochs, batch_size=batch_size, shuffle=True)
# save the model
ut.save_model(model1)
# show model structure
model1.summary()
# pt.plot_model_loss(H1, "loss_plot_model_1.png", epochs)
# pt.plot_weights_model(model1, "weights_plot_model_1.png")
encoder1, decoder1 = ut.split_model_into_encoder_decoder(model1, show_summary=True)
trainY2 = ut.generate_enhanced_training_set(
    trainY, encoder1, decoder1
)  # generate enhanced training set2
model2 = shallowNet.build()  # build model2 based on the enhanced data set: trainY2
H2 = model2.fit(
    trainY2, trainY2, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True
)
# pt.plot_model_loss(H2, "loss_plot_model_2.png", epochs)
# pt.plot_weights_model(model2, "weights_plot_model_2.png")
pt.plot_evolution_model(
    encoder1, decoder1, trainY, plot_name="evolution model ed1 tY.png"
)
pt.plot_global_trajectory(
    encoder1,
    decoder1,
    trainY,
    debuge_variation=True,
    epochs=10,
    plot_name="Global trajectory ed1 tY.png",
)
model3 = ut.add_layer_to_model(model2, show_summary=True)
H3 = model3.fit(
    trainY2, trainY2, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True
)
ut.save_model(model3)

pt.plot_model_loss(H3, "to_delete.png", epochs)
pt.plot_weights_model(model3, "to_delete_2.png")
encoder3, decoder3 = ut.split_model_into_encoder_decoder(model3, True)
pt.plot_global_trajectory(
    encoder3,
    decoder3,
    trainY,
    debuge_variation=True,
    epochs=10,
    plot_name="Global trajectory ed3 ty1.png",
)
progress_set_evidence1 = ut.transfer_sample_latent_flip(trainY[0], encoder1, decoder1)[
    -1
]
progress_set_evidence3 = ut.transfer_sample_latent_flip(trainY[0], encoder3, decoder3)[
    -1
]
pt.plot_fitness_development_phase(progress_set_evidence1, plot_name="fitnes1.png")
pt.plot_fitness_development_phase(progress_set_evidence3, plot_name="fitnes3.png")


########################################
########SAVING MODEL AND PLOTS #########
########################################
"""
plot_dir = os.path.join("plots")  # plots dir
plot_path = Path(plot_dir)  # plots path


trainY2 = ut.generate_new_training_set(
    trainY, encoder, decoder
)  # generating enhanced data set
model2 = shallowNet.build(
    input_shape=input_size, compression=compression, reg_cof=args["reg"]
)  # create and build new model
H2 = model2.fit(  # train the new model based on the enhanced training set
    trainY2, trainY2, epochs=epochs, batch_size=batch, shuffle=True
)
"""
