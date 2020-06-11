# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
# import the necessary packages
from shallowNet.shallowNet import shallowNet
from tensorflow.keras.optimizers import Adam
import util as ut
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import tensorflow as tf
import os
import shutil
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=int, default=32,
	help="# dimension of the string passed into the net")
ap.add_argument("-l", "--latent", type=int, default=16,
	help=" dimension of the bottle neck of the autoencoder ")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output plot file")
ap.add_argument("-s", "--size", type=int, default=5000,
	help="siz of the training set")    
ap.add_argument("-e", "--epochs", type=int, default=25,
    help="number of epochs")   
ap.add_argument("-b", "--batch", type=int, default=25,
    help="batch size")   
ap.add_argument("-a", "--actplt", type=str, default="activationL1.png",
    help="name of the plot of the activation")    
ap.add_argument("-r", "--reg", type=int, default=0.01,
    help="regularization coefficient")    
ap.add_argument("-d", "--drop", type=int, default=0.2,
    help="drop out coefficient ")    
args = vars(ap.parse_args())

input_size = args["input"]
latent_size = args["latent"]
set_size = args["size"]
epochs = args["epochs"]
batch = args["batch"]

print("input_size: ", input_size, " latent_size: ", latent_size, " set_size: ", set_size, " epochs: ", epochs, " batch: ", batch)

# generate the trainig set 
print("[INFO] generating trainnig dataset...")
(trainX, trainY) = ut.generate_training_sat(input_size, set_size)

# generate the test set 
print("[INFO] generating testing dataset...")
(testX, testY) = ut.generate_training_sat(input_size,int(set_size/10))


#model = shallowNet.build(input_size = input_size, latent_size=latent_size)
model = shallowNet.build(input_size=input_size, latent_size=latent_size, reg_cof=args["reg"])

opt = Adam(lr=1e-3)
model.compile(loss="mse", optimizer=opt)

H = model.fit(
    trainX, trainY,
    validation_data = (testX, testY),
    epochs = epochs,
    batch_size = batch,
)

# construct a plot that plots and saves the training history
N = np.arange(0, epochs)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])


# Save the entire model as a SavedModel.
dir = "saved_model"
if os.path.exists(dir):
    shutil.rmtree(dir)
os.makedirs(dir)
model.save(
    os.path.join(dir, "my_model")
    ) 

predict_model = model.predict(testX)
print("Predicted first 3 vaues:")
print(predict_model[:3])
print("True first 3 vaues:")
print(testY[:3])
