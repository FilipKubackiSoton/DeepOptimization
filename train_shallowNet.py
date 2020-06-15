# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
# import the necessary packages
from shallowNet.shallowNet import shallowNet
from shallowNet.shallowNet import Autoencoder
import util as ut
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
ap.add_argument("-i", "--input", type=int, default=32,
	help="# dimension of the string passed into the net")
ap.add_argument("-c", "--comp", type=float, default=0.8,
	help=" compression coefficient")
ap.add_argument("-s", "--size", type=int, default=100,
	help="siz of the training set")    
ap.add_argument("-e", "--epochs", type=int, default=40,
    help="number of epochs")   
ap.add_argument("-b", "--batch", type=int, default=4,
    help="batch size")   
ap.add_argument("-a", "--actplt", type=str, default="activationL1.png",
    help="name of the plot of the activation")    
ap.add_argument("-r", "--reg", type=float, default=0.001,
    help="regularization coefficient")    
ap.add_argument("-d", "--drop", type=float, default=1.0,
    help="drop out coefficient ")    
ap.add_argument("-p", "--aplot", type=str, default="activationL1.png",
	help="path to output activation plot file")
ap.add_argument("-w", "--wplot", type=str, default="weightsEnDe.png",
	help="path to output weights plot file")
args = vars(ap.parse_args())

input_size = args["input"]
compression = args["comp"]
set_size = args["size"]
epochs = args["epochs"]
batch = args["batch"]

print("input_size: ", input_size, " compression: ", compression, " set_size: ", set_size, " epochs: ", epochs, " batch: ", batch)


# generate the trainig set 
print("[INFO] generating trainnig dataset...")
(trainX, trainY) = ut.generate_training_sat(input_size, set_size)

# generate the test set 
print("[INFO] generating testing dataset...")
(testX, testY) = ut.generate_training_sat(input_size,int(set_size/10))

#model = shallowNet.build(input_size=input_size, compression=compression, reg_cof=args["reg"])
model = Autoencoder(input_size=32, compression = 0.8, dropout = 0.2, reg_cof = 0.001)
opt = Adam(lr=0.01)
model.compile(loss='mse', optimizer=opt)
H = model.fit(
    trainY, trainY,
    #validation_data = (testY, testY),
    epochs = epochs,
    batch_size = batch,
    #shuffle = True
)
model.summary()

########################################
########SAVING MODEL AND PLOT ##########
########################################

model_name  = (
    "NN-epo" + str(args["epochs"]) + 
    "-dro" + str('%3f' % args["drop"])+
    "-reg" + str('%3f' % args["reg"])+
    "-com" + str('%3f' % args["comp"])
    )

model_dir = os.path.join("saved_model", str(model_name))
plot_dir = os.path.join("plots", str(model_name))

model_path = Path(model_dir)
plot_path = Path(plot_dir)

try:
    model_path.rmdir()
except OSError as e:
    print(f'Error: {model_path} : {e.strerror}')
    
try:
    plot_path.rmdir()
except OSError as e:
    print(f'Error: {plot_path} : {e.strerror}')
    
model_path.mkdir(exist_ok = True, parents=True)
plot_path.mkdir(exist_ok = True, parents=True)

# Save the entire model as a SavedModel.
model.save(model_dir) 
dir = "last_model"
if os.path.exists(dir):
    shutil.rmtree(dir)
os.makedirs(dir)
model.save(
    os.path.join(dir, "last_model")
) 


# construct a plot that plots and saves the training history
N = np.arange(0, epochs)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
#plt.plot(N, H.history["val_loss"], label="val_loss")
plt.title("Training Loss and Accuracy\n"+model_name)
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")

#save loss plot 
plt.savefig(os.path.join(plot_dir, "loss.png"))
"""
with open(os.path.join(plot_dir,"reproduce"+".txt"), 'w') as f:
    data = str(
        'python -e '+args["epochs"]+
        " -d "+str('%3f' %args["drop"])+
        " -r "+str('%3f' % args["reg"])+
        " -c "+str('%2f' % args["comp"])
    )
    f.write(data)


"""

for i in range(len(model.weights)):
	tmp = model.get_weights()[i]
	print("layer:", i, " ", np.shape(tmp), " ", type(tmp))

fig, axs = plt.subplots(1, 2)
fig.suptitle('Weights matrix encoder/decoder', fontsize=16)
fig.subplots_adjust(hspace =0.5)
pcm = axs[0].imshow(model.get_weights()[0], interpolation='nearest', cmap=cm.Greys_r)
axs[0].set_title("Encoder")
axs[0].set_ylabel("Hidden Node #")
axs[0].set_xlabel("Visible Node #")

pcm=axs[1].imshow(model.get_weights()[0], interpolation='nearest', cmap=cm.Greys_r)
axs[1].set_title("Decoder")
axs[1].set_ylabel("Hidden Node #")
axs[1].set_xlabel("Visible Node #")
cbar_ax = fig.add_axes([0.90, 0.20, 0.05, 0.7])
fig.colorbar(pcm, cax=cbar_ax)
plt.savefig(args["wplot"])
print("[INFO]: Weights of encoder and Decoder saved in the file: ", args["wplot"])


# generate the val set 
print("[INFO] generating validating dataset...")
(valX, valY) = ut.generate_training_sat(32, args["size"])

features_list = [layer.output for layer in model.layers[:4]]
new_model = tf.keras.Model(inputs = model.input, outputs = features_list)
predict = new_model.predict(valX)
N = np.arange(0, len(predict[3][0]))

plt.figure()
for i in range(20):
    index = np.random.randint(len(predict[3][0]))
    plt.plot(N, predict[3][index], 'o',color = 'black')
plt.title("L1 activation")
plt.xlabel("Node #")
plt.ylabel("Activation value")
plt.savefig(args["aplot"])
print("[INFO]: Activation layer saved in the file: ", args["aplot"])


