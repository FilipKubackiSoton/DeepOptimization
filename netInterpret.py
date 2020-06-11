#%%
import tensorflow as tf
import numpy as np 
import os 
import matplotlib.pyplot as plt 
from shallowNet.shallowNet import shallowNet
import util as ut
import argparse
import matplotlib.cm as cm
from matplotlib.pyplot import imshow

#%%
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--aplot", type=str, default="activationL1.png",
	help="path to output activation plot file")
ap.add_argument("-w", "--wplot", type=str, default="weightsEnDe.png",
	help="path to output weights plot file")
ap.add_argument("-s", "--size", type=int, default=500,
	help="siz of the validation set")
args = vars(ap.parse_args())

model = tf.keras.models.load_model('saved_model/my_model')
print(model.summary())


for i in range(len(model.weights)):
	tmp = model.get_weights()[i]
	print("layer:", i, " ", np.shape(tmp), " ", type(tmp))

fig, axs = plt.subplots(2)
fig.suptitle('Weights matrix encoder/decoder', fontsize=16)
fig.subplots_adjust(hspace =0.5)
pcm = axs[0].imshow(model.get_weights()[2].transpose(), interpolation='nearest', cmap=cm.Greys_r)
axs[0].set_title("Encoder")
axs[0].set_ylabel("Hidden Node #")
axs[0].set_xlabel("Visible Node #")

pcm=axs[1].imshow(model.get_weights()[4], interpolation='nearest', cmap=cm.Greys_r)
axs[1].set_title("Decoder")
axs[1].set_ylabel("Hidden Node #")
axs[1].set_xlabel("Visible Node #")
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
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
