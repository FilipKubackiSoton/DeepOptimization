import tensorflow as tf
import numpy as np 
import os 
import matplotlib.pyplot as plt 
from shallowNet.shallowNet import shallowNet
import util as ut
import argparse
import matplotlib.cm as cm
from matplotlib.pyplot import imshow


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--aplot", type=str, default="activationL1.png",
	help="path to output activation plot file")
ap.add_argument("-w", "--wplot", type=str, default="weightsPlot.png",
	help="path to output weights plot file")
ap.add_argument("-s", "--size", type=int, default=500,
	help="siz of the validation set")
args = vars(ap.parse_args())

model = tf.keras.models.load_model('saved_model/my_model')
print(model.summary())

# generate the val set 
print("[INFO] generating validating dataset...")
(valX, valY) = ut.generate_training_sat(32, args["size"])

features_list = [layer.output for layer in model.layers[:4]]
new_model = tf.keras.Model(inputs = model.input, outputs = features_list)
predict = new_model.predict(valX)

print(np.shape(new_model.weights))
print(np.shape(new_model.weights[0]))
print(np.shape(new_model.weights[1]))
print(np.shape(new_model.weights[2]))
print(np.shape(new_model.weights[3]))


N = np.arange(0, len(predict[3][0]))
plt.figure()
for i in range(20):
    index = np.random.randint(len(predict[3][0]))
    plt.plot(N, predict[3][index], 'o',color = 'black')
plt.title("L1 activation")
plt.xlabel("Node #")
plt.ylabel("Activation value")
plt.savefig(args["aplot"])

for i in range()
print(type(model.get_weights()))
print("wieghts:", type(model.weights))
print("wieghts:", type(model.weights[2]))
print(model.weights[2])
print(type(tf.make_ndarray(tf.convert_to_tensor(value = model.weights[3], dtype = tf.float32))))
print(tf.make_ndarray(tf.convert_to_tensor(value = model.weights[2], dtype = tf.float32)))

fig = plt.figure()
ax1 = fig.add_subplot(121)
# Bilinear interpolation - this will look blurry
ax1.imshow(tf.make_ndarray(tf.convert_to_tensor(value = model.weights[2], dtype = tf.float32)), interpolation='bilinear', cmap=cm.Greys_r)
ax1.title("Encoder W1")
ax1.xlabel("Hidden Node #")
ax1.ylabel("Visible Node")

fig.savefig(args["wplot"])


