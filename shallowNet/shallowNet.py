import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten
import tensorflow.keras.optimizers
import numpy as np



class DenseTranspose(tf.keras.layers.Layer):
    def __init__(self, dense, activation=None, **kwargs):
        self.dense = dense
        self.activation = tf.keras.activations.get(activation)
        super(DenseTranspose, self).__init__(**kwargs)

    def build(self, batch_input_shape):
        self.b = self.add_weight(
            name="bias", shape=[self.dense.input_shape[-1]], initializer="zeros"
        )
        self.w = self.dense.weights[0]
        super().build(batch_input_shape)

    def call(self, inputs):
        z = tf.linalg.matmul(inputs, self.w, transpose_b=True)
        return self.activation(z + self.b)

    def get_weights(self):
        return self.w.numpy()

    @property
    def weights_transpose(self):
        return tf.transpose(self.dense.weights[0])


class shallowNet:
    @staticmethod
    def build(input_shape=32, compression=0.8, reg_cof=(0.001, 0.001), dropout=0.2, name="NN 1", lr = 0.01, loss = "mse", metrics = None, optimizer =  tensorflow.keras.optimizers.Adam):
        
        inputs = Input(shape=(input_shape,))
        encoder = Dense(
            int(input_shape * compression),
            activation="tanh",
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            bias_initializer=tf.keras.initializers.Zeros(),
            kernel_regularizer=tf.keras.regularizers.L1L2(reg_cof[0], reg_cof[1]),
        )
        decoder = DenseTranspose(dense=encoder)
        # the model
        x = Dropout(dropout)(inputs)
        encoded = encoder(x)
        decoded = decoder(encoded)
        model = tf.keras.Model(inputs, decoded)
        opt = optimizer(lr=lr)
        if metrics == None:
            model.compile(loss=loss, optimizer=opt)
        else:
            model.compile(loss=loss, optimizer=opt, metrics=metrics)
        #model.summary()
        return model


"""
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
import numpy as np

class DenseTranspose(tf.keras.layers.Layer):
    def __init__(self, dense, activation = None, **kwargs):
        self.dense = dense 
        self.activation = tf.keras.activations.get(activation)
        super(DenseTranspose, self).__init__(**kwargs)
    def build(self, batch_input_shape):
        self.b = self.add_weight(name= "bias", shape = [ self.dense.input_shape[-1]], initializer = "zeros")
        self.w = self.dense.weights[0]
        super().build(batch_input_shape)
        
        
    def call(self, inputs):
        z = tf.linalg.matmul(inputs, self.w, transpose_b = True)
        return self.activation(z + self.b)
    
    def get_weights(self):
        return {"w": np.shape(tf.transpose(self.w))}    
    @property 
    def weights_transpose(self):
        return tf.transpose(self.dense.weights[0])
        
class shallowNet:
    @staticmethod
    def build(input_shape=32, compression=0.8, reg_cof = 0.001, dropout =0.2):
        assert compression <1 and compression >0, "compression coefficient must be between (0,1)" % compression
        assert dropout <1 and dropout >0, "dropout coefficient must be between (0,1)" % dropout
        
        inputs = Input(shape=(input_shape,))
        encoder = Dense(int(input_shape * compression),activation="tanh",kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
        bias_initializer=tf.keras.initializers.Zeros(),kernel_regularizer=tf.keras.regularizers.l1(reg_cof))
        decoder = DenseTranspose(dense = encoder)
        #the model 
        x = Dropout(dropout)(inputs)
        encoded = encoder(x)
        decoded = decoder(encoded)
        model = tf.keras.Model(inputs, decoded)
        opt = Adam(lr=0.01)
        model.compile(loss='mse', optimizer=opt)
        model.summary()
        return model
"""


class NumpyInitializer(tf.keras.initializers.Initializer):

    def __init__(self, bias):
        self.bias = tf.convert_to_tensor(bias.tolist())
        
    def __call__(self, shape, dtype=None):
        return self.bias 

Direct = "100_5_25_1Knapsack_Layer1\\100_5_25_1Knapsack"

def restore_model_from_numpy(directory):

    def file_iterating(directory):
        pathlist = Path(directory).rglob("*.npy")
        layers = {}
        index = 0
        for file in pathlist:
            if index % 2 ==0:
                layers[int(index/2)] = []
            layers[int(index/2)].append(np.load(file))
            index +=1
            print(file)
        return layers
    

    layers = file_iterating(directory)
    layers_numbers = len(layers)

    inputs = Input(shape = (np.shape(layers[0][1])[0]))
    x = inputs 
    for key, value in layers.items():
        if key< int(layers_numbers/2):
            x = Dropout(0.)(x)
        bias_initializer = NumpyInitializer(layers[key][0][0])
        kernal_initializer = NumpyInitializer(layers[key][1])
        layer_size = np.shape(layers[key][0])[-1]
        new_layer = tf.keras.layers.Dense(
            units = layer_size, 
            kernel_initializer=kernal_initializer, 
            bias_initializer = bias_initializer,
            activation="tanh")
        new_layer.trainable = False
        x = new_layer(x)
        
    model = tf.keras.Model(inputs, x)
    model.compile()
    return model