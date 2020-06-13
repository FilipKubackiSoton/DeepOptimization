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
    def build(input_size=32, compression=0.8, reg_cof = 0.01, dropout =0.2):
        assert compression <1 and compression >0, "compression coefficient must be between (0,1)" % compression
        assert dropout <1 and dropout >0, "dropout coefficient must be between (0,1)" % dropout
        
        inputs = Input(input_size)
        encoder = Dense(input_size, kernel_regularizer=tf.keras.regularizers.l1(reg_cof))
        latent = Dense(int(input_size * compression), activation="tanh")
        decoder = DenseTranspose(dense = latent)
        #the model 
        x = encoder(inputs)
        x = Dropout(dropout)(x)
        x = latent(x)
        x = decoder(x)
        model = tf.keras.Model(inputs= inputs, outputs = x)
        opt = Adam(lr=1e-3)
        model.compile(loss="mse", optimizer=opt)
        model.summary()

        return model

"""
    @staticmethod
    def build(input_size=32, latent_size=16, reg_cof = 0.0, dropout =0.2):


        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape = (input_size, 1)),
            tf.keras.layers.Dense(input_size, kernel_regularizer=tf.keras.regularizers.l1(reg_cof)),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(latent_size, activation="tanh"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(input_size, activation="sigmoid")
        ])
        return model



class customShallowNet:
    @staticmethod
    def build(input_size =32, filters = (16,8), dropout=0, regul = (0,0)):
        assert((dropout >=1 or dropout < 0), "dropout must be in the range [0,1]")
        assert(not(regul[0]==1 or regul[0]==2),"You can choose between regularizers: L1 - 1 and L2 - 2")
        assert(regul[1]<0,"Regularization coefficient cannot be negative" )

        if regul[0]==1:
            reg = tf.keras.regularizers.l1(regul[1])
        elif regul[0]==2:
            reg = tf.keras.regularizers.l2(regul[1])
        else :
            reg = None

        inputs = tf.keras.layers.Flatten(input_shape = (input_size, 1))
        x = inputs
        for f in filters:
            x = tf.keras.layers.Dense(f, kernel_regularizer = reg)(x)
            if not dropout == 0:
                x = tf.keras.layers.Dropout(dropout)
        latent = x
        encoder  = tf.keras.models.Model(inputs, latent, name = "encoder")

        return encoder

        
"""