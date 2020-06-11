import tensorflow as tf
class shallowNet:
    @staticmethod
    def build(input_size=32, latent_size=16, reg_cof = 0.001, dropout =0.2):
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

        
