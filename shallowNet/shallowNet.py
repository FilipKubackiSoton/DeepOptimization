import tensorflow as tf
class shallowNet:
    @staticmethod
    def build(input_size=32, latent_size=16):
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape = (input_size, 1)),
            tf.keras.layers.Dense(input_size),
            tf.keras.layers.Dense(latent_size),
            tf.keras.layers.Dense(input_size),
            tf.keras.layers.Activation("sigmoid")
        ])
        return model

