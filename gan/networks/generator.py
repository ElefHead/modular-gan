import tensorflow as tf 
from tensorflow import keras


class Generator(keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.dense = keras.layers.Dense(7*7*256, use_bias=False, input_shape=(100,), name="Dense")
        self.bn0 = keras.layers.BatchNormalization(name="BatchNorm0")

        self.rshpe_lyr = keras.layers.Reshape((7, 7, 256), name="Reshape1")

        self.conv2dt1 = keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding="same", use_bias=False, name="Con2DTranspose1")
        self.bn1 = keras.layers.BatchNormalization(name="BatchNorm1")

        self.conv2dt2 = keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same", use_bias=False, name="Con2DTranspose2")
        self.bn2 = keras.layers.BatchNormalization(name="BatchNorm2")

        self.conv2dt3 = keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding="same", use_bias=False, activation="tanh", name="Con2DTranspose3")


    def call(self, x):
        x = self.dense(x)
        x = tf.nn.leaky_relu(self.bn0(x))
        
        x = self.rshpe_lyr(x)

        x = self.conv2dt1(x)
        x = tf.nn.leaky_relu(self.bn1(x))
        
        x = self.conv2dt2(x)
        x = tf.nn.leaky_relu(self.bn2(x))
        
        x = self.conv2dt3(x)
        
        return x


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
        # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


    generator = Generator()
    noise = tf.random.normal((1, 100))
    generated_image = generator(noise)
    print(generated_image.shape)