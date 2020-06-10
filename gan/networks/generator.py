import tensorflow as tf 
from tensorflow import keras


class Generator(keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.dense = keras.layers.Dense(7*7*256, use_bias=False, input_shape=(100,))
        self.conv2dt1 = keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding="same", use_bias=False)
        self.conv2dt2 = keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same", use_bias=False)
        self.conv2dt3 = keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding="same", use_bias=False, activation="tanh")

        self.bn1 = keras.layers.BatchNormalization()
        self.bn2 = keras.layers.BatchNormalization()
        self.bn3 = keras.layers.BatchNormalization()

        self.rshpe_lyr = keras.layers.Reshape((7, 7, 256))
        self.lrelu = keras.layers.LeakyReLU()
        
    def call(self, x):
        x = self.dense(x)
        x = self.bn1(x)
        x = self.lrelu(x)
        
        x = self.rshpe_lyr(x)

        x = self.conv2dt1(x)
        x = self.bn2(x)
        x = self.lrelu(x)
        
        x = self.conv2dt2(x)
        x = self.bn3(x)
        x = self.lrelu(x)
        
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