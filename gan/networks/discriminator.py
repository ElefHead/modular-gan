import tensorflow as tf 
from tensorflow import keras


class Discriminator(keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = keras.layers.Conv2D(64, (5, 5), strides=(2, 2), 
                                         padding="same", input_shape=(28, 28, 1)
                                        )
        self.conv2 = keras.layers.Conv2D(128, (5, 5), strides=(2, 2), 
                                         padding="same")
        self.dropout = keras.layers.Dropout(0.3)
        self.dense = keras.layers.Dense(1)
        self.lrelu = keras.layers.LeakyReLU()
        self.flatten = keras.layers.Flatten()
        
    def call(self, x, training=True):
        x = self.lrelu(self.conv1(x))
        if training:
            x = self.dropout(x)
            
        x = self.lrelu(self.conv2(x))
        if training:
            x = self.dropout(x)
        
        x = self.flatten(x)
        return self.dense(x)
 