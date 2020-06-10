import tensorflow as tf 
from tensorflow import keras


class Discriminator(keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = keras.layers.Conv2D(64, (5, 5), strides=(2, 2), 
                                         padding="same", input_shape=(28, 28, 1), 
                                         name="Conv1"
                                        )
        self.lrelu1 = keras.layers.LeakyReLU(name="LeakyRelu1")
        self.dropout1 = keras.layers.Dropout(0.3, name="Dropout1")
        self.conv2 = keras.layers.Conv2D(128, (5, 5), strides=(2, 2), 
                                         padding="same",
                                         name="Conv2"
                                        )
        self.lrelu2 = keras.layers.LeakyReLU(name="LeakyRelu2")
        self.dropout2 = keras.layers.Dropout(0.3, name="Dropout2")
        self.flatten = keras.layers.Flatten(name="Flatten")
        self.dense = keras.layers.Dense(1, name="OutputDense")

        
    def call(self, x, training=False):
        x = self.lrelu1(self.conv1(x))
        if training:
            x = self.dropout1(x)
            
        x = self.lrelu2(self.conv2(x))
        if training:
            x = self.dropout2(x)
        
        x = self.flatten(x)
        return self.dense(x)
 