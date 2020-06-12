import tensorflow as tf 
from tensorflow import keras

from gan.datasets import load_dataset
from pathlib import Path 

from datetime import datetime
from matplotlib import pyplot as plt

from .utils import setup

def generate_images(generator_path: str, num_images: int, latent_dim: int) :

    setup()

    print(Path(generator_path).resolve())

    generator = keras.models.load_model(generator_path)
    random_latent_vectors = tf.random.normal(shape=(num_images, latent_dim))
    generated_images = generator(random_latent_vectors)
    return generated_images