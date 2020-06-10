import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras import Model

from gan.networks import GAN, Generator, Discriminator
from gan.datasets import load_dataset

from pathlib import Path 
from datetime import datetime

def loss_fn(labels, output):
    return keras.losses.BinaryCrossentropy(from_logits=True)(labels, output)

def setup():
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    tf.debugging.set_log_device_placement(False)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

def train_model(
    dataset: str,
    epochs: int, 
    buffer_size: int,
    batch_size: int, 
    latent_dim: int,
    data_save_path: str = None,
    save_models: bool = True
    ) -> (Model, Model) :

    setup()
    
    if not data_save_path:
        train_dataset = load_dataset(dataset=dataset, buffer_size=buffer_size, batch_size=batch_size)
    else:
        train_dataset = load_dataset(dataset=dataset, dataset_save_path=Path(data_save_path), buffer_size=buffer_size, batch_size=batch_size)

    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.05)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.05)

    discriminator = Discriminator()
    generator = Generator()
    gan = GAN(discriminator, generator)
    gan.compile(discriminator_optimizer, generator_optimizer, loss_fn, latent_dim)

    gan_home_path = Path(__file__).parent.parent.absolute()

    if save_models:
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")
        checkpoint_filepath = str( gan_home_path / "checkpoints" / current_time / "checkpoint_") 
        model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath)
        _history  = gan.fit(train_dataset, epochs=epochs, callbacks=[model_checkpoint_callback])
    else: 
        _history  = gan.fit(train_dataset, epochs=epochs)

    generator.summary()
    discriminator.summary()

    if save_models:
        generator.save(gan_home_path / "saved_models" / current_time / "generator", save_format="tf")
        discriminator.save(gan_home_path / "saved_models" / current_time / "discriminator", save_format="tf")

    return generator, discriminator
