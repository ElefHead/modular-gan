import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras import Model

from gan.networks import GAN, Generator, Discriminator
from gan.datasets import load_dataset

from .utils import setup

from pathlib import Path 
from datetime import datetime


def loss_fn(labels, output):
    return keras.losses.BinaryCrossentropy(from_logits=True)(labels, output)

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
    gan = GAN(discriminator, generator, latent_dim)
    gan.compile(discriminator_optimizer, generator_optimizer, loss_fn)

    gan_home_path = Path(__file__).parent.parent.absolute()
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")

    callbacks = []
    tensorboard_callback = TensorBoard(log_dir=str(gan_home_path / "logs" / current_time))

    callbacks.append(tensorboard_callback)

    if save_models:
        checkpoint_filepath = str( gan_home_path / "checkpoints" / f"{dataset}{epochs}_{current_time}" / "checkpoint") 
        model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath)
        callbacks.append(model_checkpoint_callback)
    
    _history  = gan.fit(train_dataset, epochs=epochs, callbacks=callbacks)

    generator.summary()
    discriminator.summary()

    if save_models:
        generator.save(gan_home_path / "saved_models" / current_time / f"generator_{dataset}{epochs}", save_format="tf")
        discriminator.save(gan_home_path / "saved_models" / current_time / f"discriminator_{dataset}{epochs}", save_format="tf")

    return generator, discriminator
