import tensorflow as tf 
from tensorflow import keras 
from pathlib import Path 

def prepare_mnist(data_path, buffer_size, batch_size):
    (train_images, _), (_, _) = keras.datasets.mnist.load_data(data_path)
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5 

    return tf.data.Dataset.from_tensor_slices(train_images).shuffle(buffer_size).batch(batch_size)


DATASET_FUNCTION = {
    "mnist": prepare_mnist
}

def load_dataset(
    dataset: str = "mnist",
    dataset_save_path: Path = Path(__file__).parent.absolute(), 
    buffer_size: int = 60000,
    batch_size: int = 32
    ):
    train_dataset = DATASET_FUNCTION[dataset](
        data_path = dataset_save_path / "data" / f"{dataset}.npz", 
        buffer_size = buffer_size,
        batch_size = batch_size
    )

    return train_dataset
