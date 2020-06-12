import tensorflow as tf

from datetime import datetime
from pathlib import Path 
from matplotlib import pyplot as plt

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


def save_images(generated_images):
    gan_home_path = Path(__file__).parent.parent.absolute()
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    save_directory = gan_home_path / "images" / current_time
    if not save_directory.is_dir():
        save_directory.mkdir(parents=True)

    for i in range(generated_images.shape[0]):
        plt.imsave(save_directory / f"{i+1}.png", generated_images[i, :, :, 0], cmap='gray')