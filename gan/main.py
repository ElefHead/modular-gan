from argparse import ArgumentParser
from typing import Dict
from pathlib import Path 
from json import loads, dumps

from gan.jobs import train_model, generate_images, save_images

from matplotlib import pyplot as plt

"""
DEFAULT_TRAIN_ARGS = {
    'batch_size': 32,
    'epochs': 2,
    'dataset': "mnist",
    'latent_dim': 100,
    'buffer_size': 60000
}

DEFAULT_TEST_ARGS = {
    "latent_dim": 100,
    "generator_path": None,
    "num_images": 20
}
"""

def _parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--mode",
        default="train",
        type=str,
        help="If set to train, training job will be run"
    )

    parser.add_argument(
        "--save", 
        default=True, 
        action='store_true',
        help="If set to true, model and intermediate checkpoints will be saved"
    )
    
    parser.add_argument(
        "experiment_config",
        type=str,
        help="Experiment JSON ('{\"dataset\": \"MNIST\", \"model\": \"model_name\", \"generator_network\": \"cnn\"}'"
    )

    args = parser.parse_args()
    return args


def run_training(config: Dict, save_weights: bool = True):
    train_model(**config)

def run_evaluation(config: Dict, save: bool = True):
    gen_images = generate_images(**config)
    if save:
        save_images(gen_images)
    

if __name__ == "__main__":
    args = _parse_args()

    config = loads(args.experiment_config)
    if args.mode == "train":
        run_training(config, args.save)
    elif args.mode == "evaluate":
        run_evaluation(config, args.save)

