from argparse import ArgumentParser
from typing import Dict
from pathlib import Path 
from json import loads, dumps

from gan.jobs import train_model

DEFAULT_TRAIN_ARGS = {
    'batch_size': 32,
    'epochs': 450,
    'dataset': "mnist",
    'latent_dim': 100,
    'buffer_size': 60000
}

def _parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--mode",
        default="train",
        type=str,
        help="If set to train, training job will be run"
    )
    
    parser.add_argument(
        "--experimental-config",
        default=dumps(DEFAULT_TRAIN_ARGS),
        type=str,
        help="Experiment JSON ('{\"dataset\": \"MNIST\", \"model\": \"model_name\", \"generator_network\": \"cnn\"}'"
    )

    parser.add_argument(
        "--save", 
        default=True, 
        type=bool, 
        help="If set to true, model and intermediate checkpoints will be saved"
    )

    args = parser.parse_args()
    return args


def run_training(config: Dict = DEFAULT_TRAIN_ARGS, save_weights: bool = True):
    train_model(**config)

if __name__ == "__main__":
    args = _parse_args()

    config = loads(args.experimental_config)
    if args.mode == "train":
        run_training(config, args.save)

