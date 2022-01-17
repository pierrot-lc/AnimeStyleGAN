"""Script that launch the training of a model.
"""
import sys

import torch
import wandb as wb
from torchinfo import summary

from src.train import create_config, prepare_training, train


config = create_config()
config = {
    # Global params
    'dim_image': 32,
    'batch_size': 128,
    'epochs': 10,
    'lr': 1e-4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    # StyleGAN params
    'n_channels': 128,
    'dim_z': 32,
    'n_layers_z': 3,

    # Discriminator params
    'n_first_channels': 4,
}
config = prepare_training('./data/', config)

print('Generator model:')
summary(config['netG'], input_size=(config['batch_size'], config['dim_z']))

print('\n\nDiscriminator model:')
summary(config['netD'], input_size=(config['batch_size'], 3, config['dim_image'], config['dim_image']))

print(f'\n\nContinue with training for {config["epochs"]} epochs?')
if input('[y/n]> ') != 'y':
    sys.exit(0)

with wb.init(project='test', config=config, entity='pierrotlc', save_code=True):
    train(config)
