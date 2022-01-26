"""Script that launch the training of a model.
"""
import sys

import torch
import wandb as wb
from torchinfo import summary

from src.train import create_config, prepare_training, train


config = create_config()
config = prepare_training('./data/', config)

print('Generator model:')
summary(config['netG'], input_size=(config['batch_size'], config['dim_z']))

print('\n\nDiscriminator model:')
summary(config['netD'], input_size=(config['batch_size'], 3, config['dim_image'], config['dim_image']))

params = [
    'batch_size',
    'dim_image',
    'epochs',
    'dropout',
    'lr_g',
    'betas_g',
    'lr_d',
    'betas_d',
    'device',
]
print('\n\nTraining details:')
for param in params:
    expand = 20
    if 'lr' in param:
        print(f'     [{param}]\t-\t{config[param]:.1e}'.expandtabs(expand))
    else:
        print(f'     [{param}]\t-\t{config[param]}'.expandtabs(expand))


print(f'\nContinue with training for {config["epochs"]} epochs?')
if input('[y/n]> ') != 'y':
    sys.exit(0)

with wb.init(
    entity='pierrotlc',
    group=f'BCE - {config["dim_image"]}x{config["dim_image"]}',
    project='AnimeStyleGAN',
    # project='test',
    config=config,
    save_code=True,
):
    train(config)
