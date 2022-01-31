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
summary(config['netG'], input_size=(config['netG'].generate_z(1).shape))

print('\n\nDiscriminator model:')
summary(config['netD'], input_size=(config['batch_size'], 3, config['dim_image'], config['dim_image']))

params = [
    'batch_size',
    'dim_image',
    'epochs',
    'dropout',
    None,
    'lr_g',
    'betas_g',
    'weight_decay_g',
    'milestones_g',
    'gamma_g',
    'running_avg_factor_G',
    'weight_avg_factor_g',
    None,
    'lr_d',
    'betas_d',
    'weight_decay_d',
    'milestones_d',
    'gamma_d',
    'weight_fake_loss',
    'running_avg_factor_D',
    'weight_avg_factor_d',
    None,
    'device',
]
print('\n\nTraining details:')
for param in params:
    if param is None:
        print('', end='\n')
        continue

    expand = 25
    param_exp = f'[{param}]\t'.expandtabs(expand)
    if 'lr' in param:
        print(f'     {param_exp}-\t\t{config[param]:.1e}')
    else:
        print(f'     {param_exp}-\t\t{config[param]}')


print(f'\nContinue with training for {config["epochs"]} epochs?')
if input('[y/n]> ') != 'y':
    sys.exit(0)

with wb.init(
    entity='pierrotlc',
    group=f'ADAIN - {config["dim_image"]}x{config["dim_image"]}',
    project='AnimeStyleGAN',
    config=config,
    save_code=True,
):
    train(config)
