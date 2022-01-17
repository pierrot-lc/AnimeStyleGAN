"""Training functions.
"""
import os

import wandb as wb
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.generator import StyleGAN
from src.discriminator import Discriminator
from src.data import load_dataset


def train(config: dict):
    """Training loop.
    WandB should be initialise as the results will be logged.
    """
    netG, netD = config['netG'], config['netD']
    optimG, optimD = config['optimG'], config['optimD']
    train_loader, test_loader = config['train_loader'], config['test_loader']
    batch_size, device = config['batch_size'], config['device']

    torch.manual_seed(config['seed'])
    netG.to(device), netD.to(device)
    fixed_latent = netG.generate_z(64).to(device)

    for _ in tqdm(range(config['epochs'])):
        netG.train()
        netD.train()

        # Train discriminator
        for real in train_loader:
            optimD.zero_grad()
            real = real.to(device)

            # On real images first
            predicted = netD(real)
            errD_real = -predicted.mean()

            # On fake images then
            with torch.no_grad():
                latents = netG.generate_z(batch_size).to(device)
                fake = netG(latents)
            predicted = netD(fake)
            errD_fake = predicted.mean()

            # Final discriminator loss
            errD = errD_real + errD_fake
            errD.backward()
            optimD.step()

        # Train generator
        for _ in range(len(train_loader)):
            optimG.zero_grad()

            latents = netG.generate_z(batch_size).to(device)
            fake = netG(latents)
            predicted = netD(fake)

            errG = -predicted.mean()  # We want G to fool D
            errG.backward()
            nn.utils.clip_grad_norm_(netG.parameters(), 0.1)
            optimG.step()

        # Generate fake images and logs everything to WandB
        with torch.no_grad():
            fake = netG(fixed_latent).cpu()

        wb.log({
            'Generated images': wb.Image(fake),
        })


def prepare_training(data_path: str, config: dict) -> dict:
    """Instanciate the models, the dataloaders and
    the optimizers.
    Return everything inside a dictionnary.
    """
    # Instanciate the models
    config['netG'] = StyleGAN(
        config['dim_image'],
        config['n_channels'],
        config['dim_z'],
        config['n_layers_z']
    )
    config['netD'] = Discriminator(
        config['dim_image'],
        config['n_first_channels']
    )

    # Optimizers
    config['optimG'] = optim.RMSprop(
        config['netG'].parameters(),
        lr=config['lr_g'],
    )
    config['optimD'] = optim.RMSprop(
        config['netD'].parameters(),
        lr=config['lr_d'],
    )

    # Datasets and dataloaders
    train_dataset = load_dataset(os.path.join(data_path, 'train'), config['dim_image'])
    test_dataset = load_dataset(os.path.join(data_path, 'test'), config['dim_image'])

    config['train_loader'] = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2,
    )
    config['test_loader'] = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2,
    )

    return config


def create_config() -> dict:
    """Return the basic parameters for training.
    """
    config = {
        # Global params
        'dim_image': 64,
        'batch_size': 128,
        'epochs': 10,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': 0,

        # StyleGAN params
        'n_channels': 128,
        'dim_z': 32,
        'n_layers_z': 3,
        'lr_g': 1e-4,

        # Discriminator params
        'n_first_channels': 4,
        'lr_d': 1e-3,
    }

    return config
