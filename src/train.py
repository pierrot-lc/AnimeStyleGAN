"""Training functions.
"""
import os

import numpy as np
import wandb as wb
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.generator import StyleGAN
from src.discriminator import Discriminator
from src.data import load_dataset


def eval_loader(dataloader: DataLoader, config: dict) -> dict:
    """Evaluate the models on the given dataloader.
    Return the evaluate metrics.
    """
    netG, netD = config['netG'], config['netD']
    device, batch_size = config['device'], config['batch_size']
    loss = nn.BCEWithLogitsLoss()

    metrics = {
        'D_real_loss': [],
        'D_fake_loss': [],
        'D_loss': [],
        'G_loss': [],
        'real_acc': [],
        'fake_acc': [],
    }

    netG.to(device), netD.to(device)
    netG.eval(), netD.eval()

    with torch.no_grad():
        # Eval discriminator
        for real in dataloader:
            real = real.to(device)
            b_size = real.shape[0]

            # On real images first
            predicted = netD(real)
            errD_real = loss(
                predicted,
                torch.ones_like(predicted).to(device),
            )
            metrics['D_real_loss'].append(errD_real.item())
            metrics['real_acc'].append(
                torch.sigmoid(predicted).mean().item()
            )

            # On fake images then
            latents = netG.generate_z(b_size).to(device)
            fake = netG(latents)
            predicted = netD(fake)
            errD_fake = loss(
                predicted,
                torch.zeros_like(predicted).to(device),
            )
            metrics['D_fake_loss'].append(errD_fake.item())
            metrics['fake_acc'].append(
                1 - torch.sigmoid(predicted).mean().item()
            )

            # Final discriminator loss
            errD = errD_real + errD_fake
            metrics['D_loss'].append(errD.item())

        for _ in range(len(dataloader)):
            latents = netG.generate_z(batch_size).to(device)
            fake = netG(latents)
            predicted = netD(fake)
            errG = loss(
                predicted,
                torch.ones_like(predicted).to(device),
            )  # We want G to fool D

            metrics['G_loss'].append(errG.item())

    for metric_name, values in metrics.items():
        metrics[metric_name] = np.mean(values)

    return metrics


def train_critic(config: dict):
    """Train the discriminator for one epoch.
    """
    netG, netD = config['netG'], config['netD']
    dataloader, optimD = config['dataloader'], config['optimD']
    device = config['device']
    loss = nn.BCEWithLogitsLoss()

    for real in dataloader:
        optimD.zero_grad()
        real = real.to(device)
        b_size = real.shape[0]

        # On real images first
        predicted = netD(real)
        labels = 1 - torch.rand_like(predicted, device=device) / 5
        errD_real = loss(
            predicted,
            labels
        )

        # On fake images then
        latents = netG.generate_z(b_size).to(device)
        fake = netG(latents).detach()
        predicted = netD(fake)
        labels = torch.rand_like(predicted, device=device) / 5
        errD_fake = loss(
            predicted,
            labels
        )

        # Final discriminator loss
        errD = errD_real + errD_fake
        errD.backward()
        optimD.step()


def train_generator(config: dict):
    """Train the generator for one epoch.
    """
    netG, netD = config['netG'], config['netD']
    dataloader, optimG = config['dataloader'], config['optimG']
    batch_size, device = config['batch_size'], config['device']
    loss = nn.BCEWithLogitsLoss()

    for _ in range(len(dataloader)):
        optimG.zero_grad()

        latents = netG.generate_z(batch_size).to(device)
        fake = netG(latents)
        predicted = netD(fake)
        errG = loss(
            predicted,
            torch.ones_like(predicted).to(device),
        )  # We want G to fool D

        errG.backward()
        optimG.step()


def train(config: dict):
    """Training loop.
    WandB should be initialise as the results will be logged.

    Use label smoothing for stability.
    """
    netG, netD = config['netG'], config['netD']
    dataloader = config['dataloader']
    batch_size, device = config['batch_size'], config['device']
    dim_im = config['dim_image']

    torch.manual_seed(config['seed'])
    netG.to(device), netD.to(device)
    fixed_latent = netG.generate_z(64).to(device)

    for _ in tqdm(range(config['epochs'])):
        netG.train()
        netD.train()

        # Train discriminator
        train_critic(config)

        # Train generator
        train_generator(config)

        # Generate fake images and logs everything to WandB
        with torch.no_grad():
            fake = netG(fixed_latent).cpu()

        logs = dict()
        metrics = eval_loader(dataloader, config)
        for metric_name, value in metrics.items():
            logs[f'{metric_name}'] = value

        logs['Generated images'] = wb.Image(fake)

        wb.log(logs)

        # Save models on disk
        torch.save(netG.state_dict(), 'models/netG.pth')
        torch.save(netD.state_dict(), 'models/netD.pth')

    # Save models in the WandB run
    for net_name in ['netG', 'netD']:
        net_artifact = wb.Artifact(net_name, type='model')
        net_artifact.add_file(os.path.join('models', net_name + '.pth'))
        wb.log_artifact(net_artifact)


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
        config['n_layers_z'],
        config['dropout'],
    )
    config['netD'] = Discriminator(
        config['dim_image'],
        config['n_first_channels'],
        config['n_layers_d_block'],
        config['dropout'],
    )

    # Optimizers
    config['optimG'] = optim.Adam(
        config['netG'].parameters(),
        lr=config['lr_g'],
        betas=config['betas_g'],
    )
    config['optimD'] = optim.Adam(
        config['netD'].parameters(),
        lr=config['lr_d'],
        betas=config['betas_d'],
    )

    # Dataset and dataloader
    dataset = load_dataset(data_path, config['dim_image'])
    config['dataloader'] = DataLoader(
        dataset,
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
        'dim_image': 32,
        'batch_size': 64,
        'epochs': 100,
        'dropout': 0.3,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': 0,

        # StyleGAN params
        'n_channels': 32,
        'dim_z': 100,
        'n_layers_z': 2,
        'lr_g': 1e-4,
        'betas_g': (0.5, 0.99),

        # Discriminator params
        'n_first_channels': 2,
        'n_layers_d_block': 2,
        'lr_d': 1e-3,
        'betas_d': (0.5, 0.99),
    }

    return config
