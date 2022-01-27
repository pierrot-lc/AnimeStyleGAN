"""Training functions.
"""
import os
from collections import defaultdict

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


def eval_critic_batch(
        real: torch.FloatTensor,
        config: dict
    ) -> dict:
    """Eval the critic on one batch of data.
    Return the metrics.
    """
    netG, netD = config['netG'], config['netD']
    device = config['device']
    loss = config['loss']

    real = real.to(device)
    b_size = real.shape[0]

    metrics = dict()

    # On real images first
    predicted = netD(real)
    labels = 1 - torch.rand_like(predicted, device=device) / 5
    errD_real = loss(
        predicted,
        labels
    )

    metrics['D_real_loss'] = errD_real
    metrics['real_acc'] = torch.sigmoid(predicted).mean()


    # On fake images then
    latents = netG.generate_z(b_size).to(device)
    fake = netG(latents).detach()
    predicted = netD(fake)
    labels = torch.rand_like(predicted, device=device) / 5
    errD_fake = loss(
        predicted,
        labels
    )

    metrics['D_fake_loss'] = errD_fake
    metrics['fake_acc'] = 1 - torch.sigmoid(predicted).mean()


    # Final discriminator loss
    errD = errD_real + errD_fake
    metrics['D_loss'] = errD

    return metrics


def eval_generator_batch(
        config: dict,
    ) -> dict:
    """Evaluate the generator for one batch.
    Returns the metrics.
    """
    netG, netD = config['netG'], config['netD']
    batch_size, device = config['batch_size'], config['device']
    loss = config['loss']

    metrics = dict()

    latents = netG.generate_z(batch_size).to(device)
    fake = netG(latents)
    predicted = netD(fake)
    errG = loss(
        predicted,
        torch.ones_like(predicted).to(device),
    )  # We want G to fool D

    metrics['G_loss'] = errG

    return metrics


def eval_loader(dataloader: DataLoader, config: dict) -> dict:
    """Evaluate the models on the given dataloader.
    Return the evaluate metrics.
    """
    netG, netD = config['netG'], config['netD']
    device, batch_size = config['device'], config['batch_size']
    loss = nn.BCEWithLogitsLoss()

    metrics = defaultdict(list)

    netG.to(device), netD.to(device)
    netG.eval(), netD.eval()

    with torch.no_grad():
        for real in dataloader:
            # Eval discriminator
            metrics_batch = eval_critic_batch(real, config)
            for m_name, m_value in metrics_batch.items():
                metrics[m_name].append(m_value.cpu().item())

            # Eval generator
            metrics_batch = eval_generator_batch(config)
            for m_name, m_value in metrics_batch.items():
                metrics[m_name].append(m_value.cpu().item())

    for metric_name, values in metrics.items():
        metrics[metric_name] = np.mean(values)

    return metrics


def train(config: dict):
    """Training loop.
    WandB should be initialise as the results will be logged.

    Use label smoothing for stability.
    """
    netG, netD = config['netG'], config['netD']
    optimD, optimG = config['optimD'], config['optimG']
    dataloader = config['dataloader']
    batch_size, device = config['batch_size'], config['device']
    dim_im = config['dim_image']

    torch.manual_seed(config['seed'])
    netG.to(device), netD.to(device)
    fixed_latent = netG.generate_z(64).to(device)
    config['loss'] = nn.BCEWithLogitsLoss()

    for _ in tqdm(range(config['epochs'])):
        netG.train()
        netD.train()

        for real in dataloader:
            # Train discriminator
            optimD.zero_grad()
            metrics = eval_critic_batch(real, config)

            loss = metrics['D_loss']
            loss.backward()
            optimD.step()

            # Train generator
            optimG.zero_grad()
            metrics = eval_generator_batch(config)

            loss = metrics['G_loss']
            loss.backward()
            optimG.step()

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
        'dim_z': 32,
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
