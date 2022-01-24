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
    batch_size, device = config['batch_size'], config['device']
    metrics = {
        'D_real_loss': [],
        'D_real_acc': [],
        'D_fake_loss': [],
        'D_fake_acc': [],
        'D_GP': [],
        'D_total_loss': [],
    }

    netG.to(device), netD.to(device)
    netG.eval(), netD.eval()

    # Eval discriminator
    for real in dataloader:
        real = real.to(device)

        with torch.no_grad():
            # On real images first
            predicted = netD(real)
            errD_reak = -torch.mean(predicted)
            metrics['D_real_loss'].append(errD_real.item())
            metrics['D_real_acc'].append(torch.sigmoid(predicted).mean().item())

            # On fake images then
            latents = netG.generate_z(batch_size).to(device)
            fake = netG(latents)
            predicted = netD(fake)
            errD_fake = torch.mean(predicted)
            metrics['D_fake_loss'].append(errD_fake.item())
            metrics['D_fake_acc'].append(1 - torch.sigmoid(predicted).mean().item())

        # Compute gradient penalty
        fake = fake[:len(real)]  # Match the batch size (in case there is too much fakes)
        errD_gradient = gradient_penalty(netD, real, fake, device)
        metrics['D_GP'].append(errD_gradient.item())

        # Final discriminator loss
        errD = errD_real + errD_fake + config['weight_GP'] * errD_gradient
        metrics['D_total_loss'].append(errD.item())

    for metric_name, values in metrics.items():
        metrics[metric_name] = np.mean(values)

    metrics['D_total_acc'] = (
        metrics['D_real_acc'] + metrics['D_fake_acc']
    ) / 2

    return metrics


def gradient_penalty(
        netD: Discriminator,
        real: torch.FloatTensor,
        fake: torch.FloatTensor,
        device: str,
    ):
    """Compute the gradient wrt to a weighted average of real and fake samples.

    Enforce the critic gradients to be close to 1 (1-Lipchitz).
    """
    alpha = torch.rand_like(real).to(device)
    interpolated = real * alpha + fake * (1 - alpha)
    interpolated = interpolated.requires_grad_(True)
    predicted = netD(interpolated)

    weight = torch.ones_like(predicted)
    gradient = torch.autograd.grad(
        outputs = predicted,
        inputs = interpolated,
        grad_outputs = weight,
        retain_graph = True,
        create_graph = True,
        only_inputs = True,
    )[0]  # Compute gradient w.r.t. interpolated

    gradient = gradient.view(gradient.shape[0], -1)
    return (gradient.norm(2, dim=1) - 1).pow(2).mean()


def train_critic(config: dict):
    """Train the discriminator for one epoch.
    """
    netG, netD = config['netG'], config['netD']
    train_loader, optimD = config['train_loader'], config['optimD']
    batch_size, device = config['batch_size'], config['device']

    for real in train_loader:
        optimD.zero_grad()
        real = real.to(device)

        # On real images first
        predicted = netD(real)
        errD_real = -torch.mean(predicted)

        # On fake images then
        latents = netG.generate_z(batch_size).to(device)
        fake = netG(latents).detach()
        predicted = netD(fake)
        errD_fake = torch.mean(predicted)

        # Compute gradient penalty
        fake = fake[:len(real)]  # Match the batch size (in case there is too much fakes)
        errD_gradient = gradient_penalty(netD, real, fake, device)

        # Final discriminator loss
        errD = errD_real + errD_fake + config['weight_GP'] * errD_gradient
        errD.backward()
        optimD.step()


def train_generator(config: dict):
    """Train the generator for one epoch.
    """
    netG, netD = config['netG'], config['netD']
    train_loader, optimG = config['train_loader'], config['optimG']
    batch_size, device = config['batch_size'], config['device']

    for _ in range(len(train_loader)):
        optimG.zero_grad()

        latents = netG.generate_z(batch_size).to(device)
        fake = netG(latents)
        predicted = netD(fake)
        errG = -torch.mean(predicted)  # We want G to fool D

        errG.backward()
        optimG.step()


def train(config: dict):
    """Training loop.
    WandB should be initialise as the results will be logged.

    Use label smoothing for stability.
    """
    netG, netD = config['netG'], config['netD']
    optimG, optimD = config['optimG'], config['optimD']
    train_loader, test_loader = config['train_loader'], config['test_loader']
    batch_size, device = config['batch_size'], config['device']
    dim_im = config['dim_image']

    torch.manual_seed(config['seed'])
    netG.to(device), netD.to(device)
    fixed_latent = netG.generate_z(64).to(device)

    for _ in tqdm(range(config['epochs'])):
        netG.train()
        netD.train()

        # Train discriminator
        for _ in range(config['iter_D']):
            train_critic(config)

        # Train generator
        train_generator(config)

        # Generate fake images and logs everything to WandB
        with torch.no_grad():
            fake = netG(fixed_latent).cpu()

        train_metrics = eval_loader(train_loader, config)
        test_metrics = eval_loader(test_loader, config)

        logs = dict()
        for group, metrics in [('Train', train_metrics), ('Test', test_metrics)]:
            for metric_name, value in metrics.items():
                logs[f'{group} - {metric_name}'] = value

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
        config['n_layers_z']
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
        'dim_image': 32,
        'batch_size': 128,
        'epochs': 10,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': 0,

        # StyleGAN params
        'n_channels': 128,
        'dim_z': 32,
        'n_layers_z': 3,
        'lr_g': 1e-5,
        'betas_g': (0.5, 0.99),

        # Discriminator params
        'n_first_channels': 8,
        'n_layers_d_block': 2,
        'dropout': 0.3,
        'lr_d': 1e-5,
        'betas_d': (0.5, 0.99),
        'iter_D': 4,
        'weight_GP': 3,
    }

    return config
