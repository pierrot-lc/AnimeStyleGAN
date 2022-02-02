"""Training functions.
"""
import os
import random
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


def running_average_loss(model: nn.Module, running_avg: list) -> torch.FloatTensor:
    """Compute the running average loss of the model parameters.
    """
    running_avg_loss = [
        (p - r).pow(2).mean()
        for p, r in zip(model.parameters(), running_avg)
    ]
    running_avg_loss = sum(running_avg_loss) / len(running_avg_loss)
    return running_avg_loss


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
    running_avg = config['running_avg_D']
    running_avg_factor = config['running_avg_factor_D']

    real = real.to(device)
    b_size = real.shape[0]

    metrics = dict()

    # Running avg loss
    metrics['running_avg_loss_D'] = running_average_loss(netD, running_avg)

    # On real images first
    predicted = netD(real + torch.randn_like(real, device=device) / 100)
    labels = 1 - torch.rand_like(predicted, device=device) / 10  # Label smoothing
    errD_real = loss(
        predicted,
        labels
    )

    metrics['D_real_loss'] = errD_real
    metrics['real_acc'] = torch.sigmoid(predicted).mean()


    # On fake images then
    latents = netG.generate_z(b_size, n_styles=random.randint(1, 2), device=device)
    fake = netG(latents).detach()
    predicted = netD(fake + torch.randn_like(fake, device=device) / 100)
    errD_fake = loss(
        predicted,
        torch.zeros_like(predicted, device=device),
    )

    metrics['D_fake_loss'] = errD_fake
    metrics['fake_acc'] = 1 - torch.sigmoid(predicted).mean()


    # Final discriminator loss
    errD = (errD_real + config['weight_fake_loss'] * errD_fake) / 2
    metrics['D_loss'] = errD + config['weight_avg_factor_d'] * metrics['running_avg_loss_D']

    running_avg = [
        running_avg_factor * r + (1 - running_avg_factor) * p.detach()
        for r, p in zip(running_avg, netD.parameters())
    ]
    config['running_avg_D'] = running_avg

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
    running_avg = config['running_avg_G']
    running_avg_factor = config['running_avg_factor_G']

    metrics = dict()

    # Running avg loss
    metrics['running_avg_loss_G'] = running_average_loss(netG, running_avg)

    # Generator loss
    latents = netG.generate_z(batch_size, n_styles=random.randint(1, 2), device=device)
    fake = netG(latents)
    predicted = netD(fake + torch.randn_like(fake, device=device) / 100)
    errG = loss(
        predicted,
        torch.ones_like(predicted).to(device),
    )  # We want G to fool D

    metrics['G_fake_loss'] = errG

    metrics['G_loss'] = errG + config['weight_avg_factor_g'] * metrics['running_avg_loss_G']

    # Update running average of the parameters
    running_avg = [
        running_avg_factor * r + (1 - running_avg_factor) * p.detach()
        for r, p in zip(running_avg, netG.parameters())
    ]
    config['running_avg_G'] = running_avg

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
    stepD, stepG = config['stepD'], config['stepG']
    dataloader = config['dataloader']
    batch_size, device = config['batch_size'], config['device']
    dim_im = config['dim_image']

    torch.manual_seed(config['seed'])
    random.seed(config['seed'])
    netG.to(device), netD.to(device)
    fixed_latent = netG.generate_z(64, device=device)

    config['running_avg_G'] = [p.detach() for p in netG.parameters()]
    config['running_avg_D'] = [p.detach() for p in netD.parameters()]

    for _ in tqdm(range(config['epochs'])):
        netG.train()
        netD.train()

        n_iter = 0
        logs = defaultdict(list)
        for real in dataloader:
            # Train discriminator
            optimD.zero_grad()
            metrics = eval_critic_batch(real, config)
            for m, v in metrics.items():
                logs[m].append(v.item())

            loss = metrics['D_loss']
            loss.backward()
            optimD.step()

            # Train generator
            optimG.zero_grad()
            metrics = eval_generator_batch(config)
            for m, v in metrics.items():
                logs[m].append(v.item())

            loss = metrics['G_loss']
            loss.backward()
            optimG.step()

            n_iter += 1
            if n_iter < config['n_iter_log']:
                continue

            for m, v in logs.items():
                logs[m] = np.mean(v)

            # Generate fake images and logs everything to WandB
            with torch.no_grad():
                fake = netG(fixed_latent).cpu()

            logs['Generated images'] = wb.Image(fake)
            wb.log(logs)

            n_iter = 0
            logs = defaultdict(list)


        stepD.step()
        stepG.step()

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
        config['n_layers_block'],
        config['dropout'],
        config['n_noise'],
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
        weight_decay=config['weight_decay_g'],
    )
    config['optimD'] = optim.Adam(
        config['netD'].parameters(),
        lr=config['lr_d'],
        betas=config['betas_d'],
        weight_decay=config['weight_decay_d'],
    )

    # Schedulers
    config['stepG'] = optim.lr_scheduler.MultiStepLR(
        config['optimG'],
        milestones=config['milestones_g'],
        gamma=config['gamma_g'],
    )
    config['stepD'] = optim.lr_scheduler.MultiStepLR(
        config['optimD'],
        milestones=config['milestones_d'],
        gamma=config['gamma_d'],
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
        'dim_image': 64,
        'batch_size': 256,
        'epochs': 50,
        'dropout': 0.3,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': 0,
        'n_iter_log': 10,

        # StyleGAN params
        'n_channels': 512,
        'dim_z': 196,
        'n_layers_z': 4,
        'n_layers_block': 3,
        'n_noise': 10,
        'lr_g': 1e-4,
        'betas_g': (0.5, 0.5),
        'weight_decay_g': 0,
        'milestones_g': [15],
        'gamma_g': 0.1,
        'running_avg_factor_G': 0.9,
        'weight_avg_factor_g': 0.5,

        # Discriminator params
        'n_first_channels': 12,
        'n_layers_d_block': 5,
        'lr_d': 1e-4,
        'betas_d': (0.5, 0.99),
        'weight_decay_d': 0,
        'milestones_d': [15],
        'gamma_d': 0.1,
        'weight_fake_loss': 1,
        'running_avg_factor_D': 0.9,
        'weight_avg_factor_d': 0.5,
    }

    return config
