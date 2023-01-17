import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, RandomSampler
from torchinfo import summary
from tqdm import tqdm

import wandb

from .data import load_dataset
from .discriminator import Discriminator
from .generator import StyleGAN


class Trainer:
    data_cfg: dict
    training_cfg: dict
    generator_cfg: dict
    discriminator_cfg: dict

    def __init__(self, config: dict):
        self.__dict__ |= config
        self.config = config  # For WandB.

        # Models.
        self.netG = StyleGAN(
            self.data_cfg["dim_image"],
            self.generator_cfg["n_channels"],
            self.generator_cfg["dim_z"],
            self.generator_cfg["n_layers_z"],
            self.generator_cfg["n_layers_block"],
            self.generator_cfg["dropout"],
            self.generator_cfg["n_noise"],
        )
        self.netD = Discriminator(
            self.data_cfg["dim_image"],
            self.discriminator_cfg["n_channels"],
            self.discriminator_cfg["n_layers_d_block"],
            self.discriminator_cfg["dropout"],
        )

        # Optimizers
        self.optimG = optim.AdamW(
            self.netG.parameters(),
            lr=self.generator_cfg["lr"],
            betas=self.generator_cfg["betas"],
        )
        self.optimD = optim.AdamW(
            self.netD.parameters(),
            lr=self.discriminator_cfg["lr"],
            betas=self.discriminator_cfg["betas"],
        )

        # Loss
        self.loss = nn.BCEWithLogitsLoss()

        # Dataset & dataloader.
        dataset = load_dataset(Path(self.data_cfg["path"]), self.data_cfg["dim_image"])
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.training_cfg["batch_size"],
            sampler=RandomSampler(
                torch.arange(len(dataset)),
                replacement=True,
                num_samples=self.training_cfg["epoch_size"],
            ),
            num_workers=4,
        )

    def summary(self):
        latents = self.netG.generate_z(64)
        print("Generator:")
        summary(self.netG, input_data=latents)

        print("\nDiscriminator:")
        summary(
            self.netD,
            input_size=(3, self.data_cfg["dim_image"], self.data_cfg["dim_image"]),
        )

    def batch_critic(
        self, real_images: torch.Tensor, fake_images: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Evaluates the critic model on the batch of fake and real images.

        Args:
            real_images: Real images.
                Shape of [batch_size, n_channels, im_size, im_size].
            fake_images: Fake images.
                Shape of [batch_size, n_channels, im_size, im_size].

        Returns:
            The metrics computed on the given batch.
        """
        metrics = dict()

        # On real images first.
        predicted = self.netD(real_images)  # TODO: Check if noise is useful.
        labels = torch.ones_like(
            predicted, device=predicted.device
        )  # TODO: check if label smoothing is useful.
        loss_real = self.loss(predicted, labels)
        metrics["D_loss-real"] = loss_real
        metrics["D_accuracy-real"] = torch.sigmoid(predicted).mean()

        # On fake images then.
        predicted = self.netD(fake_images)
        labels = torch.zeros_like(predicted, device=predicted.device)
        loss_fake = self.loss(predicted, labels)
        metrics["D_loss-fake"] = loss_fake
        metrics["D_accuracy-fake"] = 1 - torch.sigmoid(predicted).mean()

        # Final loss.
        metrics["D_loss"] = (loss_real + loss_fake) / 2

        return metrics

    def batch_generator(
        self,
        fake_images: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Evaluates the generator model on the batch of fake and real images.

        Args:
            fake_images: Fake images from the generator.
                Shape of [batch_size, n_channels, im_size, im_size].

        Returns:
            The metrics computed on the given batch.
        """
        metrics = dict()

        predicted = self.netD(fake_images)
        labels = torch.ones_like(predicted, device=predicted.device)
        metrics["G_loss"] = self.loss(predicted, labels)

        return metrics

    def do_one_epoch(self, train: bool) -> dict[str, float]:
        device = self.training_cfg["device"]
        logs = defaultdict(list)
        self.netG.train()
        self.netD.train()

        for real_images in tqdm(self.dataloader):
            real_images = real_images.to(device)
            b_size = real_images.shape[0]

            # Train discriminator first.
            self.optimD.zero_grad()
            fake_images = self.netG.generate(
                b_size, n_styles=random.randint(1, 2), device=device
            )
            metrics = self.batch_critic(real_images, fake_images)

            if train:
                metrics["D_loss"].backward()
                self.optimD.step()

            for m_name, m_value in metrics.items():
                logs[m_name].append(m_value.cpu().item())

            # Train generator then.
            self.optimG.zero_grad()
            fake_images = self.netG.generate(
                b_size, n_styles=random.randint(1, 2), device=device
            )
            metrics = self.batch_generator(fake_images)

            if train:
                metrics["G_loss"].backward()
                self.optimG.step()

            for m_name, m_value in metrics.items():
                logs[m_name].append(m_value.cpu().item())

        logs = {
            m_name: sum(m_values) / len(m_values) for m_name, m_values in logs.items()
        }
        return logs

    def train(self):
        torch.manual_seed(self.training_cfg["seed"])
        random.seed(self.training_cfg["seed"])
        device = self.training_cfg["device"]
        fixed_latent = self.netG.generate_z(64, device=device)

        self.netG.to(device)
        self.netD.to(device)

        with wandb.init(
            entity="pierrotlc",
            group="test",
            project="AnimeStyleGAN",
            config=self.config,
            save_code=False,
        ):
            for _ in tqdm(range(self.training_cfg["epochs"])):
                logs = self.do_one_epoch(train=True)

                with torch.no_grad():
                    fake_images = self.netG(fixed_latent).cpu()

                logs["Generated images"] = wandb.Image(fake_images)
                wandb.log(logs)
