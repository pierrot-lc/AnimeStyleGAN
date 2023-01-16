from torch import optim

from .discriminator import Discriminator
from .generator import StyleGAN


class Trainer:
    data_cfg: dict
    training_cfg: dict
    generator_cfg: dict
    discriminator_cfg: dict

    def __init__(self, config: dict):
        self.__dict__ |= config

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
