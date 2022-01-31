"""StyleGAN small implementation.

Paper: https://arxiv.org/abs/1812.04948v3
"""
import numpy as np

import torch
import torch.nn as nn

import einops
from torchinfo import summary


class MappingNetwork(nn.Module):
    """Network mapping the latent space to the style space of the images.
    """
    def __init__(self, dim_z: int, n_layers: int):
        super().__init__()
        self.dim_z = dim_z

        self.norm = nn.LayerNorm(dim_z)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim_z, dim_z),
                nn.LayerNorm(dim_z),
                nn.LeakyReLU(),
            )
            for _ in range(n_layers)
        ])

        self.out = nn.Linear(dim_z, dim_z)

    def forward(self, z: torch.FloatTensor) -> torch.FloatTensor:
        """Compute the style given the latent z.

        Args
        ----
            z: Latent vectors, randomly drawn.
                Shape of [batch_size, dim_z].

        Return
        ------
            w: Style of a batch of images.
                Shape of [batch_size, dim_z].
        """
        z = self.norm(z)
        for layer in self.layers:
            z = z + layer(z)
        w = self.out(z)
        return w

    def generate_z(self, batch_size: int) -> torch.FloatTensor:
        return torch.randn(size=(batch_size, self.dim_z))


class AdaIN(nn.Module):
    """Apply the style vectors to a batch of images.

    Thanks to https://github.com/Maggiking/AdaIN-Style-Transfer-PyTorch/blob/master/AdaIN.py
    """
    def __init__(self):
        super().__init__()

    def forward(
            self,
            x: torch.FloatTensor,
            y: torch.FloatTensor
        ) -> torch.FloatTensor:
        """Apply the style y to the images x.

        Args
        ----
            x: Batch of images.
                Shape of [batch_size, n_channels, dim_height, dim_width].
            y: Batch of styles.
                Shape of [batch_size, dim_z].

        Return
        ------
            x: Batch of images with style y.
                Shape of [batch_size, n_channels, dim_height, dim_width].
        """
        eps = 1e-9  # For numerical stability

        mean_x = torch.mean(x, dim=[2, 3], keepdims=True)
        std_x = torch.std(x, dim=[2, 3], keepdims=True) + eps

        mean_y = torch.mean(y, dim=1)
        std_y = torch.std(y, dim=1) + eps

        mean_y = einops.repeat(mean_y, 'b -> b c h w', c=x.shape[1], h=1, w=1)
        std_y = einops.repeat(std_y, 'b -> b c h w', c=x.shape[1], h=1, w=1)

        x = std_y * (x - mean_x) / std_x + mean_y
        return x


class SynthesisBlock(nn.Module):
    """Upsample and then apply style vectors and convolutions.
    Reduce the number of filters.
    """
    def __init__(
            self,
            dim: int,
            n_channels: int,
            dropout: float,
            first_block: bool = False,
        ):
        super().__init__()
        self.dim = dim
        self.n_channels = n_channels
        self.first_block = first_block

        if not first_block:  # Upsample and reducing channels.
            self.upsample = nn.ConvTranspose2d(2 * n_channels, n_channels, 4, 2, 1)
            self.conv1 = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Conv2d(n_channels, n_channels, 3, 1, 1),
                nn.LeakyReLU(),
            )

        self.conv2 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv2d(n_channels, n_channels, 3, 1, 1),
            nn.LeakyReLU(),
        )
        self.ada_in = AdaIN()

    def forward(
            self,
            x: torch.FloatTensor,
            A: torch.FloatTensor,
            B1: torch.FloatTensor,
            B2: torch.FloatTensor,
        ) -> torch.FloatTensor:
        """Upsample and then pass the image through
        convolutions, AdaIN and some random noise.

        If this module is the first block of the network,
        it will not upsample the input nor reduce the number
        of channels.

        Args
        ----
            x:  Batch of images. Should be a constant for
            the first block.
                Shape of [batch_size, n_channels, dim // 2, dim // 2],
                except if first block: [batch_size, n_channels, dim, dim].
            A:  Batch of style vectors.
                Shape of [batch_size, dim_z].
            B1: Batch of random noise.
                Shape of [batch_size, n_channels // 2, dim, dim],
                except if first block: [batch_size, n_channels, dim, dim].
            B2: Batch of random noise.
                Shape of [batch_size, n_channels // 2, dim, dim],
                except if first block: [batch_size, n_channels, dim, dim].

        Return
        ------
            x: Batch of enhanced images.
                Shape of [batch_size, n_channels // 2, dim, dim],
                except if first block: [batch_size, n_channels, dim, dim].
        """
        if not self.first_block:
            # x is of shape [batch_size, n_channels, dim // 2, dim // 2].
            x = self.upsample(x)
            x = self.conv1(x)

        # Here x is of shape [batch_size, n_channels (// 2), dim, dim].
        x = x + B1
        x = self.ada_in(x, A)

        x = self.conv2(x)

        x = x + B2
        x = self.ada_in(x, A)
        return x

    def compute_noise(self, batch_size: int) -> torch.FloatTensor:
        """Return a random noise of the good shape
        for the forward of this module.
        """
        return torch.randn(
            size=(batch_size, self.n_channels, self.dim, self.dim)
        ) / 100


class SynthesisNetwork(nn.Module):
    """Stack of synthesis blocks.
    """
    def __init__(self, dim_final: int, n_channels: int, dropout: float):
        super().__init__()
        INIT_DIM = 2
        n_blocks = int(np.log2(dim_final) - np.log2(INIT_DIM)) + 1

        self.learned_cnst = nn.Parameter(
            torch.randn(size=(n_channels, INIT_DIM, INIT_DIM)),
            requires_grad=True,
        )

        self.blocks = nn.ModuleList([
            SynthesisBlock(
                INIT_DIM << block_id,
                n_channels >> block_id,
                dropout,
                first_block = block_id == 0
            )
            for block_id in range(n_blocks)
        ])

        self.to_rgb = nn.Conv2d(
            in_channels = n_channels >> (n_blocks - 1),
            out_channels = 3,
            kernel_size = 1,
        )

    def forward(self, A: torch.FloatTensor) -> torch.FloatTensor:
        """Generate a batch of images with the given styles.
        Generate the noises on the fly.

        Args
        ----
            A: Batch of style vectors.
                Shape of [batch_size, dim_z].

        Return
        ------
            x: Batch of images.
                Shape of [batch_size, 3, dim_final, dim_final].
        """
        batch_size = A.shape[0]

        x = self.learned_cnst.to(A.device)
        x = einops.repeat(x, 'c w h -> b c w h', b=batch_size)
        for block in self.blocks:
            B1 = block.compute_noise(batch_size).to(A.device)
            B2 = block.compute_noise(batch_size).to(A.device)
            x = block(x, A, B1, B2)
        x = self.to_rgb(x)
        return torch.tanh(x)


class StyleGAN(nn.Module):
    """The main generator module.
    Use the synthesis network coupled with the mapping module
    to create new images.
    """
    def __init__(
            self,
            dim_final: int,
            n_channels: int,
            dim_z: int,
            n_layers_z: int,
            dropout: float,
        ):
        super().__init__()

        self.mapping = MappingNetwork(dim_z, n_layers_z)
        self.synthesis = SynthesisNetwork(dim_final, n_channels, dropout)

    def forward(self, z: torch.FloatTensor) -> torch.FloatTensor:
        """Take the latent vectors and produce images.

        Args
        ----
            z: Batch of latent vectors (randomly drawn).
                Shape of [batch_size, dim_z].

        Return
        ------
            x: Batch of generated images.
                Shape of [batch_size, 3, dim_final, dim_final].
        """
        z = self.mapping(z)
        x = self.synthesis(z)
        return x

    def generate_z(self, batch_size: int) -> torch.FloatTensor:
        return self.mapping.generate_z(batch_size)
