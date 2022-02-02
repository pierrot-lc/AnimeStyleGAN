"""StyleGAN small implementation.

Paper: https://arxiv.org/abs/1812.04948v3
"""
import random

import numpy as np
import einops
from torchinfo import summary

import torch
import torch.nn as nn


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
                Shape of [n_styles, batch_size, dim_z].

        Return
        ------
            w: Style of a batch of images.
                Shape of [n_styles, batch_size, dim_z].
        """
        z = self.norm(z)
        for layer in self.layers:
            z = z + layer(z)
        w = self.out(z)
        return w

    def generate_z(self, batch_size: int, n_styles: int) -> torch.FloatTensor:
        return torch.randn(size=(n_styles, batch_size, self.dim_z))


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
                Shape of [batch_size, 2 * n_channels].

        Return
        ------
            x: Batch of images with style y.
                Shape of [batch_size, n_channels, dim_height, dim_width].
        """
        n_channels = x.shape[1]
        eps = 1e-9  # For numerical stability

        mean_x = torch.mean(x, dim=[2, 3], keepdims=True)
        std_x = torch.std(x, dim=[2, 3], keepdims=True) + eps
        x = (x - mean_x) / std_x

        y_s = einops.rearrange(y[:, :n_channels], 'b c -> b c () ()')
        y_b = einops.rearrange(y[:, n_channels:], 'b c -> b c () ()')

        x = y_s * x + y_b
        return x


class SynthesisBlock(nn.Module):
    """Upsample and then apply style vectors and convolutions.
    Reduce the number of filters.

    Parameters
    ----------
        dim:            Dimension size of the input (width/height).
        n_channels:     Number of channels of the output.
        n_layers:       Number of layers in this block.
        dropout:        Prob of the dropout layers.
        n_noise:        Number of filters in the noisy inputs.
        dim_style:      Dimension size of the style input.
        first_block:    Whether or not this block is the first block.
            If this block is the first one, there will be no upsampling,
            which means that the number of channels will be the same
            in input and output.
    """
    def __init__(
            self,
            dim: int,
            n_channels: int,
            n_layers: int,
            dropout: float,
            n_noise: int,
            dim_style: int,
            first_block: bool = False,
        ):
        super().__init__()
        self.dim = dim
        self.n_noise = n_noise
        self.n_channels = n_channels
        self.first_block = first_block

        if not first_block:  # Upsample and reducing channels.
            self.upsample = nn.ConvTranspose2d(2 * n_channels, n_channels, 4, 2, 1)
            self.conv = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Conv2d(n_channels, n_channels, 3, 1, 1),
                nn.BatchNorm2d(n_channels),
                nn.LeakyReLU(),
            )

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Conv2d(n_channels, n_channels, 3, 1, 1),
                nn.BatchNorm2d(n_channels),
                nn.LeakyReLU(),
            )
            for _ in range(n_layers)
        ])
        self.ada_in = AdaIN()

        self.A1 = nn.Linear(dim_style, 2 * n_channels)
        self.A2 = nn.Linear(dim_style, 2 * n_channels)
        self.B1 = nn.Conv2d(n_noise, n_channels, 3, 1, 1)
        self.B2 = nn.Conv2d(n_noise, n_channels, 3, 1, 1)

    def forward(
            self,
            x: torch.FloatTensor,
            w: torch.FloatTensor,
            n1: torch.FloatTensor,
            n2: torch.FloatTensor,
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
                Shape of [batch_size, n_channels // 2, dim // 2, dim // 2],
                except if first block: [batch_size, n_channels, dim, dim].
            w:  Batch of style vectors.
                Shape of [batch_size, dim_z].
            n1: Batch of random noise.
                Shape of [batch_size, n_channels, dim, dim],
                except if first block: [batch_size, n_channels, dim, dim].
            n2: Batch of random noise.
                Shape of [batch_size, n_channels, dim, dim],
                except if first block: [batch_size, n_channels, dim, dim].

        Return
        ------
            x: Batch of enhanced images.
                Shape of [batch_size, n_channels, dim, dim],
                except if first block: [batch_size, n_channels, dim, dim].
        """
        if not self.first_block:
            # x is of shape [batch_size, n_channels, dim // 2, dim // 2].
            x = self.upsample(x)
            x = self.conv(x)

        # Here x is of shape [batch_size, n_channels (// 2), dim, dim].
        y1 = self.A1(w)
        n1 = self.B1(n1)
        x = x + n1
        x = self.ada_in(x, y1)

        for conv_block in self.layers:
            x = x + conv_block(x)

        y2 = self.A2(w)
        n2 = self.B2(n2)
        x = x + n2
        x = self.ada_in(x, y2)
        return x

    def compute_noise(self, batch_size: int) -> torch.FloatTensor:
        """Return a random noise of the good shape
        for the forward of this module.
        """
        return torch.randn(
            size=(batch_size, self.n_noise, self.dim, self.dim)
        )


class SynthesisNetwork(nn.Module):
    """Stack of synthesis blocks.
    """
    def __init__(
        self,
        dim_final: int,
        n_channels: int,
        n_layers_block: int,
        dropout: float,
        n_noise: int,
        dim_style: int,
    ):
        super().__init__()
        INIT_DIM = 2
        self.n_blocks = int(np.log2(dim_final) - np.log2(INIT_DIM)) + 1

        self.learned_cnst = nn.Parameter(
            torch.randn(size=(n_channels, INIT_DIM, INIT_DIM)),
            requires_grad=True,
        )

        self.blocks = nn.ModuleList([
            SynthesisBlock(
                INIT_DIM << block_id,
                n_channels >> block_id,
                n_layers_block,
                dropout,
                n_noise,
                dim_style,
                first_block = block_id == 0
            )
            for block_id in range(self.n_blocks)
        ])

        self.to_rgb = nn.Conv2d(
            in_channels = n_channels >> (self.n_blocks - 1),
            out_channels = 3,
            kernel_size = 1,
        )

    def forward(self, w: torch.FloatTensor) -> torch.FloatTensor:
        """Generate a batch of images with the given styles.
        Generate the noises on the fly.

        Args
        ----
            w: Batch of style vectors.
                Shape of [n_blocks, batch_size, dim_z]

        Return
        ------
            x: Batch of images.
                Shape of [batch_size, 3, dim_final, dim_final].
        """
        batch_size = w.shape[1]
        device = w.device

        x = self.learned_cnst.to(device)
        x = einops.repeat(x, 'c w h -> b c w h', b=batch_size)

        for block, style in zip(self.blocks, w):
            n1 = block.compute_noise(batch_size).to(device)
            n2 = block.compute_noise(batch_size).to(device)
            x = block(x, style, n1, n2)

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
            n_layers_block: int,
            dropout: float,
            n_noise: int,
        ):
        super().__init__()

        self.mapping = MappingNetwork(dim_z, n_layers_z)
        self.synthesis = SynthesisNetwork(
            dim_final,
            n_channels,
            n_layers_block,
            dropout,
            n_noise,
            dim_z,
        )

    def forward(self, z: torch.FloatTensor) -> torch.FloatTensor:
        """Take the latent vectors and produce images.

        Args
        ----
            z: Batch of latent vectors (randomly drawn).
                Shape of [n_styles, batch_size, dim_z]

        Return
        ------
            x: Batch of generated images.
                Shape of [batch_size, 3, dim_final, dim_final].
        """
        w = self.mapping(z)
        w = self.style_mixing(w)
        x = self.synthesis(w)
        return x

    def generate_z(self, batch_size: int, n_styles: int = 1, device='cpu') -> torch.FloatTensor:
        return self.mapping.generate_z(batch_size, n_styles).to(device)

    def style_mixing(self, w: torch.FloatTensor) -> torch.FloatTensor:
        """Randomly select at which layer will be used the multiple style vectors.

        Args
        ----
            w: Batch of style vectors.
                Shape of [n_styles, batch_size, dim_z]

        Return
        ------
            w: Batch of style vectors.
                Shape of [n_blocks, batch_size, dim_z]
        """
        n_blocks = self.synthesis.n_blocks
        n_cuts = len(w) - 1

        cuts = random.sample(range(n_blocks-1), k=n_cuts)
        cuts = list(sorted(cuts)) + [n_blocks]

        indices, curr_i = [], 0
        for i in range(n_blocks):
            indices.append(curr_i)

            if i == cuts[curr_i]:
                curr_i += 1

        w = torch.stack([w[i] for i in indices])
        return w
