"""Basic discriminator.
"""
import numpy as np

import torch
import torch.nn as nn

from torchinfo import summary


class DiscriminatorBlock(nn.Module):
    """Two layers of convolution and one layer of downsampling.
    """
    def __init__(self, n_channels: int, n_filters: int, dropout: float):
        super().__init__()

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(p=dropout),
                nn.utils.spectral_norm(
                    nn.Conv2d(n_channels, n_channels, 3, 1, 1, bias=False)
                ),  # Spectral norm for stability training
                nn.BatchNorm2d(n_channels),
                nn.LeakyReLU(),
            )
            for _ in range(n_filters)
        ])
        self.downsample = nn.Conv2d(n_channels, 2 * n_channels, 4, 2, 1)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Apply the convolution layers and the downsampling.

        Args
        ----
            x: Batch of image features.
                Shape of [batch_size, n_channels, width, height].

        Return
        ------
            x: Batch of images features.
                Shape of [batch_size, 2 * n_channels, width // 2, height // 2].
        """
        for conv in self.convs:
            x = x + conv(x)
        x = self.downsample(x)
        return x


class Discriminator(nn.Module):
    """Basic discriminator implementation.
    """
    def __init__(self,
            dim: int,
            n_first_channels: int,
            n_layers_block: int,
            dropout: float,
        ):
        super().__init__()
        n_blocks = int(np.log2(dim))

        self.first_conv = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.utils.spectral_norm(
                    nn.Conv2d(3, n_first_channels, 3, 1, 1)
                ),
        )

        self.blocks = nn.ModuleList([
            DiscriminatorBlock(n_first_channels << block_id, n_layers_block, dropout)
            for block_id in range(n_blocks)
        ])

        self.classify = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(n_first_channels << n_blocks, 1, 3, 1, 1, bias=False),
            ),
            nn.Flatten(),
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Predict if images are fake or not.
        The output is of a linear layer, so we can leverage BCEWithLogitsLoss
        numerical stability.

        Args
        ----
            x: Batch of images.
                Shape of [batch_size, 3, dim, dim].

        Return
        ------
            y: Batch of predictions (fake or real).
                Shape of [batch_size,]
        """
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)

        y = self.classify(x)
        return y


if __name__ == '__main__':
    config = {
        'dim': 64,
        'n_first_channels': 4,
        'n_layers_block': 3,
    }

    model = Discriminator(**config)
    summary(model, input_size=([128, 3, 64, 64]))
