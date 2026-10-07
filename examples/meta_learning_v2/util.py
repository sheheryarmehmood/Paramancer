from __future__ import annotations

from torch import nn


# Helper Methods for Encoder

def conv_block(
    in_channels: int,
    out_channels: int,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        ),
        nn.GroupNorm(
            num_groups=1,
            num_channels=out_channels,
        ),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )

def conv_blocks(
    num_channels: int,
    num_blocks: int,
) -> list[nn.Sequential]:
    return [
        conv_block(num_channels, num_channels)
        for _ in range(num_blocks)
    ]


# Loss Function