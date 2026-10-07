from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from .util import conv_block, conv_blocks


class ConvEncoder(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            in_channels: int,
            hidden_dim: int,
            num_blocks: int,
        ) -> None:
        super().__init__()
        if num_blocks < 1:
            raise ValueError("num_blocks must be at least 1.")
        
        self.features = nn.Sequential(
            conv_block(in_channels, hidden_dim),
            *conv_blocks(hidden_dim, num_blocks - 1),
        )
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.projection = nn.Linear(hidden_dim, embedding_dim)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images.flatten(0, 1))
        features = self.pool(features)
        features = features.flatten(start_dim=1)
        embeddings = self.projection(features)

        return F.normalize(
            embeddings,
            p=2,
            dim=-1,
            eps=1e-12,
        ).unflatten(0, images.shape[:2])

class Conv2OmniEncoder(ConvEncoder):
    """Two-block ConvNet encoder for standard image batches."""

    def __init__(
        self,
        embedding_dim: int,
        in_channels: int = 1,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__(
            embedding_dim=embedding_dim,
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_blocks=2,
        )


class Conv4CIFAREncoder(ConvEncoder):
    """Four-block ConvNet encoder for standard image batches."""

    def __init__(
        self,
        embedding_dim: int,
        in_channels: int = 3,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__(
            embedding_dim=embedding_dim,
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_blocks=4,
        )