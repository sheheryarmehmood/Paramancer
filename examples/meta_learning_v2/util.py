from __future__ import annotations

from torch import nn
from collections import defaultdict


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


# Helper methods for EpisodeDataset check

def _get_label(dataset, index: int) -> int:
    if hasattr(dataset, "_flat_character_images"):
        _, class_idx = dataset._flat_character_images[index]
        return int(class_idx)

    if hasattr(dataset, "targets"):
        return int(dataset.targets[index])

    _, label = dataset[index]
    return int(label)


def build_class_index(dataset) -> dict[int, list[int]]:
    class_to_indices: dict[int, list[int]] = defaultdict(list)
    for index in range(len(dataset)):
        class_to_indices[_get_label(dataset, index)].append(index)
    return dict(class_to_indices)


def verify_dataset(name: str, dataset) -> None:
    sample_x, _ = dataset[0]
    print(f"{name} dataset size: {len(dataset)}")
    print(f"{name} sample shape: {tuple(sample_x.shape)}")