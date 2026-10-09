# * For constructing Dataset / Episodes
# * Creates support and query datasets
# * Should also return the mapping which converts between original labels
# and episode-local labels.

"""Episode dataset helpers for the meta-learning example."""

from __future__ import annotations
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .util import verify_dataset, build_class_index


class EpisodeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        class_to_indices: dict[int, list[int]],
        num_episodes: int,
        n_way: int = 5,
        n_support_per_class: int = 1,
        n_query_per_class: int = 2,
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.class_to_indices = class_to_indices
        self._n_way = n_way
        self.n_support_per_class = n_support_per_class
        self.n_query_per_class = n_query_per_class
        self.num_episodes = num_episodes
        self._classes = sorted(class_to_indices)
        self._rng = random.Random(seed)
        self._episodes = [
            self._sample_episode_indices()
            for _ in range(num_episodes)
        ]

    def __len__(self) -> int:
        return self.num_episodes

    def __getitem__(self, index: int):
        episode = self._episodes[index]
        
        support_images = torch.stack([
            self.dataset[sample_index][0]
            for sample_index in episode["support_indices"]
        ])
        
        query_images = torch.stack([
            self.dataset[sample_index][0]
            for sample_index in episode["query_indices"]
        ])
        
        return (
            index,
            support_images,
            episode["support_targets"],
            query_images,
            episode["query_targets"],
            torch.tensor(episode["local_to_global"], dtype=torch.long),
        )
    
    def _sample_episode_indices(self):
        chosen_classes = self._rng.sample(self._classes, self._n_way)
        support_indices = []
        support_labels = []
        query_indices = []
        query_labels = []
        
        for remapped_label, class_id in enumerate(chosen_classes):
            indices = self._rng.sample(
                self.class_to_indices[class_id],
                self.n_support_per_class + self.n_query_per_class
            )
            support_indices += indices[:self.n_support_per_class]
            support_labels += [remapped_label] * self.n_support_per_class
            query_indices += indices[self.n_support_per_class:]
            query_labels += [remapped_label] * self.n_query_per_class
        
        support_labels = torch.tensor(support_labels, dtype=torch.long)
        query_labels = torch.tensor(query_labels, dtype=torch.long)
        
        return {
            "local_to_global": chosen_classes,
            "support_indices": support_indices,
            "support_targets": support_labels,
            "query_indices": query_indices,
            "query_targets": query_labels,
        }


def main() -> None:
    DATA_ROOT = Path(__file__).resolve().parent / "data"

    base_dataset = datasets.Omniglot(
        root=DATA_ROOT,
        background=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
            ]
        ),
    )
    verify_dataset("Omniglot", base_dataset)
    
    batch_size = 2
    num_episodes = 32

    episode_dataset = EpisodeDataset(
        base_dataset,
        build_class_index(base_dataset),
        num_episodes=num_episodes,
    )
    
    loader_generator = torch.Generator().manual_seed(0)
    loader = DataLoader(
        episode_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    
    for episode_id, support_x, support_y, query_x, query_y, _ in loader:
        print(f"episode_id shape: {tuple(episode_id.shape)}")
        print(f"episode_id: {episode_id.tolist()}")
        print(f"support_x shape: {tuple(support_x.shape)}")
        print(f"support_y shape: {tuple(support_y.shape)}")
        print(f"query_x shape: {tuple(query_x.shape)}")
        print(f"query_y shape: {tuple(query_y.shape)}")
        print()


if __name__ == "__main__":
    main()