from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


class IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        return image, label, index


def _datasets(config: dict, seed: int) -> tuple[Dataset, Dataset]:
    data_cfg = config["dataset"]
    name = data_cfg["name"].lower()
    if name == "cifar100":
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        val_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )
        root = Path(data_cfg["root"])
        train_set = datasets.CIFAR100(
            root=root, train=True, transform=train_transform, download=data_cfg.get("download", False)
        )
        val_set = datasets.CIFAR100(
            root=root, train=False, transform=val_transform, download=data_cfg.get("download", False)
        )
    elif name == "fake":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)]
        )
        train_set = datasets.FakeData(
            size=int(data_cfg.get("train_size", 128)),
            image_size=(3, 32, 32),
            num_classes=int(data_cfg["num_classes"]),
            transform=transform,
            random_offset=seed * 100_000,
        )
        val_set = datasets.FakeData(
            size=int(data_cfg.get("val_size", 64)),
            image_size=(3, 32, 32),
            num_classes=int(data_cfg["num_classes"]),
            transform=transform,
            random_offset=seed * 100_000 + 50_000,
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return IndexedDataset(train_set), IndexedDataset(val_set)


def build_loaders(config: dict, seed: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_set, val_set = _datasets(config, seed)
    data_cfg = config["dataset"]
    workers = int(data_cfg.get("num_workers", 0))
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_set,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    subset_size = min(int(data_cfg["feature_subset_size"]), len(val_set))
    # One fixed deterministic ordering is reused at every budget.
    indices = torch.randperm(len(val_set), generator=torch.Generator().manual_seed(seed)).tolist()
    feature_loader = DataLoader(
        Subset(val_set, indices[:subset_size]),
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, feature_loader
