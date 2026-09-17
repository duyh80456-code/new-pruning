from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import numpy as np
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


@dataclass(frozen=True)
class ConfirmatoryLoaders:
    train: DataLoader
    validation: DataLoader
    test: DataLoader
    calibration: DataLoader
    geometry: DataLoader


@dataclass(frozen=True)
class InterimValidationLoaders:
    """Train-derived loaders that never construct or read the CIFAR-100 test split."""

    validation: DataLoader
    calibration: DataLoader
    geometry: DataLoader


@dataclass(frozen=True)
class PolicyGeometryLoaders:
    """Training-derived deterministic loaders used only to construct online policies."""

    calibration: DataLoader
    geometry: DataLoader


@dataclass(frozen=True)
class DevelopmentTrainLoaders:
    """Leakage-safe training loaders that never instantiate the CIFAR test split."""

    train: DataLoader
    validation: DataLoader
    calibration: DataLoader


def _cifar100_transforms():
    mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    augmented = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    deterministic = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )
    return augmented, deterministic


def _datasets(config: dict, seed: int) -> tuple[Dataset, Dataset]:
    data_cfg = config["dataset"]
    name = data_cfg["name"].lower()
    if name == "cifar100":
        train_transform, val_transform = _cifar100_transforms()
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


def _stratified_cifar_split(targets: list[int], validation_size: int, seed: int):
    targets_array = np.asarray(targets)
    classes = np.unique(targets_array)
    if validation_size % len(classes):
        raise ValueError("validation_size must be divisible by the number of classes")
    per_class = validation_size // len(classes)
    rng = np.random.default_rng(seed)
    validation_indices = []
    train_indices = []
    for class_id in classes:
        indices = np.flatnonzero(targets_array == class_id)
        rng.shuffle(indices)
        validation_indices.extend(indices[:per_class].tolist())
        train_indices.extend(indices[per_class:].tolist())
    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    return train_indices, validation_indices


def build_confirmatory_loaders(config: dict, training_seed: int) -> ConfirmatoryLoaders:
    """Build leakage-safe CIFAR loaders for checkpoint selection and final testing."""
    data_cfg = config["dataset"]
    if data_cfg["name"].lower() != "cifar100" or data_cfg.get("fake_data", False):
        raise ValueError("Confirmatory loaders require real CIFAR-100")
    augmented, deterministic = _cifar100_transforms()
    root = Path(data_cfg["root"])
    download = data_cfg.get("download", False)
    raw_train = datasets.CIFAR100(root=root, train=True, transform=None, download=download)
    train_augmented = IndexedDataset(
        datasets.CIFAR100(root=root, train=True, transform=augmented, download=False)
    )
    train_deterministic = IndexedDataset(
        datasets.CIFAR100(root=root, train=True, transform=deterministic, download=False)
    )
    test = IndexedDataset(
        datasets.CIFAR100(root=root, train=False, transform=deterministic, download=download)
    )
    split_seed = int(data_cfg["split_seed"])
    validation_size = int(data_cfg["validation_size"])
    train_indices, validation_indices = _stratified_cifar_split(
        raw_train.targets, validation_size, split_seed
    )
    calibration_size = int(data_cfg["bn_calibration_size"])
    geometry_size = int(data_cfg["feature_subset_size"])
    if calibration_size > len(train_indices) or geometry_size > len(validation_indices):
        raise ValueError("Calibration/geometry subset exceeds its source split")
    # These slices are fixed by split_seed and therefore identical across model seeds.
    calibration_indices = train_indices[:calibration_size]
    geometry_indices = validation_indices[:geometry_size]
    workers = int(data_cfg.get("num_workers", 0))
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        Subset(train_augmented, train_indices),
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        generator=torch.Generator().manual_seed(training_seed),
    )
    evaluation_batch_size = int(config["evaluation"]["batch_size"])
    validation_loader = DataLoader(
        Subset(train_deterministic, validation_indices),
        batch_size=evaluation_batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test,
        batch_size=evaluation_batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
    )
    calibration_loader = DataLoader(
        Subset(train_deterministic, calibration_indices),
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
    )
    geometry_loader = DataLoader(
        Subset(train_deterministic, geometry_indices),
        batch_size=evaluation_batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
    )
    return ConfirmatoryLoaders(
        train=train_loader,
        validation=validation_loader,
        test=test_loader,
        calibration=calibration_loader,
        geometry=geometry_loader,
    )


def build_development_train_loaders(config: dict, training_seed: int) -> DevelopmentTrainLoaders:
    """Build augmented train plus fixed validation/BN loaders without opening test data."""
    data_cfg = config["dataset"]
    if data_cfg["name"].lower() != "cifar100" or data_cfg.get("fake_data", False):
        raise ValueError("Development training loaders require real CIFAR-100")
    augmented, deterministic = _cifar100_transforms()
    root = Path(data_cfg["root"])
    download = data_cfg.get("download", False)
    raw_train = datasets.CIFAR100(root=root, train=True, transform=None, download=download)
    train_augmented = IndexedDataset(
        datasets.CIFAR100(root=root, train=True, transform=augmented, download=False)
    )
    train_deterministic = IndexedDataset(
        datasets.CIFAR100(root=root, train=True, transform=deterministic, download=False)
    )
    train_indices, validation_indices = _stratified_cifar_split(
        raw_train.targets, int(data_cfg["validation_size"]), int(data_cfg["split_seed"])
    )
    calibration_size = int(data_cfg["bn_calibration_size"])
    if calibration_size > len(train_indices):
        raise ValueError("BN calibration subset exceeds the training split")
    workers = int(data_cfg.get("num_workers", 0))
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        Subset(train_augmented, train_indices),
        batch_size=int(config["training"]["batch_size"]), shuffle=True,
        num_workers=workers, pin_memory=pin_memory,
        generator=torch.Generator().manual_seed(int(training_seed)),
    )
    validation_loader = DataLoader(
        Subset(train_deterministic, validation_indices),
        batch_size=int(config["evaluation"]["batch_size"]), shuffle=False,
        num_workers=workers, pin_memory=pin_memory,
    )
    calibration_loader = DataLoader(
        Subset(train_deterministic, train_indices[:calibration_size]),
        batch_size=int(config["training"]["batch_size"]), shuffle=False,
        num_workers=workers, pin_memory=pin_memory,
    )
    return DevelopmentTrainLoaders(
        train=train_loader, validation=validation_loader, calibration=calibration_loader
    )


def build_interim_validation_loaders(config: dict) -> InterimValidationLoaders:
    """Build fixed validation diagnostics while keeping the test split sealed."""
    data_cfg = config["dataset"]
    if data_cfg["name"].lower() != "cifar100" or data_cfg.get("fake_data", False):
        raise ValueError("Interim validation loaders require real CIFAR-100")
    _, deterministic = _cifar100_transforms()
    root = Path(data_cfg["root"])
    download = data_cfg.get("download", False)
    raw_train = datasets.CIFAR100(root=root, train=True, transform=None, download=download)
    deterministic_train = IndexedDataset(
        datasets.CIFAR100(root=root, train=True, transform=deterministic, download=False)
    )
    train_indices, validation_indices = _stratified_cifar_split(
        raw_train.targets, int(data_cfg["validation_size"]), int(data_cfg["split_seed"])
    )
    calibration_size = int(data_cfg["bn_calibration_size"])
    geometry_size = int(data_cfg["feature_subset_size"])
    if calibration_size > len(train_indices) or geometry_size > len(validation_indices):
        raise ValueError("Calibration/geometry subset exceeds its source split")
    workers = int(data_cfg.get("num_workers", 0))
    pin_memory = torch.cuda.is_available()
    evaluation_batch_size = int(config["evaluation"]["batch_size"])
    return InterimValidationLoaders(
        validation=DataLoader(
            Subset(deterministic_train, validation_indices),
            batch_size=evaluation_batch_size, shuffle=False,
            num_workers=workers, pin_memory=pin_memory,
        ),
        calibration=DataLoader(
            Subset(deterministic_train, train_indices[:calibration_size]),
            batch_size=int(config["training"]["batch_size"]), shuffle=False,
            num_workers=workers, pin_memory=pin_memory,
        ),
        geometry=DataLoader(
            Subset(deterministic_train, validation_indices[:geometry_size]),
            batch_size=evaluation_batch_size, shuffle=False,
            num_workers=workers, pin_memory=pin_memory,
        ),
    )


def build_policy_geometry_loaders(config: dict) -> PolicyGeometryLoaders:
    """Build leakage-safe online-policy geometry from the 45k training split.

    BN calibration and policy geometry are fixed, deterministic, disjoint
    slices of the training split.  Neither slice intersects validation.
    """
    data_cfg = config["dataset"]
    if data_cfg["name"].lower() != "cifar100" or data_cfg.get("fake_data", False):
        raise ValueError("Policy geometry loaders require real CIFAR-100")
    _, deterministic = _cifar100_transforms()
    root = Path(data_cfg["root"])
    download = data_cfg.get("download", False)
    raw_train = datasets.CIFAR100(root=root, train=True, transform=None, download=download)
    deterministic_train = IndexedDataset(
        datasets.CIFAR100(root=root, train=True, transform=deterministic, download=False)
    )
    train_indices, validation_indices = _stratified_cifar_split(
        raw_train.targets, int(data_cfg["validation_size"]), int(data_cfg["split_seed"])
    )
    calibration_size = int(data_cfg["bn_calibration_size"])
    geometry_size = int(data_cfg["feature_subset_size"])
    if calibration_size + geometry_size > len(train_indices):
        raise ValueError("BN calibration plus policy geometry exceeds the training split")
    calibration_indices = train_indices[:calibration_size]
    geometry_indices = train_indices[calibration_size:calibration_size + geometry_size]
    calibration_set, geometry_set, validation_set = map(
        set, (calibration_indices, geometry_indices, validation_indices)
    )
    if calibration_set & geometry_set:
        raise RuntimeError("Policy geometry overlaps BN calibration")
    if calibration_set & validation_set or geometry_set & validation_set:
        raise RuntimeError("Online-policy inputs overlap validation")
    workers = int(data_cfg.get("num_workers", 0))
    pin_memory = torch.cuda.is_available()
    return PolicyGeometryLoaders(
        calibration=DataLoader(
            Subset(deterministic_train, calibration_indices),
            batch_size=int(config["training"]["batch_size"]), shuffle=False,
            num_workers=workers, pin_memory=pin_memory,
        ),
        geometry=DataLoader(
            Subset(deterministic_train, geometry_indices),
            batch_size=int(config["evaluation"]["batch_size"]), shuffle=False,
            num_workers=workers, pin_memory=pin_memory,
        ),
    )
