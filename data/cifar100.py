"""CIFAR-100 pipeline

train.py looks up data_transforms, dataset and data_loader by module name
when the built-in names do not match, so setting all three to data.cifar100
in the yml routes them here.

The loader fallback in train.py returns early, before it records
data_size_train, so this module sets it itself. The linear and cosine
schedules divide by it.
"""
import os

import torch
from torchvision import datasets, transforms

from utils.config import FLAGS
from utils.distributed import get_world_size


MEAN = [0.5071, 0.4865, 0.4409]
STD = [0.2673, 0.2564, 0.2762]


def data_transforms():
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    return train_transforms, val_transforms, val_transforms


def dataset(train_transforms, val_transforms, test_transforms):
    root = getattr(FLAGS, 'dataset_dir', 'data')
    # torchvision unpacks into cifar-100-python; download only when it is
    # not already there, so a Kaggle run with no internet still works if the
    # folder was attached as an input.
    download = not os.path.isdir(os.path.join(root, 'cifar-100-python'))
    if download:
        os.makedirs(root, exist_ok=True)
    if not FLAGS.test_only:
        train_set = datasets.CIFAR100(
            root=root, train=True, download=download,
            transform=train_transforms)
    else:
        train_set = None
    val_set = datasets.CIFAR100(
        root=root, train=False, download=download, transform=val_transforms)
    return train_set, val_set, None


def data_loader(train_set, val_set, test_set):
    if getattr(FLAGS, 'batch_size', False):
        if getattr(FLAGS, 'batch_size_per_gpu', False):
            assert FLAGS.batch_size == (
                FLAGS.batch_size_per_gpu * FLAGS.num_gpus_per_job)
        else:
            assert FLAGS.batch_size % FLAGS.num_gpus_per_job == 0
            FLAGS.batch_size_per_gpu = (
                FLAGS.batch_size // FLAGS.num_gpus_per_job)
    elif getattr(FLAGS, 'batch_size_per_gpu', False):
        FLAGS.batch_size = FLAGS.batch_size_per_gpu * FLAGS.num_gpus_per_job
    else:
        raise ValueError('batch size (per gpu) is not defined')
    batch_size = int(FLAGS.batch_size / get_world_size())
    workers = getattr(FLAGS, 'data_loader_workers', 2)

    train_loader = None
    if not FLAGS.test_only:
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=workers,
            drop_last=getattr(FLAGS, 'drop_last', False))
        FLAGS.data_size_train = len(train_set)
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=workers,
        drop_last=False)
    FLAGS.data_size_val = len(val_set)
    return train_loader, val_loader, val_loader
