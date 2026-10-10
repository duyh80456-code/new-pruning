"""CIFAR-100 transforms for the ViT runs (data_transforms: data.cifar100_vit)

ViTs trained from scratch on CIFAR need stronger augmentation than the
ResNets' crop and flip. This adds the DeiT recipe's image-level parts:
RandAugment and random erasing. Mixup and cutmix act on the batch, in
train.py (utils/mixing.py). The dataset and loader stay data.cifar100.
"""
from torchvision import transforms

from data.cifar100 import MEAN, STD
from utils.config import FLAGS


def data_transforms():
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(
            num_ops=getattr(FLAGS, 'randaug_ops', 2),
            magnitude=getattr(FLAGS, 'randaug_magnitude', 9)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
        transforms.RandomErasing(p=getattr(FLAGS, 'random_erasing', 0.25)),
    ])
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    return train_transforms, val_transforms, val_transforms
