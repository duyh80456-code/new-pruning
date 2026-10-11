"""Tiny ImageNet transforms for the ViT runs (data_transforms:
data.tinyimagenet_vit)

The same recipe as data.cifar100_vit at 64 pixels: the ResNets' crop and
flip (data.tinyimagenet) plus RandAugment and random erasing. Mixup and
cutmix act on the batch, in train.py (utils/mixing.py). The dataset and
loader stay data.tinyimagenet.
"""
from torchvision import transforms

from data.tinyimagenet import MEAN, STD
from utils.config import FLAGS


def data_transforms():
    train_transforms = transforms.Compose([
        transforms.RandomCrop(64, padding=8),
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
