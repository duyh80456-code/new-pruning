"""Tiny ImageNet pipeline (200 classes, 100,000 train and 10,000 val images,
64x64)

Set dataset, data_transforms and data_loader to data.tinyimagenet in the
yml, as data.cifar100 is set for CIFAR-100. The official val split, labelled
by val_annotations.txt, is the test set; the unlabelled test split is not
used, as everywhere this set is reported.

The images are JPEGs, 64x64. Decoding 100,000 of them every epoch would
keep the loader busier than the GPU on a two-core Kaggle machine, so they
are decoded once into uint8 arrays and cached next to the data (1.2 GB for
train). The cache is written by whichever process gets there first and read
by the rest.

Where the data comes from, first hit wins: dataset_dir/tiny-imagenet-200,
any tiny-imagenet-200 under /kaggle/input (an attached dataset), or the
official zip from Stanford, downloaded into dataset_dir.
"""
import hashlib
import os
import zipfile

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from utils.config import FLAGS
from data.cifar100 import data_loader  # noqa: F401  same loader, same sizes

URL = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
FOLDER = 'tiny-imagenet-200'
# where Kaggle mounts attached datasets
INPUTS = '/kaggle/input'
# per-channel statistics of the 100,000 train images
MEAN = [0.4802, 0.4481, 0.3975]
STD = [0.2770, 0.2691, 0.2821]


def data_transforms():
    # the CIFAR recipe at twice the size: pad an eighth of the side and crop
    train_transforms = transforms.Compose([
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    return train_transforms, val_transforms, val_transforms


def is_tiny(folder):
    """the official layout, whatever the folder is called"""
    return (os.path.isfile(os.path.join(folder, 'wnids.txt')) and
            os.path.isfile(os.path.join(folder, 'val', 'val_annotations.txt'))
            and os.path.isdir(os.path.join(folder, 'train')))


# md5 of the official release's 200 class ids, sorted, and of its 10,000
# val labels as sorted "file<TAB>class" lines, read off the Stanford zip
# (md5 90528d7ca1a48142e341f4ef8d21d0de). Order and line endings do not
# enter, so a re-zipped Kaggle copy passes; other classes or relabelled
# val images do not.
WNIDS_MD5 = '62ec2c49c5abdc833b8ea5082c79667d'
VAL_MD5 = '10ec1929ae7b5715fce6b8e1d04e1b39'


def check_official(folder):
    """raise unless folder holds the official classes and val labels"""
    with open(os.path.join(folder, 'wnids.txt')) as handle:
        wnids = sorted(line.strip() for line in handle if line.strip())
    with open(os.path.join(folder, 'val', 'val_annotations.txt')) as handle:
        val = sorted('\t'.join(line.split('\t')[:2])
                     for line in handle if line.strip())
    found = (hashlib.md5('\n'.join(wnids).encode()).hexdigest(),
             hashlib.md5('\n'.join(val).encode()).hexdigest())
    if found != (WNIDS_MD5, VAL_MD5):
        raise ValueError(
            '{} is not the official Tiny ImageNet: class ids {}, val labels '
            '{} (expected {}, {})'.format(
                folder, found[0], found[1], WNIDS_MD5, VAL_MD5))
    print('Tiny ImageNet at {}: official classes and val labels'.format(
        folder), flush=True)


def locate(root):
    """the tiny-imagenet-200 folder, downloading it if nothing has it

    Kaggle copies of the set name their folders differently, so an
    attached input is recognised by its layout, not its name.
    """
    here = os.path.join(root, FOLDER)
    if is_tiny(here):
        return here
    # An unpacked copy is used where it lies. A copy uploaded as the zip
    # mounts as one file rather than 120,000, which Kaggle has failed to
    # mount at least once; it is unpacked into root like the download.
    archives = []
    for top, dirs, files in os.walk(INPUTS):
        if 'wnids.txt' in files and is_tiny(top):
            return top
        archives += [os.path.join(top, f) for f in files
                     if f.lower().endswith('.zip')]
        # do not descend into the 200 class folders of a train split
        dirs[:] = [d for d in dirs if not d.startswith('n0')
                   and not d.startswith('n1') and d != 'images']
    os.makedirs(root, exist_ok=True)
    archive = os.path.join(root, FOLDER + '.zip')
    for candidate in archives:
        with zipfile.ZipFile(candidate) as handle:
            if FOLDER + '/wnids.txt' in handle.namelist():
                print('unpacking Tiny ImageNet from', candidate, flush=True)
                handle.extractall(root)
                return here
    if not os.path.isfile(archive):
        print('no Tiny ImageNet with the official layout under {}; '
              'downloading {}'.format(INPUTS, URL), flush=True)
        torch.hub.download_url_to_file(URL, archive)
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(root)
    return here


def decode(paths):
    images = np.empty((len(paths), 64, 64, 3), dtype=np.uint8)
    for i, path in enumerate(paths):
        # a few train images are greyscale
        images[i] = np.asarray(Image.open(path).convert('RGB'))
    return images


def arrays(folder, cache):
    """(train images, train labels, val images, val labels), decoded once"""
    names = ['train_x', 'train_y', 'val_x', 'val_y']
    files = [os.path.join(cache, name + '.npy') for name in names]
    if all(os.path.isfile(f) for f in files):
        return [np.load(f) for f in files]

    with open(os.path.join(folder, 'wnids.txt')) as handle:
        wnids = sorted(line.strip() for line in handle if line.strip())
    label = {wnid: i for i, wnid in enumerate(wnids)}

    train_paths, train_y = [], []
    for wnid in wnids:
        images = os.path.join(folder, 'train', wnid, 'images')
        for name in sorted(os.listdir(images)):
            train_paths.append(os.path.join(images, name))
            train_y.append(label[wnid])
    val_paths, val_y = [], []
    with open(os.path.join(folder, 'val', 'val_annotations.txt')) as handle:
        for line in handle:
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            val_paths.append(os.path.join(folder, 'val', 'images', parts[0]))
            val_y.append(label[parts[1]])
    assert len(train_paths) == 100000 and len(val_paths) == 10000, (
        len(train_paths), len(val_paths))

    out = [decode(train_paths), np.asarray(train_y, dtype=np.int64),
           decode(val_paths), np.asarray(val_y, dtype=np.int64)]
    os.makedirs(cache, exist_ok=True)
    for f, a in zip(files, out):
        save_atomic(f, a)
    return out


def save_atomic(f, a):
    """write then rename, so a reader never sees half a file

    The two runs of a notebook start together and both decode when no
    cache is there yet. Each writes its own part file, named by its pid,
    so neither renames the other's away; the second rename overwrites the
    first with the same arrays. Windows refuses that overwrite while the
    other rename is under way; the file is then already there, complete.
    """
    part = '{}.{}.part.npy'.format(f, os.getpid())
    np.save(part, a)
    try:
        os.replace(part, f)
    except OSError:
        if not os.path.isfile(f):
            raise
        os.remove(part)


class TinyImageNet(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = Image.fromarray(self.images[index])
        return self.transform(image), int(self.labels[index])


def dataset(train_transforms, val_transforms, test_transforms):
    root = getattr(FLAGS, 'dataset_dir', 'data')
    folder = locate(root)
    check_official(folder)
    # the cache goes where it can be written: an attached Kaggle input is
    # read-only, so it lands under dataset_dir rather than beside the JPEGs
    cache = os.path.join(root, 'tinyimagenet_cache')
    train_x, train_y, val_x, val_y = arrays(folder, cache)
    train_set = (None if FLAGS.test_only
                 else TinyImageNet(train_x, train_y, train_transforms))
    val_set = TinyImageNet(val_x, val_y, val_transforms)
    return train_set, val_set, None
