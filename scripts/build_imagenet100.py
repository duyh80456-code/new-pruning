"""configs for ImageNet-100 with ResNet-50: SlimOT (KH) and US-Net (KI)

    python scripts/build_imagenet100.py

ImageNet-100 is the 100-class ImageNet subset of the class-incremental
literature (seed 1993; 128,856 train and 5,000 val images at full
resolution), already on the lab's RTX 4090 machine. KH is DO (SlimOT on
CIFAR-100, ResNet-50, seed 1995) and KI is BX (US-Net, same) with only the
data and the recipe moved to ImageNet: 224x224 inputs through the
torchvision stem (stem: imagenet), the US-Net code's imagenet1k_basic
augmentation and loader, batch 128 to fit 24 GB with lr 0.05 (0.1 per 256,
the linear rule), weight decay 1e-4 as on ImageNet, BN recalibration on 40
batches of 128 (the 5,120 images CIFAR used). SlimOT keeps every CIFAR-100
setting of its own: blur 0.2, both transport weights 1, four Width-Band
Heads, transport from epoch 6.
"""
import io
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = '/home/s24gbn1/Documents/vinhpad/data/seed_1993_subset_100_imagenet/data'

# (letter, branch, source branch, what it is)
RUNS = [
    ('KH', 'kh_k_heads4_r50_in100', 'do_k_late_heads4_r50', 'SlimOT'),
    ('KI', 'ki_a_r50_in100', 'bx_a_r50', 'US-Net'),
]
SHARED = [
    ('dataset: data.cifar100\n', 'dataset: imagenet1k\n'),
    ('data_transforms: data.cifar100\n', 'data_transforms: imagenet1k_basic\n'),
    ('data_loader: data.cifar100\n', 'data_loader: imagenet1k_basic\n'),
    ('dataset_dir: data\n', 'dataset_dir: {}\n'.format(DATA)),
    ('image_size: 32\n', 'image_size: 224\nstem: imagenet\n'),
    ('weight_decay: 0.0005\n', 'weight_decay: 0.0001\n'),
    ('lr: 0.2\n', 'lr: 0.05\n'),
    ('gpu_type: "nvidia-tesla-t4"\n', 'gpu_type: "nvidia-geforce-rtx-4090"\n'),
]
FULL = [('data_loader_workers: 2\n', 'data_loader_workers: 16\n'),
        ('batch_size: 256\n', 'batch_size: 128\n'),
        ('bn_cal_batch_num: 20\n', 'bn_cal_batch_num: 40\n')]
SMOKE = [('data_loader_workers: 2\n', 'data_loader_workers: 8\n')]


def swap(text, old, new):
    if text.count(old) != 1:
        raise SystemExit('{!r} found {} times'.format(old, text.count(old)))
    return text.replace(old, new)


def body_of(text):
    """the config after its leading comment block"""
    lines = text.split('\n')
    i = 0
    while i < len(lines) and (lines[i].startswith('#') or not lines[i].strip()):
        i += 1
    return '\n'.join(lines[i:])


for letter, branch, source, what in RUNS:
    for kind in ('cifar100', 'smoke'):
        src = os.path.join(ROOT, 'apps', '{}_{}.yml'.format(kind, source))
        text = io.open(src, encoding='utf-8').read()
        for old, new in SHARED + (FULL if kind == 'cifar100' else SMOKE):
            text = swap(text, old, new)
        text = swap(text, 'log_dir: logs/{}_{}\n'.format(kind, source),
                    'log_dir: logs/{}_{}\n'.format(kind, branch))
        if kind == 'cifar100':
            header = (
                '# =========================== Branch {} ===========================\n'
                '# {} on ImageNet-100 (the seed-1993 subset, full resolution) with\n'
                '# ResNet-50, seed 1995: {} with the data and recipe moved to\n'
                '# ImageNet (224 inputs, torchvision stem, batch 128, lr 0.05, weight\n'
                '# decay 1e-4) and nothing else changed. See\n'
                '# scripts/build_imagenet100.py.\n\n'
            ).format(letter, what, source.split('_')[0].upper())
        else:
            header = '# smoke for Branch {}\n\n'.format(letter)
        out = os.path.join(ROOT, 'apps', '{}_{}.yml'.format(kind, branch))
        with io.open(out, 'w', encoding='utf-8', newline='\n') as handle:
            handle.write(header + body_of(text))
        print('wrote', out)
