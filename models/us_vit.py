"""Universally slimmable ViT for 32x32 inputs

One backbone, three ways of cutting it (vit_slicing), so that every row of
the ViT table runs on the same model and differs only in the method:

fractional (US-Net†, the method, HydraViT†)
    Width w keeps the first d = round(w * dim) channels of the residual
    stream at every layer, the way us_resnet keeps the first channels of
    every conv: patch embedding, class token, position embedding, every
    LayerNorm, the attention projections, the MLP and the classifier input
    are all sliced to that prefix, so a narrow width is a sub-network of
    every wider one and its feature is a prefix of theirs, which is what
    the feature transport terms align on (feature_align: prefix). The q, k
    and v blocks are each cut to their first d rows, so a width holds
    floor(d / head_dim) whole heads and, when d is not a multiple of
    head_dim, one partial head made of the first channels of the next. At
    d a multiple of head_dim this is exactly HydraViT's head-count slicing
    (its qkv takes the same rows: ds-kiel/HydraViT, QKVLinear); between
    those points it stays continuous. The partial head is a padded full
    head at the full head's scale, so its logits are the truncated dot
    product of the full head's.

scala (Scala†)
    As BeSpontaneous/Scala-pytorch, models_scala.py: d = int(w * dim) and
    the hidden width int(w * hidden), the narrowest and widest widths take
    the first channels of every tensor and every width between takes the
    last ones (isolated activation); the qkv weight is cut to its first
    (or last) 3d rows as one block and reshaped to the full number of
    heads, each of d / heads channels, so d must divide by the heads.

ffn (MatFormer†)
    As MatViT (google-research/scenic, projects/matvit): only the MLP's
    hidden layer nests, the residual stream and attention stay full. In
    training the width is the hidden fraction g, the same at every layer
    (matformer_granularities). In evaluation the width is a point of the
    common grid and the model picks, by Mix'n'Match, the non-decreasing
    per-layer choice of granularities whose MACs come closest to those of
    the fractional model at that width; below its floor (all layers at
    the smallest granularity) it can only return the floor.

There is no BatchNorm, so calibration has nothing to recompute; the cal
phase in train.py still runs and leaves the weights as they are.
"""
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import FLAGS


def _norm(norm, x, sl):
    d = x.shape[-1]
    return F.layer_norm(x, (d,), norm.weight[sl], norm.bias[sl], norm.eps)


def _cut(size, side):
    """the slice of the first or last size entries"""
    return slice(0, size) if side == 'first' else slice(-size, None)


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0, 'dim must divide by the number of heads'
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, d, side, mode):
        batch, tokens, _ = x.shape
        cut = _cut(d, side)
        if mode == 'scala':
            rows = _cut(3 * d, side)
            qkv = F.linear(x, self.qkv.weight[rows, cut], self.qkv.bias[rows])
            if d % self.heads:
                raise ValueError('scala slicing needs the width to divide '
                                 'by {} heads, got {}'.format(self.heads, d))
            heads, head_dim = self.heads, d // self.heads
            q, k, v = (qkv.view(batch, tokens, 3, heads, head_dim)
                       .permute(2, 0, 3, 1, 4))
            pad = 0
        else:
            weight = self.qkv.weight.view(3, self.dim, self.dim)[:, cut, cut]
            bias = self.qkv.bias.view(3, self.dim)[:, cut]
            qkv = F.linear(x, weight.reshape(3 * d, d), bias.reshape(3 * d))
            q, k, v = qkv.split(d, dim=-1)
            heads = -(-d // self.head_dim)
            head_dim = self.head_dim
            pad = heads * head_dim - d
            if pad:
                q, k, v = (F.pad(t, (0, pad)) for t in (q, k, v))
            q, k, v = (t.view(batch, tokens, heads, head_dim).transpose(1, 2)
                       for t in (q, k, v))
        # the full head's scale in both modes: Scala's attention keeps the
        # scale it was built with (models_scala.py, self.scale)
        out = F.scaled_dot_product_attention(
            q, k, v, scale=self.head_dim ** -0.5)
        out = out.transpose(1, 2).reshape(batch, tokens, heads * head_dim)
        if pad:
            out = out[..., :d]
        return F.linear(out, self.proj.weight[cut, cut], self.proj.bias[cut])


class Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.hidden = hidden
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x, cut, h, side):
        inner = _cut(h, side)
        x = F.linear(x, self.fc1.weight[inner, cut], self.fc1.bias[inner])
        x = F.gelu(x)
        return F.linear(x, self.fc2.weight[cut, inner], self.fc2.bias[cut])


class Block(nn.Module):
    def __init__(self, dim, heads, hidden, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, hidden)
        self.drop_path = drop_path

    def _drop(self, x):
        # stochastic depth, per sample, as in DeiT
        if not self.training or self.drop_path == 0.0:
            return x
        keep = 1.0 - self.drop_path
        mask = x.new_empty(x.shape[0], 1, 1).bernoulli_(keep)
        return x * mask / keep

    def forward(self, x, d, h, side, mode):
        cut = _cut(d, side)
        x = x + self._drop(self.attn(_norm(self.norm1, x, cut), d, side, mode))
        return x + self._drop(self.mlp(_norm(self.norm2, x, cut), cut, h,
                                       side))


class Model(nn.Module):
    def __init__(self, num_classes=100, input_size=32):
        super().__init__()
        dim = getattr(FLAGS, 'vit_dim', 384)
        depth = getattr(FLAGS, 'vit_depth', 7)
        heads = getattr(FLAGS, 'vit_heads', 8)
        patch = getattr(FLAGS, 'vit_patch', 4)
        hidden = int(dim * getattr(FLAGS, 'vit_mlp_ratio', 2))
        self.slicing = getattr(FLAGS, 'vit_slicing', 'fractional')
        if self.slicing not in ('fractional', 'scala', 'ffn'):
            raise ValueError('unknown vit_slicing {}'.format(self.slicing))
        assert input_size % patch == 0
        self.dim = dim
        self.heads = heads
        self.hidden = hidden
        self.patch = patch
        self.num_patches = (input_size // patch) ** 2
        self.num_classes = num_classes
        self.width_mult = max(FLAGS.width_mult_list)

        self.patch_embed = nn.Conv2d(3, dim, patch, patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, dim))
        rates = torch.linspace(0, getattr(FLAGS, 'drop_path', 0.0), depth)
        self.blocks = nn.ModuleList([
            Block(dim, heads, hidden, float(rate)) for rate in rates])
        self.norm = nn.LayerNorm(dim)

        # The same banding as us_resnet: head_groups equal bands of the
        # width range, one classifier each, the widest band's keeping the
        # name classifier and registered last.
        groups = getattr(FLAGS, 'head_groups', 1)
        if groups > 1:
            if self.slicing != 'fractional':
                raise ValueError('head_groups is written for fractional '
                                 'slicing')
            self.narrow_heads = nn.ModuleList([
                nn.Linear(dim, num_classes) for _ in range(groups - 1)])
        self.classifier = nn.Linear(dim, num_classes)
        self.reset_parameters()
        # train.py's profiler hooks every module; the sub-modules here
        # compute through sliced weights, so the count is analytic and
        # lives on the model alone
        for module in self.modules():
            if module is not self:
                module.ignore_model_profiling = True

    def reset_parameters(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    # ---- what the current width runs -------------------------------------

    def _macs(self, d, hidden_per_layer):
        """multiply-accumulates of one image, embedding d, MLP hidden per
        layer"""
        tokens = self.num_patches + 1
        total = self.num_patches * d * 3 * self.patch * self.patch
        for h in hidden_per_layer:
            total += tokens * (4 * d * d + 2 * d * h) + 2 * tokens * tokens * d
        return total + d * self.num_classes

    def _fractional_dims(self, width):
        d = max(1, min(self.dim, int(round(width * self.dim))))
        h = max(1, min(self.hidden, int(round(width * self.hidden))))
        return d, h

    def mix_n_match(self, width):
        """per-layer MLP hidden sizes whose MACs come closest to the
        fractional model's at this width, over non-decreasing choices of
        matformer_granularities"""
        cache = self.__dict__.setdefault('_mix_n_match', {})
        if width in cache:
            return cache[width]
        grans = sorted(getattr(FLAGS, 'matformer_granularities',
                               [0.125, 0.25, 0.5, 1.0]))
        d, h = self._fractional_dims(width)
        target = self._macs(d, [h] * len(self.blocks))
        best = None
        for choice in itertools.combinations_with_replacement(
                grans, len(self.blocks)):
            hidden = [int(round(g * self.hidden)) for g in choice]
            gap = abs(self._macs(self.dim, hidden) - target)
            if best is None or gap < best[0]:
                best = (gap, hidden)
        cache[width] = best[1]
        return best[1]

    def plan(self):
        """(embedding width, MLP hidden per layer, side) at this width"""
        width = self.width_mult
        if callable(width):
            raise ValueError('us_vit takes one width per forward')
        depth = len(self.blocks)
        if self.slicing == 'fractional':
            d, h = self._fractional_dims(width)
            return d, [h] * depth, 'first'
        if self.slicing == 'scala':
            low, high = FLAGS.width_mult_range
            d = int(round(width * self.dim))
            h = int(round(width * self.hidden))
            side = 'first' if width in (low, high) else 'last'
            return d, [h] * depth, side
        # ffn: the hidden fraction in training, a matched grid point in
        # evaluation
        if self.training:
            return (self.dim, [int(round(width * self.hidden))] * depth,
                    'first')
        return self.dim, self.mix_n_match(width), 'first'

    def head(self):
        """the classifier for the width the model is set to"""
        if not hasattr(self, 'narrow_heads'):
            return self.classifier
        groups = len(self.narrow_heads) + 1
        low, high = FLAGS.width_mult_range
        band = min(int((self.width_mult - low) / (high - low) * groups),
                   groups - 1)
        if band == groups - 1:
            return self.classifier
        return self.narrow_heads[band]

    def forward(self, x):
        d, hidden, side = self.plan()
        mode = 'scala' if self.slicing == 'scala' else 'fractional'
        cut = _cut(d, side)
        x = F.conv2d(x, self.patch_embed.weight[cut],
                     self.patch_embed.bias[cut], stride=self.patch)
        x = x.flatten(2).transpose(1, 2)
        cls = self.cls_token[..., cut].expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed[..., cut]
        for block, h in zip(self.blocks, hidden):
            x = block(x, d, h, side, mode)
        feature = _norm(self.norm, x, cut)[:, 0]
        head = self.head()
        logits = F.linear(feature, head.weight[:, cut], head.bias)
        if not getattr(FLAGS, 'return_features', False):
            return logits
        taps = tuple(getattr(FLAGS, 'feature_layers', ['final']))
        if taps != ('final',):
            raise ValueError('us_vit taps only the final feature')
        return logits, (feature,)

    def analytic_macs(self):
        """multiply-accumulates of one image at the current width"""
        d, hidden, _ = self.plan()
        return self._macs(d, hidden)

    def analytic_params(self):
        """parameters the current width uses"""
        d, hidden, _ = self.plan()
        total = d * 3 * self.patch * self.patch + d
        total += (self.num_patches + 2) * d
        for h in hidden:
            total += 4 * d * d + 4 * d + 2 * d * h + h + d + 4 * d
        return total + 2 * d + d * self.num_classes + self.num_classes
