"""Universally slimmable ViT for 32x32 inputs

Width w keeps the first d = round(w * dim) channels of the residual stream
at every layer, the way us_resnet keeps the first channels of every conv:
patch embedding, class token, position embedding, every LayerNorm, the
attention projections, the MLP and the classifier input are all sliced to
that prefix, so a narrow width is a sub-network of every wider one and its
pooled feature is a prefix of theirs. That prefix is what the feature
transport terms align on (feature_align: prefix).

Attention (vit_slicing: fractional). The q, k and v blocks are each cut
to their first d rows, so a width holds floor(d / head_dim) whole heads
and, when d is not a multiple of head_dim, one partial head made of the
first channels of the next. At d a multiple of head_dim this is exactly
HydraViT's head-count slicing (its qkv picks the same rows); between
those points it stays continuous, which universal slimming needs. The
partial head is computed as a padded full head at the full head's scale,
so its logits are the truncated dot product of the full head's.

There is no BatchNorm, so calibration has nothing to recompute; the cal
phase in train.py still runs and leaves the weights as they are.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import FLAGS


def _width_dim(width, full):
    return max(1, min(full, int(round(width * full))))


def _layer_norm(norm, x, d):
    return F.layer_norm(x, (d,), norm.weight[:d], norm.bias[:d], norm.eps)


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0, 'dim must divide by the number of heads'
        self.dim = dim
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, d):
        batch, tokens, _ = x.shape
        weight = self.qkv.weight.view(3, self.dim, self.dim)[:, :d, :d]
        bias = self.qkv.bias.view(3, self.dim)[:, :d]
        qkv = F.linear(x, weight.reshape(3 * d, d), bias.reshape(3 * d))
        q, k, v = qkv.split(d, dim=-1)
        heads = -(-d // self.head_dim)
        pad = heads * self.head_dim - d
        if pad:
            q, k, v = (F.pad(t, (0, pad)) for t in (q, k, v))
        q, k, v = (t.view(batch, tokens, heads, self.head_dim).transpose(1, 2)
                   for t in (q, k, v))
        out = F.scaled_dot_product_attention(
            q, k, v, scale=self.head_dim ** -0.5)
        out = out.transpose(1, 2).reshape(batch, tokens, heads * self.head_dim)
        out = out[..., :d]
        return F.linear(out, self.proj.weight[:d, :d], self.proj.bias[:d])


class Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.hidden = hidden
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x, d, h):
        x = F.linear(x, self.fc1.weight[:h, :d], self.fc1.bias[:h])
        x = F.gelu(x)
        return F.linear(x, self.fc2.weight[:d, :h], self.fc2.bias[:d])


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

    def forward(self, x, d, h):
        x = x + self._drop(self.attn(_layer_norm(self.norm1, x, d), d))
        return x + self._drop(self.mlp(_layer_norm(self.norm2, x, d), d, h))


class Model(nn.Module):
    def __init__(self, num_classes=100, input_size=32):
        super().__init__()
        dim = getattr(FLAGS, 'vit_dim', 384)
        depth = getattr(FLAGS, 'vit_depth', 7)
        heads = getattr(FLAGS, 'vit_heads', 8)
        patch = getattr(FLAGS, 'vit_patch', 4)
        hidden = int(dim * getattr(FLAGS, 'vit_mlp_ratio', 2))
        slicing = getattr(FLAGS, 'vit_slicing', 'fractional')
        if slicing != 'fractional':
            raise ValueError('vit_slicing {} is not implemented'.format(
                slicing))
        assert input_size % patch == 0
        self.dim = dim
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

    def dims(self):
        width = self.width_mult
        if callable(width):
            raise ValueError('us_vit takes one width per forward')
        return (_width_dim(width, self.dim),
                _width_dim(width, self.hidden))

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
        d, h = self.dims()
        x = F.conv2d(x, self.patch_embed.weight[:d], self.patch_embed.bias[:d],
                     stride=self.patch)
        x = x.flatten(2).transpose(1, 2)
        cls = self.cls_token[..., :d].expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed[..., :d]
        for block in self.blocks:
            x = block(x, d, h)
        feature = _layer_norm(self.norm, x, d)[:, 0]
        head = self.head()
        logits = F.linear(feature, head.weight[:, :d], head.bias)
        if not getattr(FLAGS, 'return_features', False):
            return logits
        taps = tuple(getattr(FLAGS, 'feature_layers', ['final']))
        if taps != ('final',):
            raise ValueError('us_vit taps only the final feature')
        return logits, (feature,)

    def analytic_macs(self):
        """multiply-accumulates of one image at the current width"""
        d, h = self.dims()
        tokens = self.num_patches + 1
        embed = self.num_patches * d * 3 * self.patch * self.patch
        per_block = tokens * (3 * d * d + d * d + 2 * d * h)
        attention = 2 * tokens * tokens * d
        head = d * self.num_classes
        return embed + len(self.blocks) * (per_block + attention) + head

    def analytic_params(self):
        """parameters the current width uses"""
        d, h = self.dims()
        embed = d * 3 * self.patch * self.patch + d
        tokens = (self.num_patches + 2) * d
        per_block = (3 * d * d + 3 * d + d * d + d + d * h + h + h * d + d
                     + 4 * d)
        return (embed + tokens + len(self.blocks) * per_block + 2 * d
                + d * self.num_classes + self.num_classes)
