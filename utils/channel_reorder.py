"""Permute channels so the prefix holds the ones a criterion likes best.

US-Net slices channels as weight[:k], and which channels sit in that
prefix is settled by initialisation. Once-for-All reorders them by
importance when it makes width elastic. This does the same thing here.

The hard part is not the score, it is which tensors have to move
together. A BasicBlock computes body(x) + shortcut(x), so the two
branches write into one channel space and any permutation of it has to
reach both. The identity shortcut extends that space backwards: when a
block has inp == outp and stride 1 its shortcut is the input itself, so
the block's output space is the same space as its input. Following that
through, a whole stage shares one space, and for the first stage it
reaches back into the stem.

Two kinds of group come out of it:

  shared   the output space of a stage. Written by the stem or by every
           block's second conv and shortcut conv, read by every block's
           first conv and shortcut conv in that stage and the first
           block of the next, and for the last stage by the classifier.

  private  the space between a block's two convs. Written by the first
           conv, read by the second, and touched by nothing else.

A permutation applied consistently is a symmetry of the full network:
at width 1.00 the output must not move at all. At a narrow width it
must, because a different set of channels is now in the prefix. That
pair of facts is what tests/test_channel_reorder.py checks, and it is
the only way to know the groups are right, since getting them wrong
produces a network that still runs.
"""
import torch


def _unwrap(model):
    return model.module if hasattr(model, 'module') else model


def _conv(block, index):
    return block.body[index]


def collect(model):
    """the channel spaces, each as the tensors that have to move together

    Returns a list of dicts with:
      size     how many channels the space has
      produce  (conv weight, bn) pairs writing into it, rows to permute
      consume  conv or linear weights reading from it, columns to permute
    """
    net = _unwrap(model)
    features = net.features
    stem_conv, stem_bn = features[0][0], features[0][1]

    # Walked in the order the forward pass runs, because that is the only
    # way to get it right. A block whose shortcut is a conv opens a new
    # space and reads the old one; a block whose shortcut is the identity
    # writes into the space it read. Assigning by stage instead puts the
    # first block of every stage in the wrong group, which is what the
    # first version of this did.
    groups = []
    live = {'size': stem_conv.out_channels_max,
            'produce': [(stem_conv.weight, stem_bn)],
            'consume': [], 'shared': True}
    groups.append(live)

    for block in [m for m in features if hasattr(m, 'body')]:
        changes = len(block.shortcut) > 0
        live['consume'].append(_conv(block, 0).weight)
        if changes:
            live['consume'].append(block.shortcut[0].weight)
            live = {'size': _conv(block, 3).out_channels_max,
                    'produce': [(_conv(block, 3).weight, block.body[4]),
                                (block.shortcut[0].weight,
                                 block.shortcut[1])],
                    'consume': [], 'shared': True}
            groups.append(live)
        else:
            live['produce'].append((_conv(block, 3).weight, block.body[4]))
        groups.append({
            'size': _conv(block, 0).out_channels_max,
            'produce': [(_conv(block, 0).weight, block.body[1])],
            'consume': [_conv(block, 3).weight],
            'shared': False})

    live['consume'].append(net.classifier[-1].weight)
    return groups


def score_l1(group):
    """how much the convs writing into this space put into each channel

    Summed over every producer, because they all write the same space
    and a channel is only as quiet as the loudest thing writing to it.
    """
    total = None
    for weight, _ in group['produce']:
        piece = weight.detach().abs().sum(dim=(1, 2, 3))
        total = piece if total is None else total + piece
    return total


def score_read(group):
    """how much the convs reading this space take from each channel

    The criterion Once-for-All uses: importance is what the next layer
    asks for, not what this one produces.
    """
    total = None
    for weight in group['consume']:
        piece = weight.detach().abs()
        piece = piece.sum(dim=(0, 2, 3)) if piece.dim() == 4 \
            else piece.sum(dim=0)
        total = piece if total is None else total + piece
    return total


def _move(param, order, dim, optimizer):
    """permute a parameter and whatever the optimizer remembers about it

    The momentum buffer is indexed the same way the parameter is, so
    permuting one without the other feeds every channel the velocity of
    whichever channel used to sit at its index. It would keep training
    and the damage would be invisible.
    """
    with torch.no_grad():
        param.data = param.data.index_select(dim, order).clone()
        if optimizer is None:
            return
        state = optimizer.state.get(param)
        if not state:
            return
        for key, value in list(state.items()):
            if torch.is_tensor(value) and value.shape == param.shape:
                state[key] = value.index_select(dim, order).clone()


def apply(group, order, optimizer=None):
    """move every tensor in the space into the given order"""
    for weight, bn in group['produce']:
        _move(weight, order, 0, optimizer)
        if bn is None:
            continue
        if bn.weight is not None:
            _move(bn.weight, order, 0, optimizer)
        if bn.bias is not None:
            _move(bn.bias, order, 0, optimizer)
        # The per-width running statistics cannot be permuted into
        # anything meaningful: a channel that moves from index 300 to
        # index 5 now needs statistics in the width-0.25 copy, which
        # never tracked it. They are reset instead of rearranged into a
        # plausible-looking lie. Training reads batch statistics, and
        # the calibration pass rebuilds these from scratch, so what is
        # lost is the per-epoch validation line until it refills.
        with torch.no_grad():
            for inner in getattr(bn, 'bn', []):
                inner.reset_running_stats()
    for weight in group['consume']:
        _move(weight, order, 1, optimizer)


def overlap(score, fraction=0.25):
    """how much of the top fraction the prefix already holds

    Reported before the permutation runs, because it is the number that
    says whether there was anything to do. Read on a finished K it comes
    out at 96 to 98 per cent, which is why this branch exists to test
    the timing rather than the idea: what that reading cannot say is
    whether the prefix was already sorted at epoch ten, when there is
    still training left to benefit.
    """
    k = max(1, int(round(score.numel() * fraction)))
    top = set(score.argsort(descending=True)[:k].tolist())
    return len(top & set(range(k))) / k


def reorder(model, criterion='l1', optimizer=None):
    """permute every channel space so the prefix holds the best channels"""
    scorer = {'l1': score_l1, 'read': score_read}[criterion]
    moved = 0
    already = []
    groups = collect(model)
    for group in groups:
        score = scorer(group)
        already.append(overlap(score))
        order = score.argsort(descending=True)
        moved += int((order != torch.arange(
            order.numel(), device=order.device)).sum())
        apply(group, order, optimizer)
    return len(groups), moved, sum(already) / len(already)
