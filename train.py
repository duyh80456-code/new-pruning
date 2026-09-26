import importlib
import os
import time
import random
import math

import torch
from torch import multiprocessing
from torchvision import datasets, transforms
from torch.utils.data.distributed import DistributedSampler
import numpy as np

from utils.model_profiling import model_profiling
from utils.transforms import Lighting
from utils.distributed import init_dist, master_only, is_master
from utils.distributed import get_rank, get_world_size
from utils.distributed import dist_all_reduce_tensor
from utils.distributed import master_only_print as print
from utils.distributed import AllReduceDistributedDataParallel, allreduce_grads
from utils.loss_ops import CrossEntropyLossSoft, CrossEntropyLossSmooth
from utils.loss_ops import WassersteinLossSoft, WassersteinPairLoss
from utils.loss_ops import build_soft_criterion, build_pair_criterion
from utils import channel_reorder
from utils.loss_ops import build_feature_criterion
from utils.loss_ops import build_feature_pair_criterion
from utils.loss_ops import horizontal_pairs, ClasswiseFeatureLoss
from utils.loss_ops import sample_width
from utils.loss_ops import training_widths
from utils.loss_ops import build_confusion_embedding, build_cost_matrix
from utils.loss_ops import build_spread_criterion, get_classifier_weight
from utils.loss_ops import width_gate
from models.slimmable_ops import bn_calibration_init
from models.slimmable_ops import make_divisible
from utils.config import FLAGS
from utils.meters import ScalarMeter, flush_scalar_meters


def get_model():
    """get model"""
    model_lib = importlib.import_module(FLAGS.model)
    model = model_lib.Model(FLAGS.num_classes, input_size=FLAGS.image_size)
    if getattr(FLAGS, 'distributed', False):
        gpu_id = init_dist()
        if getattr(FLAGS, 'distributed_all_reduce', False):
            # seems faster
            model_wrapper = AllReduceDistributedDataParallel(model.cuda())
        else:
            model_wrapper = torch.nn.parallel.DistributedDataParallel(
                model.cuda(), [gpu_id], gpu_id)
    else:
        model_wrapper = torch.nn.DataParallel(model).cuda()
    return model, model_wrapper


def data_transforms():
    """get transform of dataset"""
    if FLAGS.data_transforms in [
            'imagenet1k_basic', 'imagenet1k_inception', 'imagenet1k_mobile']:
        if FLAGS.data_transforms == 'imagenet1k_inception':
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
            crop_scale = 0.08
            jitter_param = 0.4
            lighting_param = 0.1
        elif FLAGS.data_transforms == 'imagenet1k_basic':
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            crop_scale = 0.08
            jitter_param = 0.4
            lighting_param = 0.1
        elif FLAGS.data_transforms == 'imagenet1k_mobile':
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            crop_scale = 0.25
            jitter_param = 0.4
            lighting_param = 0.1
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(crop_scale, 1.0)),
            transforms.ColorJitter(
                brightness=jitter_param, contrast=jitter_param,
                saturation=jitter_param),
            Lighting(lighting_param),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        test_transforms = val_transforms
    else:
        try:
            transforms_lib = importlib.import_module(FLAGS.data_transforms)
            return transforms_lib.data_transforms()
        except ImportError:
            raise NotImplementedError(
                'Data transform {} is not yet implemented.'.format(
                    FLAGS.data_transforms))
    return train_transforms, val_transforms, test_transforms


def dataset(train_transforms, val_transforms, test_transforms):
    """get dataset for classification"""
    if FLAGS.dataset == 'imagenet1k':
        if not FLAGS.test_only:
            train_set = datasets.ImageFolder(
                os.path.join(FLAGS.dataset_dir, 'train'),
                transform=train_transforms)
        else:
            train_set = None
        val_set = datasets.ImageFolder(
            os.path.join(FLAGS.dataset_dir, 'val'),
            transform=val_transforms)
        test_set = None
    elif FLAGS.dataset == 'imagenet1k_val50k':
        if not FLAGS.test_only:
            train_set = datasets.ImageFolder(
                os.path.join(FLAGS.dataset_dir, 'train'),
                transform=train_transforms)
            if hasattr(FLAGS, 'random_seed'):
                seed = FLAGS.random_seed
            else:
                seed = 0
            random.seed(seed)
            val_size = 50000
            random.shuffle(train_set.samples)
            if getattr(FLAGS, 'autoslim', False):
                train_set.samples = train_set.samples[:val_size]
            else:
                train_set.samples = train_set.samples[val_size:]
        else:
            train_set = None
        val_set = datasets.ImageFolder(
            os.path.join(FLAGS.dataset_dir, 'val'),
            transform=val_transforms)
        test_set = None
    else:
        try:
            dataset_lib = importlib.import_module(FLAGS.dataset)
            return dataset_lib.dataset(
                train_transforms, val_transforms, test_transforms)
        except ImportError:
            raise NotImplementedError(
                'Dataset {} is not yet implemented.'.format(FLAGS.dataset_dir))
    return train_set, val_set, test_set


def data_loader(train_set, val_set, test_set):
    """get data loader"""
    train_loader = None
    val_loader = None
    test_loader = None
    # infer batch size
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
    batch_size = int(FLAGS.batch_size/get_world_size())
    if FLAGS.data_loader == 'imagenet1k_basic':
        if getattr(FLAGS, 'distributed', False):
            if FLAGS.test_only:
                train_sampler = None
            else:
                train_sampler = DistributedSampler(train_set)
            val_sampler = DistributedSampler(val_set)
        else:
            train_sampler = None
            val_sampler = None
        if not FLAGS.test_only:
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                pin_memory=True,
                num_workers=FLAGS.data_loader_workers,
                drop_last=getattr(FLAGS, 'drop_last', False))
        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            sampler=val_sampler,
            pin_memory=True,
            num_workers=FLAGS.data_loader_workers,
            drop_last=getattr(FLAGS, 'drop_last', False))
        test_loader = val_loader
    else:
        try:
            data_loader_lib = importlib.import_module(FLAGS.data_loader)
            return data_loader_lib.data_loader(train_set, val_set, test_set)
        except ImportError:
            raise NotImplementedError(
                'Data loader {} is not yet implemented.'.format(
                    FLAGS.data_loader))
    if train_loader is not None:
        FLAGS.data_size_train = len(train_loader.dataset)
    if val_loader is not None:
        FLAGS.data_size_val = len(val_loader.dataset)
    if test_loader is not None:
        FLAGS.data_size_test = len(test_loader.dataset)
    return train_loader, val_loader, test_loader


def get_lr_scheduler(optimizer):
    """get learning rate"""
    warmup_epochs = getattr(FLAGS, 'lr_warmup_epochs', 0)
    if FLAGS.lr_scheduler == 'multistep':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=FLAGS.multistep_lr_milestones,
            gamma=FLAGS.multistep_lr_gamma)
    elif FLAGS.lr_scheduler == 'exp_decaying':
        lr_dict = {}
        for i in range(FLAGS.num_epochs):
            if i == 0:
                lr_dict[i] = 1
            else:
                lr_dict[i] = lr_dict[i-1] * FLAGS.exp_decaying_lr_gamma
        lr_lambda = lambda epoch: lr_dict[epoch]
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda)
    elif FLAGS.lr_scheduler == 'linear_decaying':
        num_epochs = FLAGS.num_epochs - warmup_epochs
        lr_dict = {}
        for i in range(FLAGS.num_epochs):
            lr_dict[i] = 1. - (i - warmup_epochs) / num_epochs
        lr_lambda = lambda epoch: lr_dict[epoch]
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda)
    elif FLAGS.lr_scheduler == 'cosine_decaying':
        num_epochs = FLAGS.num_epochs - warmup_epochs
        lr_dict = {}
        for i in range(FLAGS.num_epochs):
            lr_dict[i] = (
                1. + math.cos(
                    math.pi * (i - warmup_epochs) / num_epochs)) / 2.
        lr_lambda = lambda epoch: lr_dict[epoch]
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda)
    else:
        try:
            lr_scheduler_lib = importlib.import_module(FLAGS.lr_scheduler)
            return lr_scheduler_lib.get_lr_scheduler(optimizer)
        except ImportError:
            raise NotImplementedError(
                'Learning rate scheduler {} is not yet implemented.'.format(
                    FLAGS.lr_scheduler))
    return lr_scheduler


def get_optimizer(model):
    """get optimizer"""
    if FLAGS.optimizer == 'sgd':
        # all depthwise convolution (N, 1, x, x) has no weight decay
        # weight decay only on normal conv and fc
        model_params = []
        for params in model.parameters():
            ps = list(params.size())
            if len(ps) == 4 and ps[1] != 1:
                weight_decay = FLAGS.weight_decay
            elif len(ps) == 2:
                weight_decay = FLAGS.weight_decay
            else:
                weight_decay = 0
            item = {'params': params, 'weight_decay': weight_decay,
                    'lr': FLAGS.lr, 'momentum': FLAGS.momentum,
                    'nesterov': FLAGS.nesterov}
            model_params.append(item)
        optimizer = torch.optim.SGD(model_params)
    else:
        try:
            optimizer_lib = importlib.import_module(FLAGS.optimizer)
            return optimizer_lib.get_optimizer(model)
        except ImportError:
            raise NotImplementedError(
                'Optimizer {} is not yet implemented.'.format(FLAGS.optimizer))
    return optimizer


def set_random_seed(seed=None):
    """set random seed"""
    if seed is None:
        seed = getattr(FLAGS, 'random_seed', 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@master_only
def get_meters(phase):
    """util function for meters"""
    def get_single_meter(phase, suffix=''):
        meters = {}
        meters['loss'] = ScalarMeter('{}_loss/{}'.format(phase, suffix))
        for k in FLAGS.topk:
            meters['top{}_error'.format(k)] = ScalarMeter(
                '{}_top{}_error/{}'.format(phase, k, suffix))
        if phase == 'train':
            meters['lr'] = ScalarMeter('learning_rate')
        return meters

    assert phase in ['train', 'val', 'test', 'cal'], 'Invalid phase.'
    if getattr(FLAGS, 'slimmable_training', False):
        meters = {}
        for width_mult in FLAGS.width_mult_list:
            meters[str(width_mult)] = get_single_meter(phase, str(width_mult))
    else:
        meters = get_single_meter(phase)
    if phase == 'val':
        meters['best_val'] = ScalarMeter('best_val')
    if (phase == 'train' and getattr(FLAGS, 'horizontal_kd', False)
            and getattr(FLAGS, 'slimmable_training', False)):
        # hung off the widest meter because only the per-width meters are
        # flushed, and an unfilled meter divides by zero there
        meters[str(max(FLAGS.width_mult_list))]['pair_loss'] = ScalarMeter(
            'train_pair_loss')
    if (phase == 'train' and getattr(FLAGS, 'ensemble_teacher', False)
            and getattr(FLAGS, 'slimmable_training', False)):
        # same place and the same reason as pair_loss above
        meters[str(max(FLAGS.width_mult_list))]['ens_loss'] = ScalarMeter(
            'train_ens_loss')
    return meters


@master_only
def profiling(model, use_cuda):
    """profiling on either gpu or cpu"""
    print('Start model profiling, use_cuda: {}.'.format(use_cuda))
    # model_profiling runs its own forward and wants a tensor back, so the
    # feature taps have to be off for it. It counts multiply-accumulates
    # and does not care what the forward returns.
    taps = getattr(FLAGS, 'return_features', False)
    FLAGS.return_features = False
    try:
        return _profiling(model, use_cuda)
    finally:
        FLAGS.return_features = taps


def _profiling(model, use_cuda):
    if getattr(FLAGS, 'autoslim', False):
        flops, params = model_profiling(
            model, FLAGS.image_size, FLAGS.image_size, use_cuda=use_cuda,
            verbose=getattr(FLAGS, 'profiling_verbose', False))
    elif getattr(FLAGS, 'slimmable_training', False):
        for width_mult in sorted(FLAGS.width_mult_list, reverse=True):
            model.apply(
                lambda m: setattr(m, 'width_mult', width_mult))
            print('Model profiling with width mult {}x:'.format(width_mult))
            flops, params = model_profiling(
                model, FLAGS.image_size, FLAGS.image_size, use_cuda=use_cuda,
                verbose=getattr(FLAGS, 'profiling_verbose', False))
    else:
        flops, params = model_profiling(
            model, FLAGS.image_size, FLAGS.image_size, use_cuda=use_cuda,
            verbose=getattr(FLAGS, 'profiling_verbose', True))
    return flops, params


def lr_schedule_per_iteration(optimizer, epoch, batch_idx=0):
    """ function for learning rate scheuling per iteration """
    warmup_epochs = getattr(FLAGS, 'lr_warmup_epochs', 0)
    num_epochs = FLAGS.num_epochs - warmup_epochs
    iters_per_epoch = FLAGS.data_size_train / FLAGS.batch_size
    current_iter = epoch * iters_per_epoch + batch_idx + 1
    if getattr(FLAGS, 'lr_warmup', False) and epoch < warmup_epochs:
        linear_decaying_per_step = FLAGS.lr/warmup_epochs/iters_per_epoch
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_iter * linear_decaying_per_step
    elif FLAGS.lr_scheduler == 'linear_decaying':
        linear_decaying_per_step = FLAGS.lr/num_epochs/iters_per_epoch
        for param_group in optimizer.param_groups:
            param_group['lr'] -= linear_decaying_per_step
    elif FLAGS.lr_scheduler == 'cosine_decaying':
        mult = (
            1. + math.cos(
                math.pi * (current_iter - warmup_epochs * iters_per_epoch)
                / num_epochs / iters_per_epoch)) / 2.
        for param_group in optimizer.param_groups:
            param_group['lr'] = FLAGS.lr * mult
    else:
        pass


def equalize_by_width_count(model, widths_train):
    """divide each output channel's gradient by how many widths wrote to it

    Gradient Equilibrium, from the depth-exit literature, moved onto the
    channel axis. A parameter that several predictors traverse accumulates
    more gradient terms than one only the largest touches, and the fix
    there is to divide by that count. Prefix slicing makes the same thing
    happen along the channel index: output channel i is written by every
    sampled width whose channel count exceeds i, so with the sandwich rule
    drawing {1.00, 0.25, w1, w2} the first quarter of channels is updated
    four times a step and the last channel once.

    The count is taken from the widths this step actually sampled, not
    from the closed-form expectation, so nothing here assumes the sampler.

    width_equalize_q is the exponent: 0 leaves the gradient alone and is
    the default, 1 divides by the full count, and the point of a sweep is
    that the one convergence theory covering nested-mask training says the
    per-coordinate count does not need compensating at all. If the mean
    over sixteen widths is flat in q, that theory is right here.
    """
    q = getattr(FLAGS, 'width_equalize_q', 0.0)
    if not q:
        return
    for module in model.modules():
        weight = getattr(module, 'weight', None)
        if weight is None or weight.grad is None:
            continue
        out_max = getattr(module, 'out_channels_max', None)
        if out_max is None:
            out_max = getattr(module, 'out_features_max', None)
        if out_max is None or out_max != weight.size(0):
            # not a slimmable op, or not sliced on dim 0
            continue
        counts = weight.new_zeros(out_max)
        for width in widths_train:
            live = make_divisible(out_max * width)
            counts[:live] += 1.0
        counts.clamp_(min=1.0)
        shape = [out_max] + [1] * (weight.dim() - 1)
        weight.grad.div_(counts.pow(q).view(shape))


def forward_loss(
        model, criterion, input, target, meter, soft_target=None,
        soft_criterion=None, return_soft_target=False, return_acc=False,
        return_output=False):
    """forward model and return loss"""
    output = model(input)
    feature = None
    if isinstance(output, tuple):
        output, feature = output
    if soft_target is not None:
        # Inplace distillation runs at temperature 1 upstream, and by the
        # time the widest width has memorized the training set its soft
        # target is one-hot in all but name. Every divergence agrees on a
        # one-hot target, which is why six branches here landed inside 0.8
        # points of each other. Temperature is what leaves the teacher a
        # distribution for them to disagree about; the squared factor keeps
        # the gradient magnitude comparable across settings.
        temperature = getattr(FLAGS, 'kd_temperature', 1.0)
        per_sample = soft_criterion(
            output / temperature, soft_target).view(-1)
        # The teacher reaches a training error of 0.000, so on most
        # samples its soft target is one-hot in all but name and KD is
        # repeating the label. Its entropy says which samples it still
        # has something to say about. Normalised to mean one, so this
        # moves KD's attention without moving its scale.
        weighting = getattr(FLAGS, 'kd_weighting', 'none')
        if weighting != 'none':
            with torch.no_grad():
                entropy = -(soft_target
                            * soft_target.clamp_min(1e-12).log()).sum(1)
                if weighting == 'entropy':
                    weight = entropy
                elif weighting == 'confidence':
                    weight = entropy.max() - entropy
                else:
                    raise ValueError(
                        'unknown kd_weighting {}'.format(weighting))
                weight = weight / weight.mean().clamp_min(1e-12)
            per_sample = per_sample * weight
        loss = (temperature * temperature) * torch.mean(per_sample)
    else:
        loss = torch.mean(criterion(output, target))
    # topk
    _, pred = output.topk(max(FLAGS.topk))
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct_k = []
    for k in FLAGS.topk:
        correct_k.append(correct[:k].float().sum(0))
    tensor = torch.cat([loss.view(1)] + correct_k, dim=0)
    # allreduce
    tensor = dist_all_reduce_tensor(tensor)
    # cache to meter
    tensor = tensor.cpu().detach().numpy()
    bs = (tensor.size-1)//2
    for i, k in enumerate(FLAGS.topk):
        error_list = list(1.-tensor[1+i*bs:1+(i+1)*bs])
        if return_acc and k == 1:
            top1_error = sum(error_list) / len(error_list)
            return loss, top1_error
        if meter is not None:
            meter['top{}_error'.format(k)].cache_list(error_list)
    if meter is not None:
        meter['loss'].cache(tensor[0])
    if return_soft_target:
        temperature = getattr(FLAGS, 'kd_temperature', 1.0)
        soft = torch.nn.functional.softmax(output / temperature, dim=1)
        # the ensemble target needs the widest width's raw logits too: it
        # is the only width US-Net never makes a student, and the logits
        # are what a divergence takes on the student side
        if return_output:
            return loss, soft, feature, output
        return loss, soft, feature
    if return_output:
        return loss, output, feature
    return loss


def run_one_epoch(
        epoch, loader, model, criterion, optimizer, meters, phase='train',
        soft_criterion=None, pair_criterion=None, feature_criterion=None,
        feature_pair_criterion=None, confusion=None,
        spread_criterion=None):
    """run one epoch for train/val/test/cal"""
    t_start = time.time()
    assert phase in ['train', 'val', 'test', 'cal'], 'Invalid phase.'
    train = phase == 'train'
    if train:
        model.train()
    else:
        model.eval()
        if phase == 'cal':
            model.apply(bn_calibration_init)
    # change learning rate in each iteration
    if getattr(FLAGS, 'universally_slimmable_training', False):
        max_width = FLAGS.width_mult_range[1]
        min_width = FLAGS.width_mult_range[0]
    elif getattr(FLAGS, 'slimmable_training', False):
        max_width = max(FLAGS.width_mult_list)
        min_width = min(FLAGS.width_mult_list)
    needs_cost = (isinstance(soft_criterion, WassersteinLossSoft)
                  or isinstance(pair_criterion, WassersteinPairLoss))

    if getattr(FLAGS, 'distributed', False):
        loader.sampler.set_epoch(epoch)
    for batch_idx, (input, target) in enumerate(loader):
        if phase == 'cal':
            if batch_idx == getattr(FLAGS, 'bn_cal_batch_num', -1):
                break
        # smoke testing: cut every phase short without touching the data
        if batch_idx == getattr(FLAGS, 'max_iters_per_epoch', -1):
            break
        target = target.cuda(non_blocking=True)
        if train:
            # change learning rate if necessary
            lr_schedule_per_iteration(optimizer, epoch, batch_idx)
            optimizer.zero_grad()
            if getattr(FLAGS, 'slimmable_training', False):
                if getattr(FLAGS, 'universally_slimmable_training', False):
                    # universally slimmable model (us-nets)
                    # where the free samples land is width_sampling,
                    # which US-Net left uniform without saying why
                    # Hold the narrow end back for the first epochs, so
                    # that the ordering of the channels stays arbitrary
                    # while the weights become worth ranking. Without it
                    # the two happen together: what makes a criterion
                    # meaningful here is the prefix taking gradient at
                    # every width, which is also what sorts it, so there
                    # is no moment at which a permutation has both a
                    # signal to use and something left to move. This is
                    # what Once-for-All gets for free by making width
                    # elastic last.
                    # one width during the warm-up, so those epochs
                    # cost less than a sandwich step rather than more:
                    # the branch spends less total compute than the run
                    # it is compared against, not more
                    widths_train = training_widths(
                        epoch, min_width, max_width)
                    if getattr(FLAGS, 'teacher_chain', False):
                        # each width is taught by the next larger one, so
                        # they have to run widest first. mid_widths is put
                        # back in the order horizontal_pairs documents
                        # once the loop is done.
                        widths_train = sorted(widths_train, reverse=True)
                    # the class cost matrix is read off the classifier, which
                    # keeps moving, so refresh it before the widths are run
                    if needs_cost:
                        cost = build_cost_matrix(model, confusion)
                        if isinstance(soft_criterion, WassersteinLossSoft):
                            soft_criterion.set_cost(cost)
                        if isinstance(pair_criterion, WassersteinPairLoss):
                            pair_criterion.set_cost(cost)
                    # with a horizontal term the graphs of two widths have to
                    # be alive at the same time, so backward is deferred to
                    # the end of the loop. Summing first and calling backward
                    # once is arithmetically what the per-width backward did,
                    # it only costs memory
                    deferred = (pair_criterion is not None
                                or feature_pair_criterion is not None
                                or getattr(FLAGS, 'ensemble_teacher', False))
                    # the classwise form needs this batch's labels, set
                    # once a step the way the class cost matrix is
                    for term in (feature_criterion, feature_pair_criterion):
                        if isinstance(term, ClasswiseFeatureLoss):
                            term.set_target(target)
                    losses = []
                    mid_outputs = []
                    mid_features = []
                    mid_widths = []
                    teacher_feature = None
                    for width_mult in widths_train:
                        # the sandwich rule
                        if width_mult in [max_width, min_width]:
                            model.apply(
                                lambda m: setattr(m, 'width_mult', width_mult))
                        elif getattr(FLAGS, 'nonuniform', False):
                            model.apply(lambda m: setattr(
                                m, 'width_mult',
                                lambda: random.uniform(min_width, max_width)))
                        else:
                            model.apply(lambda m: setattr(
                                m, 'width_mult',
                                width_mult))

                        # always track largest model and smallest model
                        if is_master() and width_mult in [
                                max_width, min_width]:
                            meter = meters[str(width_mult)]
                        else:
                            meter = None

                        # inplace distillation. chain_target is what
                        # this width learns from: the widest width
                        # normally, the previous one under teacher_chain.
                        if width_mult == max_width:
                            (loss, soft_target, teacher_feature,
                             teacher_output) = forward_loss(
                                model, criterion, input, target, meter,
                                return_soft_target=True, return_output=True)
                            if confusion is not None:
                                confusion.update(soft_target.detach(), target)
                            # once a step, not once a width: the rows are
                            # shared by every width, so charging it four
                            # times would only rescale it
                            chain_target = soft_target
                            if spread_criterion is not None:
                                loss = loss + (
                                    getattr(FLAGS, 'spread_weight', 0.0)
                                    * spread_criterion(
                                        get_classifier_weight(model)))
                        else:
                            if getattr(FLAGS, 'inplace_distill', False):
                                loss, output, feature = forward_loss(
                                    model, criterion, input, target, meter,
                                    soft_target=chain_target.detach(),
                                    soft_criterion=soft_criterion,
                                    return_output=True)
                            else:
                                loss, output, feature = forward_loss(
                                    model, criterion, input, target, meter,
                                    return_output=True)
                            if getattr(FLAGS, 'teacher_chain', False):
                                # the next width down learns from this one
                                # rather than from the widest, so the gap
                                # the soft target has to describe is small
                                with torch.no_grad():
                                    hot = getattr(FLAGS, 'kd_temperature',
                                                  1.0)
                                    chain_target = torch.softmax(
                                        output / hot, dim=1)
                            # vertical term at the feature level, alongside
                            # whatever the logits are being matched with
                            if (feature_criterion is not None
                                    and teacher_feature is not None):
                                loss = loss + (
                                    getattr(FLAGS, 'feature_weight', 1.0)
                                    * width_gate(width_mult)
                                    * feature_criterion(
                                        feature,
                                        tuple(t.detach()
                                              for t in teacher_feature)))
                            # every width but the teacher is a peer of
                            # every other: none of them stands in a
                            # teacher-student relation to another, which is
                            # the whole argument for coupling them. Which
                            # pairs are actually used is horizontal_pairs.
                            if deferred:
                                mid_outputs.append(output)
                                mid_features.append(feature)
                                mid_widths.append(width_mult)
                        if deferred:
                            losses.append(loss)
                        else:
                            loss.backward()
                    if deferred:
                        if getattr(FLAGS, 'teacher_chain', False):
                            # the loop ran widest first; horizontal_pairs
                            # documents narrowest first, so undo it here
                            # rather than let the pairing quietly change
                            order = sorted(range(len(mid_widths)),
                                           key=lambda i: mid_widths[i])
                            mid_widths = [mid_widths[i] for i in order]
                            mid_outputs = [mid_outputs[i] for i in order]
                            mid_features = [mid_features[i] for i in order]
                        pair_loss = 0.0
                        pairs = horizontal_pairs(mid_widths)
                        for left, right in pairs:
                            here = 0.0
                            if pair_criterion is not None:
                                # the same temperature, on both sides: two
                                # students that have each memorized the data
                                # have as little to say to each other as a
                                # memorized teacher has to say to either
                                hot = getattr(FLAGS, 'kd_temperature', 1.0)
                                here = here + (hot * hot) * torch.mean(
                                    pair_criterion(
                                        mid_outputs[left] / hot,
                                        mid_outputs[right] / hot))
                            if feature_pair_criterion is not None:
                                here = here + feature_pair_criterion(
                                    mid_features[left], mid_features[right])
                            # each pair spans two widths, so the schedule
                            # reads the middle of that pair, not of the set
                            pair_loss = pair_loss + width_gate(
                                0.5 * (mid_widths[left]
                                       + mid_widths[right])) * here
                        # A warm-up epoch runs the widest width alone,
                        # so there are no middle widths, no pairs, and
                        # pair_loss is still the float it was initialised
                        # to. Adding that to losses is a no-op and calling
                        # .item() on it is an AttributeError, which is
                        # how bd and be died on their first real epoch.
                        # The term is skipped rather than cached as a
                        # zero, because a zero would read as a pair term
                        # that ran and had nothing to say.
                        # flush_scalar_meters drops the empty meter, so a
                        # warm-up epoch simply has no pair_loss on its
                        # line.
                        if pairs:
                            # averaged, so horizontal_weight keeps its
                            # meaning when the number of pairs changes
                            pair_loss = pair_loss / len(pairs)
                            losses.append(
                                getattr(FLAGS, 'horizontal_weight', 1.0)
                                * pair_loss)
                            if is_master():
                                meters[str(max_width)]['pair_loss'].cache(
                                    pair_loss.item())
                        # every sampled width learns from the mean of
                        # what all of them said, the widest included.
                        # US-Net trains the widest on the hard label alone
                        # and makes it the sole teacher; EED's ablation
                        # puts that configuration last of the five it
                        # tried, and CoQuant's puts always-the-widest
                        # fifth of six. Both found the weakest member
                        # carries information the strongest one does not.
                        if (getattr(FLAGS, 'ensemble_teacher', False)
                                and mid_outputs):
                            hot = getattr(FLAGS, 'kd_temperature', 1.0)
                            with torch.no_grad():
                                probs = [soft_target] + [
                                    torch.nn.functional.softmax(o / hot, dim=1)
                                    for o in mid_outputs]
                                mean_target = sum(probs) / len(probs)
                            ens = 0.0
                            for logits in [teacher_output] + mid_outputs:
                                ens = ens + (hot * hot) * torch.mean(
                                    soft_criterion(logits / hot, mean_target))
                            ens = ens / (1 + len(mid_outputs))
                            losses.append(
                                getattr(FLAGS, 'ensemble_weight', 1.0) * ens)
                            if is_master():
                                meters[str(max_width)]['ens_loss'].cache(
                                    ens.item())
                        loss = sum(losses)
                        loss.backward()
                else:
                    # slimmable model (s-nets)
                    for width_mult in sorted(
                            FLAGS.width_mult_list, reverse=True):
                        model.apply(
                            lambda m: setattr(m, 'width_mult', width_mult))
                        if is_master():
                            meter = meters[str(width_mult)]
                        else:
                            meter = None
                        if width_mult == max_width:
                            loss, soft_target, _ = forward_loss(
                                model, criterion, input, target, meter,
                                return_soft_target=True)
                        else:
                            if getattr(FLAGS, 'inplace_distill', False):
                                loss = forward_loss(
                                    model, criterion, input, target, meter,
                                    soft_target=soft_target.detach(),
                                    soft_criterion=soft_criterion)
                            else:
                                loss = forward_loss(
                                    model, criterion, input, target, meter)
                        loss.backward()
            else:
                loss = forward_loss(
                    model, criterion, input, target, meters)
                loss.backward()
            if (getattr(FLAGS, 'distributed', False)
                    and getattr(FLAGS, 'distributed_all_reduce', False)):
                allreduce_grads(model)
            # Four losses are summed into one step and the KD terms are not
            # all bounded the way cross entropy is, so a single bad batch
            # early on can take the weights somewhere they never come back
            # from. Off by default, which is upstream behaviour.
            clip = getattr(FLAGS, 'grad_clip', 0)
            if clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            equalize_by_width_count(model, widths_train)
            optimizer.step()
            if is_master() and getattr(FLAGS, 'slimmable_training', False):
                for width_mult in sorted(FLAGS.width_mult_list, reverse=True):
                    meter = meters[str(width_mult)]
                    meter['lr'].cache(optimizer.param_groups[0]['lr'])
            elif is_master():
                meters['lr'].cache(optimizer.param_groups[0]['lr'])
            else:
                pass
        else:
            if getattr(FLAGS, 'slimmable_training', False):
                for width_mult in sorted(FLAGS.width_mult_list, reverse=True):
                    model.apply(
                        lambda m: setattr(m, 'width_mult', width_mult))
                    if is_master():
                        meter = meters[str(width_mult)]
                    else:
                        meter = None
                    forward_loss(model, criterion, input, target, meter)
            else:
                forward_loss(model, criterion, input, target, meters)
    if is_master() and getattr(FLAGS, 'slimmable_training', False):
        for width_mult in sorted(FLAGS.width_mult_list, reverse=True):
            results = flush_scalar_meters(meters[str(width_mult)])
            print('{:.1f}s\t{}\t{}\t{}/{}: '.format(
                time.time() - t_start, phase, str(width_mult), epoch,
                FLAGS.num_epochs) + ', '.join(
                    '{}: {:.4f}'.format(k, v) for k, v in results.items()))
    elif is_master():
        results = flush_scalar_meters(meters)
        print(
            '{:.1f}s\t{}\t{}/{}: '.format(
                time.time() - t_start, phase, epoch, FLAGS.num_epochs) +
            ', '.join('{}: {:.4f}'.format(k, v) for k, v in results.items()))
    else:
        results = None
    return results


def get_conv_layers(m):
    layers = []
    if (isinstance(m, torch.nn.Conv2d) and hasattr(m, 'width_mult') and
            getattr(m, 'us', [False, False])[1] and
            not getattr(m, 'depthwise', False) and
            not getattr(m, 'linked', False)):
        layers.append(m)
    for child in m.children():
        layers += get_conv_layers(child)
    return layers


def slimming(loader, model, criterion):
    """network slimming by slimmable network"""
    model.eval()
    bn_calibration_init(model)
    model.apply(lambda m: setattr(m, 'width_mult', 1.0))
    if getattr(FLAGS, 'distributed', False):
        layers = get_conv_layers(model.module)
    else:
        raise NotImplementedError
    print('Totally {} layers to slim.'.format(len(layers)))
    error = np.zeros(len(layers))
    # get data
    if getattr(FLAGS, 'distributed', False):
        loader.sampler.set_epoch(0)
    input, target = next(iter(loader))
    input = input.cuda()
    target = target.cuda()
    # start to slim
    print('Start to slim...')
    flops = 10e10
    FLAGS.autoslim_target_flops = sorted(FLAGS.autoslim_target_flops)
    autoslim_target_flop = FLAGS.autoslim_target_flops.pop()
    while True:
        flops, params = model_profiling(
            model, FLAGS.image_size, FLAGS.image_size,
            verbose=getattr(FLAGS, 'profiling_verbose', False))
        if flops < autoslim_target_flop:
            if len(FLAGS.autoslim_target_flops) == 0:
                break
            else:
                print('Find autoslim net at flops {}'.format(
                    autoslim_target_flop))
                autoslim_target_flop = FLAGS.autoslim_target_flops.pop()
        for i in range(len(layers)):
            torch.cuda.empty_cache()
            error[i] = 0.
            outc = layers[i].out_channels - layers[i].divisor
            if outc <= 0 or outc > layers[i].out_channels_max:
                error[i] += 1.
                continue
            layers[i].out_channels -= layers[i].divisor
            loss, error_batch = forward_loss(
                model, criterion, input, target, None, return_acc=True)
            error[i] += error_batch
            layers[i].out_channels += layers[i].divisor
        best_index = np.argmin(error)
        print(*[f'{element:.4f}' for element in error])
        layers[best_index].out_channels -= layers[best_index].divisor
        print(
            'Adjust layer {} for {} to {}, error: {}.'.format(
                best_index, -layers[best_index].divisor,
                layers[best_index].out_channels, error[best_index]))
    return


def train_val_test():
    """train and val"""
    torch.backends.cudnn.benchmark = True
    # seed
    set_random_seed()

    # for universally slimmable networks only
    if getattr(FLAGS, 'universally_slimmable_training', False):
        if getattr(FLAGS, 'test_only', False):
            if getattr(FLAGS, 'width_mult_list_test', None) is not None:
                FLAGS.test_only = False
                # skip training and goto BN calibration
                FLAGS.skip_training = True
        else:
            FLAGS.width_mult_list = FLAGS.width_mult_range

    # Upstream never creates this; its job system did. Without it every run
    # trains a full epoch and then dies on the first torch.save.
    if is_master():
        os.makedirs(FLAGS.log_dir, exist_ok=True)

    # model
    model, model_wrapper = get_model()
    if getattr(FLAGS, 'label_smoothing', 0):
        criterion = CrossEntropyLossSmooth(reduction='none')
    else:
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
    if getattr(FLAGS, 'inplace_distill', False):
        soft_criterion = build_soft_criterion()
    else:
        soft_criterion = None
    pair_criterion = build_pair_criterion()
    feature_criterion = build_feature_criterion()
    feature_pair_criterion = build_feature_pair_criterion()
    confusion = build_confusion_embedding()
    spread_criterion = build_spread_criterion()
    horizontal = pair_criterion is not None or (
        feature_pair_criterion is not None)
    if horizontal or feature_criterion is not None:
        if soft_criterion is None:
            raise ValueError(
                'the extra terms need inplace_distill: they are additions '
                'to the vertical logit term, not replacements for it')
        if not getattr(FLAGS, 'universally_slimmable_training', False):
            raise ValueError(
                'the extra terms are only wired into the us-net branch')
    if horizontal:
        # the sandwich rule spends two of the samples on max and min width,
        # so a pair of unrelated middle students needs at least four
        if getattr(FLAGS, 'num_sample_training', 2) < 4:
            raise ValueError(
                'horizontal_kd needs num_sample_training >= 4, got {}'.format(
                    getattr(FLAGS, 'num_sample_training', 2)))
    # the model returns its pooled feature alongside the logits only when
    # something asks for it, so val, calibration and profiling keep the
    # plain signature everywhere else
    FLAGS.return_features = (feature_criterion is not None
                             or feature_pair_criterion is not None)

    # check pretrained
    if getattr(FLAGS, 'pretrained', False):
        checkpoint = torch.load(
            FLAGS.pretrained, map_location=lambda storage, loc: storage,
            weights_only=False)
        # update keys from external models
        if type(checkpoint) == dict and 'model' in checkpoint:
            checkpoint = checkpoint['model']
        if getattr(FLAGS, 'pretrained_model_remap_keys', False):
            new_checkpoint = {}
            new_keys = list(model_wrapper.state_dict().keys())
            old_keys = list(checkpoint.keys())
            for key_new, key_old in zip(new_keys, old_keys):
                new_checkpoint[key_new] = checkpoint[key_old]
                print('remap {} to {}'.format(key_new, key_old))
            checkpoint = new_checkpoint
        model_wrapper.load_state_dict(checkpoint)
        print('Loaded model {}.'.format(FLAGS.pretrained))

    optimizer = get_optimizer(model_wrapper)

    # check resume training
    if os.path.exists(os.path.join(FLAGS.log_dir, 'latest_checkpoint.pt')):
        # weights_only defaults to True from torch 2.6, and a resume
        # checkpoint carries the optimizer state, so the default turns
        # every resume into an UnpicklingError. Kaggle is on 2.10 and
        # the notebooks tell you to resume a timed-out session from
        # exactly this file. It is a file this run wrote itself.
        checkpoint = torch.load(
            os.path.join(FLAGS.log_dir, 'latest_checkpoint.pt'),
            map_location=lambda storage, loc: storage, weights_only=False)
        model_wrapper.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        last_epoch = checkpoint['last_epoch']
        lr_scheduler = get_lr_scheduler(optimizer)
        lr_scheduler.last_epoch = last_epoch
        best_val = checkpoint['best_val']
        train_meters, val_meters = checkpoint['meters']
        print('Loaded checkpoint {} at epoch {}.'.format(
            FLAGS.log_dir, last_epoch))
    else:
        lr_scheduler = get_lr_scheduler(optimizer)
        last_epoch = lr_scheduler.last_epoch
        best_val = 1.
        train_meters = get_meters('train')
        val_meters = get_meters('val')
        # if start from scratch, print model and do profiling
        print(model_wrapper)
        if getattr(FLAGS, 'profiling', False):
            if 'gpu' in FLAGS.profiling:
                profiling(model, use_cuda=True)
            if 'cpu' in FLAGS.profiling:
                profiling(model, use_cuda=False)
            if getattr(FLAGS, 'profiling_only', False):
                return

    # data
    train_transforms, val_transforms, test_transforms = data_transforms()
    train_set, val_set, test_set = dataset(
        train_transforms, val_transforms, test_transforms)
    train_loader, val_loader, test_loader = data_loader(
        train_set, val_set, test_set)

    # autoslim only
    if getattr(FLAGS, 'autoslim', False):
        with torch.no_grad():
            slimming(train_loader, model_wrapper, criterion)
        return

    if getattr(FLAGS, 'test_only', False) and (test_loader is not None):
        print('Start testing.')
        test_meters = get_meters('test')
        with torch.no_grad():
            if getattr(FLAGS, 'slimmable_training', False):
                for width_mult in sorted(FLAGS.width_mult_list, reverse=True):
                    model_wrapper.apply(
                        lambda m: setattr(m, 'width_mult', width_mult))
                    run_one_epoch(
                        last_epoch, test_loader, model_wrapper, criterion,
                        optimizer, test_meters, phase='test')
            else:
                run_one_epoch(
                    last_epoch, test_loader, model_wrapper, criterion,
                    optimizer, test_meters, phase='test')
        return

    if getattr(FLAGS, 'nonuniform_diff_seed', False):
        set_random_seed(getattr(FLAGS, 'random_seed', 0) + get_rank())
    print('Start training.')
    for epoch in range(last_epoch+1, FLAGS.num_epochs):
        if getattr(FLAGS, 'skip_training', False):
            print('Skip training at epoch: {}'.format(epoch))
            break
        lr_scheduler.step()
        # Permute the channels so the prefix holds the ones the criterion
        # likes, the way Once-for-All does when it makes width elastic.
        # Once, at a named epoch: the permutation is free but it discards
        # the co-adaptation the narrow widths have built, so doing it
        # repeatedly would keep paying that without ever settling.
        if getattr(FLAGS, 'reorder_report', False):
            # Reported by a criterion that needs no gradient, and named
            # in the line so an l1 trajectory is never read later as a
            # taylor one. Asking for taylor here would print nan every
            # epoch: nothing has charged the gains at the top of an
            # epoch, so the score does not exist yet and the trajectory
            # this exists to draw would be lost for the whole run. The
            # taylor reading at the epoch that decides anything is the
            # one reorder() returns below.
            shown = getattr(FLAGS, 'reorder_by', 'l1')
            if shown == 'taylor':
                shown = 'l1'
            print('prefix_sorted {} {} {:.4f}'.format(
                epoch, shown, channel_reorder.report(model_wrapper, shown)))
        if getattr(FLAGS, 'width_scalars', False):
            # These have no weight decay - every one-dimensional parameter
            # gets zero - so nothing pulls them back toward 1.0, and a
            # scalar drifting to zero turns its block into a bare
            # identity. Printed per epoch so that failure is legible in a
            # pasted log instead of showing up only as a bad table.
            scales = [p for n, p in model_wrapper.named_parameters()
                      if n.endswith('branch_scale')]
            if scales:
                flat = torch.cat([s.detach().flatten() for s in scales])
                print('branch_scale {} min {:.4f} mean {:.4f} max {:.4f}'
                      .format(epoch, float(flat.min()), float(flat.mean()),
                              float(flat.max())))
        if epoch == getattr(FLAGS, 'reorder_epoch', -1):
            if getattr(FLAGS, 'reorder_by', 'l1') == 'taylor':
                # The Taylor score needs a gradient on the gains, and
                # nothing in the loop has put one there yet at the top
                # of an epoch. A few batches at the widest width, then
                # the gradients are dropped: they are a measurement, not
                # a step, and leaving them would add themselves to the
                # first real update of the epoch.
                model_wrapper.train()
                model_wrapper.apply(
                    lambda m: setattr(m, 'width_mult', 1.0))
                model_wrapper.zero_grad(set_to_none=True)
                charged = getattr(FLAGS, 'taylor_batches', 8)
                for seen, (image, label) in enumerate(train_loader):
                    if seen >= charged:
                        break
                    label = label.cuda(non_blocking=True)
                    # feature_kd branches return (output, features), and
                    # every other call site unwraps that; this one fed
                    # the tuple straight to the loss. Only reachable by
                    # running train.py, which is why it survived every
                    # CPU suite.
                    charge = model_wrapper(image)
                    if isinstance(charge, tuple):
                        charge = charge[0]
                    torch.mean(criterion(charge, label)).backward()
                print('Charged the gains with {} batches for the Taylor '
                      'score.'.format(charged))
            spaces, moved, held = channel_reorder.reorder(
                model_wrapper, getattr(FLAGS, 'reorder_by', 'l1'),
                optimizer)
            # held is the diagnostic the branch exists to collect: how
            # much of the best quarter the prefix already had. Near one
            # means the permutation had nothing to move and whatever
            # this run reports is about noise, not about the idea.
            print('Reordered {} channels across {} spaces by {} at '
                  'epoch {}. The prefix already held {:.0%} of the best '
                  'quarter.'.format(
                      moved, spaces, getattr(FLAGS, 'reorder_by', 'l1'),
                      epoch, held))
            if getattr(FLAGS, 'reorder_by', 'l1') == 'taylor':
                # prefix_sorted tracks l1, because taylor has no score
                # at the top of an epoch. A taylor permutation does not
                # sort l1, so that line stays near a quarter afterwards
                # instead of jumping the way a reorder_by: l1 branch
                # does. The number above is the taylor one, and it is
                # the one that says what this permutation did.
                print('prefix_sorted tracks l1 and will not jump for a '
                      'taylor permutation. The figure above is the '
                      'taylor reading.')
            model_wrapper.zero_grad(set_to_none=True)
        # train
        results = run_one_epoch(
            epoch, train_loader, model_wrapper, criterion, optimizer,
            train_meters, phase='train', soft_criterion=soft_criterion,
            pair_criterion=pair_criterion,
            feature_criterion=feature_criterion,
            feature_pair_criterion=feature_pair_criterion,
            confusion=confusion, spread_criterion=spread_criterion)

        # val
        if val_meters is not None:
            val_meters['best_val'].cache(best_val)
        with torch.no_grad():
            results = run_one_epoch(
                epoch, val_loader, model_wrapper, criterion, optimizer,
                val_meters, phase='val')
        if is_master() and results['top1_error'] < best_val:
            best_val = results['top1_error']
            torch.save(
                {
                    'model': model_wrapper.state_dict(),
                },
                os.path.join(FLAGS.log_dir, 'best_model.pt'))
            print('New best validation top1 error: {:.4f}'.format(best_val))
        # save latest checkpoint
        if is_master():
            torch.save(
                {
                    'model': model_wrapper.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'last_epoch': epoch,
                    'best_val': best_val,
                    'meters': (train_meters, val_meters),
                },
                os.path.join(FLAGS.log_dir, 'latest_checkpoint.pt'))

    if getattr(FLAGS, 'calibrate_bn', False):
        if getattr(FLAGS, 'universally_slimmable_training', False):
            # need to rebuild model according to width_mult_list_test
            width_mult_list = FLAGS.width_mult_range.copy()
            for width in FLAGS.width_mult_list_test:
                if width not in FLAGS.width_mult_list:
                    width_mult_list.append(width)
            FLAGS.width_mult_list = width_mult_list
            new_model, new_model_wrapper = get_model()
            profiling(new_model, use_cuda=True)
            new_model_wrapper.load_state_dict(
                model_wrapper.state_dict(), strict=False)
            model_wrapper = new_model_wrapper
        cal_meters = get_meters('cal')
        print('Start calibration.')
        results = run_one_epoch(
            -1, train_loader, model_wrapper, criterion, optimizer,
            cal_meters, phase='cal')
        print('Start validation after calibration.')
        with torch.no_grad():
            results = run_one_epoch(
                -1, val_loader, model_wrapper, criterion, optimizer,
                cal_meters, phase='val')
        if is_master():
            torch.save(
                {
                    'model': model_wrapper.state_dict(),
                },
                os.path.join(FLAGS.log_dir, 'best_model_bn_calibrated.pt'))
    return


def init_multiprocessing():
    # print(multiprocessing.get_start_method())
    try:
        multiprocessing.set_start_method('fork')
    except (RuntimeError, ValueError):
        # already set, or a platform without fork such as Windows
        pass


def main():
    """train and eval model"""
    init_multiprocessing()
    train_val_test()


if __name__ == "__main__":
    main()
