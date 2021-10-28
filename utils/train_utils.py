import torch
import torchvision.transforms as transforms
import shutil

import os, sys

from imagenet.utils.autoaugment.autoaugment import ImageNetPolicy
from imagenet.utils.cosine_with_warmup import CosineAnnealingLRWarmup
import torch.nn as nn
import math
from imagenet.ema import EMA
from imagenet.utils.rmsprop_tf import RMSpropTF

from src.backend.backend import G

from math import sqrt

def get_optimizer_with_lr(model, lr, optimizer='sgd'):
    if optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr,
                                    momentum=0.9,
                                    weight_decay=0,
                                    nesterov=True)
    elif optimizer == 'sgd++':
        optimizer = torch.optim.SGD(model.parameters(), lr,
                                    momentum=0.875,
                                    weight_decay=0,
                                    nesterov=True)
    elif optimizer == 'rmsprop-tf':
        # This is in-consistent with TF-Rmsprop. eps should be tuned.'
        print("Using RMSProp-TF optimizer.")
        optimizer = RMSpropTF(model.parameters(), lr, eps=1e-3, momentum=.9)
    elif optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr)
    else:
        raise NotImplementedError

    return optimizer


def get_lr_scheduler(optimizer, lr_scheduler='cosine', num_epochs=90):
    if lr_scheduler == "none":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [num_epochs*10, num_epochs*20])
    elif lr_scheduler == "multistep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [num_epochs//3, num_epochs*2//3], gamma=0.1)
    elif lr_scheduler == "cosine":
        lr_scheduler = CosineAnnealingLRWarmup(optimizer, T_max=num_epochs, last_epoch=-1)
    elif lr_scheduler == "cosine-no-warmup":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, last_epoch=-1)
    else:
        raise NotImplementedError

    return lr_scheduler


# -------------Build the PyTorch transformation--------------------
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def build_transform(augmentation, training=True, center_crop_fraction=0.875,
                    image_size=224):
    if not training:
        return transforms.Compose([
                transforms.Resize(int(image_size/center_crop_fraction)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ])
    else:
        if augmentation == "inception":
            return transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(.4, .4, .4, .2),
                transforms.ToTensor(),
                normalize,
            ])
        elif augmentation == "inception-no-color-jitter":
            return transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        elif augmentation == "naive":
            return transforms.Compose([
                transforms.RandomResizedCrop(image_size, ratio=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        elif augmentation == "autoaugment":
            return transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                ImageNetPolicy(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            raise NotImplementedError("Data augmentation {} is not found!".format(augmentation))
    return transforms


#----------Average meters-----------------------------
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def save_checkpoint(state, is_best, ckpt_path='checkpoint.pth.tar'):
    save_ckpt_path = os.path.join(ckpt_path, "model.pth")
    torch.save(state, save_ckpt_path)
    if is_best:
        best_ckpt_path = os.path.join(ckpt_path, "model_best.pth")
        shutil.copyfile(save_ckpt_path, best_ckpt_path)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def init_weights(m):
    if isinstance(m, nn.Linear):
        # fan_out = m.weight.size(0)  # fan-out
        init_range = 0.01
        m.weight.data.normal_(0, init_range)
        m.bias.data.zero_()
    elif isinstance(m, nn.Conv2d):
        # This fixes the TensorFlow prototxts, where Convolution layers are used to represent FC for efficiency.
        try:
            if m.dense_initializer:
                init_range = 0.01
                m.weight.data.normal_(0, init_range)
                m.bias.data.zero_()
            else:
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels / m.groups
                m.weight.data.normal_(0, math.sqrt(1.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()
        except Exception:
            # Otherwise, weights are initialized as usual.
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels / m.groups
            m.weight.data.normal_(0, math.sqrt(1.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()

_DEVICE_CFG = torch.cuda.is_available()

# @torch.cuda.amp.autocast()
def get_l2_loss(model, weight_decay):
    assert isinstance(model, torch.nn.Module)
    if isinstance(model, EMA):
        reg_model = model.model
    else:
        reg_model = model
    l2_loss = None
    for n, m in reg_model.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if isinstance(m, nn.Conv2d):
                if m.groups == m.out_channels and (not G.regularize_depthwise):
                    # Put much fewer regularization on depthwise.
                    # This seems to be even better than a rough 'regularize_depth' option.
                    if l2_loss is None:
                        l2_loss = sqrt(1. / m.groups) * torch.square(torch.norm(m.weight, 2))
                    else:
                        l2_loss += sqrt(1. / m.groups) * torch.square(torch.norm(m.weight, 2))
                else:
                    if l2_loss is None:
                        l2_loss = torch.square(torch.norm(m.weight, 2))
                    else:
                        l2_loss += torch.square(torch.norm(m.weight, 2))
            else:
                if l2_loss is None:
                    l2_loss = torch.square(torch.norm(m.weight, 2))
                else:
                    l2_loss += torch.square(torch.norm(m.weight, 2))
    if _DEVICE_CFG:
        l2_loss = torch.mul(l2_loss, torch.tensor(weight_decay).cuda())
    else:
        l2_loss = torch.mul(l2_loss, torch.tensor(weight_decay))

    return l2_loss
