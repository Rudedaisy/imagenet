# From https://github.com/pytorch/examples/blob/master/imagenet/main.py
import argparse
import time
import os
import random
import warnings

import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from ptflops.flops_counter import get_model_complexity_info

from imagenet.folder2lmdb import ImageFolderLMDB
from imagenet.utils.train_utils import init_weights, AverageMeter, get_l2_loss, get_lr_scheduler, \
    accuracy, get_optimizer_with_lr, save_checkpoint, build_transform
from imagenet.utils.cross_entropy_with_label_smoothing import LabelSmoothingCrossEntropy
from imagenet.ema import EMA


from src.utils.architecture import Architecture
from src.yaml_utils.yaml_parser import load_and_apply_yaml_config
from models.pruned_layers import *
from models.prune import prune

best_acc1 = 0

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    l2_losses = AverageMeter("L2_loss", ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    # switch to train mode
    model.train()
    end = time.time()
    num_train_batches = len(train_loader)
    scaler = GradScaler()

    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        #if args.gpu is not None:
        #    images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute gradient and do SGD step
        optimizer.zero_grad()

        # compute output
        with autocast(enabled=(args.amp != 0)):
            output = model(images)
            loss = criterion(output, target)
            l2_loss = get_l2_loss(model, args.weight_decay)
            total_loss = loss + l2_loss

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        l2_losses.update(l2_loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))
        top5.update(acc5[0].item(), images.size(0))

        if args.amp != 0:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()

        # Update EMA weights if possible.
        if args.EMA != 0:
            model.update()

        if i % args.report_freq == 0:
            print("[{}/{}] \t Data:: {:.4f} \t Batch: {:.4f} \t Loss: {:.4f} \t L2: {:.4f} \t top1: {:.4f} \t top5: {:.4f}".format(
                i, num_train_batches, data_time.avg, batch_time.avg, losses.avg, l2_losses.avg, top1.avg, top5.avg))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return top1.avg, top5.avg, losses.avg

def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            #if args.gpu is not None:
            #    images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(None, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
            top5.update(acc5[0].item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.report_freq == 0:
                print("Batch: {:.4f} \t Loss: {:.4f} \t top1: {:.4f} \t top5: {:.4f}".format(batch_time.avg, losses.avg, top1.avg, top5.avg))

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.4f} Acc@5 {top5.avg:.4f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg


def main_worker(ngpus_per_node, config):
    print(vars(config))

    global best_acc1
    if config.multiprocessing_distributed:
        dist_url = "env://"
        if dist_url == "env://" and config.rank == -1:
            config.rank = int(os.environ["RANK"])
        if config.multiprocessing_distributed:
            config.rank = config.rank * ngpus_per_node
        dist.init_process_group(backend="nccl", init_method=dist_url,
                                world_size=config.world_size, rank=config.rank)


    print("=> Creating model from '{}':".format(config.prototxt))
    model = torch.hub.load('pytorch/vision', 'alexnet', pretrained=True) #Architecture(config.prototxt)
    model_name = model.model_name
    print("Created model %s" %(model_name))


    flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=False, verbose=False,
                                              print_per_layer_stat=False, ignore_modules=[nn.ReLU, nn.BatchNorm2d])

    print("FLOPS: %.5f (M)" % (flops / 1e6))
    print("Params: %.5f (M)" % (params / 1e6))

    model_root_path = model.save_path
    if not os.path.exists(model_root_path):
        os.makedirs(model_root_path)

    if config.verbose:
        print(model)

    model.apply(init_weights)

    tb_path = os.path.join(model_root_path, "training_log")
    writer = SummaryWriter(tb_path)
    writer.flush()
    writer.add_graph(model, torch.rand((1, 3, 224, 224)))

    if config.multiprocessing_distributed:
        print("Use DDP...")
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        model.cuda()
        # DistributedDataParallel will divide and allocate batch_size to all
        # available GPUs if device_ids are not set
        model = torch.nn.parallel.DistributedDataParallel(model)

    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    # Finally, use EMA to stablize training.
    if config.EMA != 0:
        print("Wrapping model with EMA and decay={}".format(config.EMA))
        model = EMA(model, decay=config.EMA)

    # Check if best checkpoint exists, or evaluate force the weight loading.
    best_model_path = os.path.join(model_root_path, "model_best.pth")
    if os.path.exists(best_model_path) or args.evaluate:
        print("Loading checkpoint...")
        if isinstance(model, EMA):
            checkpoint = torch.load(best_model_path)
            # Both model and shadow load the same weights.
            model.model.load_state_dict(checkpoint['state_dict'])
            model.shadow.load_state_dict(checkpoint['state_dict'])
        else:
            checkpoint = torch.load(best_model_path)
            model.load_state_dict(checkpoint['state_dict'])

        print("Done!")

    # define loss function (criterion) and optimizer
    criterion = LabelSmoothingCrossEntropy(config.label_smoothing).cuda()

    optimizer = get_optimizer_with_lr(model, config.lr, config.optimizer)

    lr_scheduler = get_lr_scheduler(optimizer, config.lr_schedule, config.epochs)

    # optionally resume from a checkpoint
    cudnn.benchmark = True

    # Build the dataset. Whether LMDB or not.
    if config.lmdb:
        traindir = os.path.join(config.data, 'train.lmdb')
        valdir = os.path.join(config.data, 'val.lmdb')
        train_transforms = build_transform(config.augmentation, training=True)
        train_dataset = ImageFolderLMDB(traindir, train_transforms)
        val_transforms = build_transform(config.augmentation, training=False)
        val_dataset = ImageFolderLMDB(valdir, val_transforms)
    else:
        traindir = os.path.join(config.data, 'train')
        valdir = os.path.join(config.data, 'val')
        train_transforms = build_transform(config.augmentation, training=True)
        train_dataset = datasets.ImageFolder(traindir, train_transforms)
        val_transforms = build_transform(config.augmentation, training=False)
        val_dataset = datasets.ImageFolder(valdir, val_transforms)

    if config.multiprocessing_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, drop_last=True)
    else:
        train_sampler = None

    print("Number of workers: {}".format(config.workers))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.train_batch_size, shuffle=(train_sampler is None),
        num_workers=config.workers, pin_memory=True, sampler=train_sampler,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.val_batch_size, shuffle=False,
        num_workers=config.workers, pin_memory=True)

    if config.evaluate:
        validate(val_loader, model, criterion, config)
        return

    for _ in range(config.start_epochs):
        lr_scheduler.step()

    for epoch in range(config.start_epochs, config.epochs):
        print("Epoch %d/%d" %(epoch, config.epochs))
        print("Learning rate: %s" %lr_scheduler.get_last_lr())
        if config.multiprocessing_distributed:
            train_sampler.set_epoch(epoch)
        # train for one epoch
        acc1, acc5, loss = train(train_loader, model, criterion, optimizer, epoch, config)
        tr_acc1, tr_acc5, tr_loss = acc1, acc5, loss
        # evaluate on validation set
        acc1, acc5, loss = validate(val_loader, model, criterion, config)
        val_acc1, val_acc5, val_loss = acc1, acc5, loss

        writer.add_scalar('train_top1_acc', tr_acc1, epoch)
        writer.add_scalar('train_top5_acc', tr_acc5, epoch)
        writer.add_scalar('val_top1_acc', val_acc1, epoch)
        writer.add_scalar('val_top5_acc', val_acc5, epoch)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not config.multiprocessing_distributed or (config.multiprocessing_distributed
                                                    and config.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'model_name': model_name,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best, model_root_path)

        lr_scheduler.step()

def main(args):
    config = load_and_apply_yaml_config(args.yaml_cfg)
    if config.cudnn_deterministic != 0:
        seed = 233
        random.seed(seed)
        torch.manual_seed(seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    config.world_size = 1
    config.rank = -1
    # config.distributed = config.world_size > 1 or config.multiprocessing_distributed
    config.verbose = args.verbose
    config.evaluate = args.evaluate

    ngpus_per_node = torch.cuda.device_count()
    if config.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        config.world_size = ngpus_per_node * config.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(ngpus_per_node, config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    parser.add_argument("--yaml_cfg", type=str, default=None,
                        help="Path to yaml config")
    parser.add_argument("--verbose", action='store_true', default=False,
                        help="Whether to verbose model.")
    parser.add_argument("--evaluate", action='store_true', default=False,
                        help="Evaluation only option.")
    global_args = parser.parse_args()
    main(global_args)
