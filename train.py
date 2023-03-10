# train.py
# !/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import os
import random
import time
import warnings
from itertools import combinations
from pprint import pprint

import torch
import torch.nn as nn
import torch.optim as optim
from torch.backends import cudnn
from torch.nn.functional import kl_div
from torch.utils.tensorboard import SummaryWriter

from conf import get_args
from conf import get_experiment_name
from conf import settings, get_checkpoint_path
from datasets import get_dataloaders
from utils import WarmUpLR
from utils import best_acc_weights
from utils import get_network
from utils import last_epoch
from utils import most_recent_weights
from validate_utils import AverageMeter


def cross_loss(
        output1: torch.Tensor, output2: torch.Tensor,
        T: int = 1, labels=None
):
    p = (output1 / T).softmax(dim=1)
    q = (output2 / T).softmax(dim=1)

    # loss = (p - q) * (p.log() - q.log())
    loss_pq = kl_div(p.log(), q, reduction='none')
    loss_qp = kl_div(q.log(), p, reduction='none')

    # get only correct predictions by labels 
    if labels is not None:
        label1 = labels[:len(labels) // 2]
        label2 = labels[len(labels) // 2:]

        correct_pred1 = output1.argmax(1).eq(label1)
        correct_pred2 = output2.argmax(1).eq(label2)

        correct_pred = correct_pred1 * correct_pred2

        loss_pq *= correct_pred[:, None]
        loss_qp *= correct_pred[:, None]

    return (loss_pq.mean() + loss_qp.mean()) * T ** 2


def grad_logging(net, n_iter):
    last_layer = list(net.children())[-1]
    for name, param in last_layer.named_parameters():
        if 'weight' in name:
            writer.add_scalar('LastLayerGradients/grad_norm2_weights', param.grad.norm(), n_iter)
        if 'bias' in name:
            writer.add_scalar('LastLayerGradients/grad_norm2_bias', param.grad.norm(), n_iter)


def train(net, dataloader, epoch):
    start = time.time()
    net.train()

    total_losses = AverageMeter()
    student_losses = AverageMeter()
    cross_losses = AverageMeter()

    for batch_index, (images, labels) in enumerate(dataloader):

        labels = torch.cat(labels, 0)
        images = torch.cat(images, 0)
        # print(torch.norm(images[:len(images) // 2] - images[len(images) // 2:]))

        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)

        student_losses.update(loss.item(), images.size(0))

        if args.use_cross_loss and args.multiply_data > 1 and epoch > args.cross_loss_start_epoch:
            cl_labels = None
            if args.only_correct_cross_loss:
                cl_labels = labels

            num_elem_in_split = round(args.b / args.multiply_data)
            out_split = torch.split(outputs, num_elem_in_split)
            out_split = [out for out in out_split if len(out) == num_elem_in_split]

            cross_l = torch.zeros_like(loss)
            for out1, out2 in combinations(out_split, 2):
                cross_l += cross_loss(
                    out1, out2,
                    T=args.soft_temper,
                    labels=cl_labels
                )

            cross_losses.update(cross_l.item(), images.size(0))
            cross_l *= args.cross_loss_weight
            loss += cross_l

        total_losses.update(loss.item(), images.size(0))

        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(dataloader) + batch_index + 1

        print(
            f'Training Epoch: {epoch} '
            f'[{batch_index * args.b + len(images)}/{len(dataloader) * args.b}]\t'
            f'LR: {optimizer.param_groups[0]["lr"]:0.6f}\n'
        )

        grad_logging(net, n_iter)

        if args.use_cross_loss and args.multiply_data > 1:
            print(
                f'Total Loss {total_losses.val:.4f} ({total_losses.avg:.4f})\t'
                f'CE Loss {student_losses.val:.4f} ({student_losses.avg:.4f})\t'
                f'Cross loss {cross_losses.val:.4f} ({cross_losses.avg:.4f})\t'
            )

            writer.add_scalar('Train/loss', total_losses.val, n_iter)
            writer.add_scalar('Train/CE_loss', student_losses.val, n_iter)
            writer.add_scalar('Train/cross_loss', cross_losses.val, n_iter)
        else:
            print(
                f'CE Loss {student_losses.val:.4f} ({student_losses.avg:.4f})\t'
            )
            writer.add_scalar('Train/CE_loss', student_losses.val, n_iter)

        if epoch <= args.warm:
            warmup_scheduler.step()

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram(f'{layer}/{attr}', param, epoch)

    finish = time.time()

    print(f'epoch {epoch} training time consumed: {finish - start:.2f}s')


@torch.no_grad()
def eval_training(net, dataloader, epoch=0, tb=True):
    start = time.time()
    net.eval()

    n_sampels = len(dataloader.dataset)
    test_losses = AverageMeter()
    correct = 0.0

    for (images, labels) in dataloader:

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_losses.update(loss.item(), images.size(0))
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print(
        f'Test set: Epoch: {epoch},'
        f'Average loss: {test_losses.avg:.4f},'
        f'Accuracy: {correct.float() / n_sampels:.4f},'
        f'Time consumed:{finish - start:.2f}s'
        '\n'
    )

    # add informations to tensorboard
    if tb:
        writer.add_scalar('Test/Average loss', test_losses.avg, epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / n_sampels, epoch)

    return correct.float() / n_sampels


def train_val_loop(net, training_loader, test_loader, checkpoint_path, best_acc, resume_epoch):
    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        if args.resume:
            if epoch <= resume_epoch:
                continue

        train(net, training_loader, epoch)
        acc = eval_training(net, test_loader, epoch)

        # start to save best performance model after learning rate decay to 0.01
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            weights_path = checkpoint_path / f'{args.net}-{epoch}-best.pth'
            print(f'saving weights file to {weights_path}')
            torch.save(net.state_dict(), weights_path)
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            weights_path = checkpoint_path / f'{args.net}-{epoch}-regular.pth'
            print(f'saving weights file to {weights_path}')
            torch.save(net.state_dict(), weights_path)


def resume_network(net, checkpoint_path):
    best_acc = 0.0
    resume_epoch = -1
    if args.resume:
        best_weights = best_acc_weights(str(checkpoint_path))
        if best_weights:
            weights_path = checkpoint_path / best_weights
            print(f'found best acc weights file: {weights_path}')
            print('load best training file to test acc...')
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(net, tb=False)
            print(f'best acc is {best_acc:0.2f}')

        recent_weights_file = most_recent_weights(str(checkpoint_path))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = checkpoint_path / recent_weights_file
        print(f'loading weights file {weights_path} to resume training.....')
        net.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(str(checkpoint_path))

    return net, best_acc, resume_epoch


if __name__ == '__main__':

    args = get_args()

    if args.x2_epoch:
        settings.EPOCH *= 2
        settings.SAVE_EPOCH *= 2

        settings.MILESTONES[0] *= 2
        settings.MILESTONES[1] *= 2
        settings.MILESTONES[2] *= 2

        print("New settings")
        pprint(vars(settings))

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    warnings.warn('You have chosen to seed training. '
                  'This will turn on the CUDNN deterministic setting, '
                  'which can slow down your training considerably! '
                  'You may see unexpected behavior when restarting '
                  'from checkpoints.')

    exp_name = get_experiment_name(args)

    net = get_network(args)

    # data preprocessing:
    training_loader, test_loader = get_dataloaders(args)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES,
                                                     gamma=0.2)  # learning rate decay
    iter_per_epoch = len(training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    checkpoint_path = get_checkpoint_path(args, exp_name)

    # use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    # since tensorboard can't overwrite old values
    # so the only way is to create a new tensorboard log
    log_dir = settings.LOG_DIR
    if args.use_distil_aug:
        log_dir = os.path.join(
            log_dir, 'distil_aug',
            f'w_{args.distil_aug_weight}_'
            f'func_{args.distil_function}_temp_{args.temperature}_1mlstone'
        )

    writer = SummaryWriter(log_dir=os.path.join(
        log_dir, exp_name, f"seed{args.seed}", settings.TIME_NOW))

    input_tensor = next(iter(test_loader))[0]
    if args.gpu:
        input_tensor = input_tensor.cuda()
    writer.add_graph(net, input_tensor)

    # create checkpoint folder to save model
    checkpoint_path.mkdir(exist_ok=True, parents=True)

    net, best_acc, resume_epoch = resume_network(net, checkpoint_path)

    train_val_loop(net, training_loader, test_loader, checkpoint_path, best_acc, resume_epoch)

    writer.close()
