import logging
import time
from pathlib import Path

import numpy as np
import torch
from tqdm.notebook import tqdm
from tqdm.notebook import trange


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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


def agreement(output0, output1):
    pred0 = output0.argmax(dim=1, keepdim=False)
    pred1 = output1.argmax(dim=1, keepdim=False)
    agree = pred0.eq(pred1)
    agree = 100. * torch.mean(agree.type(torch.FloatTensor).to(output0.device))
    return agree


def validate(val_loader, model, criterion, args=None, epoch=None, gpu=0,
             print_log=False, print_freq=10, print_acc=True, tb_writer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if args:
        gpu = args.gpu
        print_freq = args.print_freq
        tb_writer = args.writer

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if gpu is not None:
                input = input.cuda(gpu, non_blocking=True)
            target = target.cuda(gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if print_log and i % print_freq == 0:
                logging.info('Test: [{0}/{1}]\t'
                             'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                             'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                             'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                             'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

        if print_acc:
            logging.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                         .format(top1=top1, top5=top5))

        if tb_writer and epoch:
            tb_writer.add_scalar("Acc/val", top1.avg, epoch)

    return top1.avg


def validate_shift(val_loader, model, epochs_shift=5, gpu=0, print_log=False, print_freq=100, name=None, D=4):
    batch_time = AverageMeter()
    consist = AverageMeter()

    if name:
        ep_shift_range = trange(epochs_shift, desc=name)
        iter_range = enumerate(tqdm(val_loader, leave=False))
    else:
        ep_shift_range = range(epochs_shift)
        iter_range = enumerate(val_loader)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for ep in ep_shift_range:
            for i, (input, target) in iter_range:
                if gpu is not None:
                    input = input.cuda(gpu, non_blocking=True)

                orig_size = input.shape[-1] - D

                off0 = np.random.randint(D, size=2)
                off1 = np.random.randint(D, size=2)

                output0 = model(input[:, :, off0[0]:off0[0] + orig_size, off0[1]:off0[1] + orig_size])
                output1 = model(input[:, :, off1[0]:off1[0] + orig_size, off1[1]:off1[1] + orig_size])

                cur_agree = agreement(output0, output1).type(torch.FloatTensor).to(output0.device)

                # measure agreement and record
                consist.update(cur_agree.item(), input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if print_log and i % print_freq == 0:
                    logging.info('Ep [{0}/{1}]:\t'
                                 'Test: [{2}/{3}]\t'
                                 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                 'Consist {consist.val:.4f} ({consist.avg:.4f})\t'.format(
                        ep, epochs_shift, i, len(val_loader), batch_time=batch_time, consist=consist))
        if print_log:
            logging.info(' * Consistency {consist.avg:.3f}'.format(consist=consist))

    return consist.avg


def validate_diagonal(val_loader, model, out_dir: Path, D=32, name='', gpu=0, print_log=False, print_freq=10):
    batch_time = AverageMeter()
    prob = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    diag_probs = np.zeros((len(val_loader.dataset), D))
    diag_probs2 = np.zeros((len(val_loader.dataset), D))  # save highest probability, not including ground truth
    diag_corrs = np.zeros((len(val_loader.dataset), D))
    diag_preds = np.zeros((len(val_loader.dataset), D))

    it_val_loader = val_loader
    if name:
        it_val_loader = tqdm(val_loader, desc=name, leave=False)

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(it_val_loader):
            if gpu is not None:
                input = input.cuda(gpu, non_blocking=True)
                target = target.cuda(gpu, non_blocking=True)
            
            orig_size = input.shape[-1] - D
            inputs = []
            for off in range(D):
                inputs.append(input[:, :, off:off + orig_size, off:off + orig_size])

            inputs = torch.cat(inputs, dim=0)
            probs = torch.nn.Softmax(dim=1)(model(inputs))
            preds = probs.argmax(dim=1).cpu().data.numpy()
            corrs = preds == target.item()
            outputs = 100. * probs[:, target.item()]

            acc1, acc5 = accuracy(probs, target.repeat(D), topk=(1, 5))

            probs[:, target.item()] = 0
            probs2 = 100. * probs.max(dim=1)[0].cpu().data.numpy()

            diag_probs[i, :] = outputs.cpu().data.numpy()
            diag_probs2[i, :] = probs2
            diag_corrs[i, :] = corrs
            diag_preds[i, :] = preds

            # measure agreement and record
            prob.update(np.mean(diag_probs[i, :]), input.size(0))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if print_log and i % print_freq == 0:
                logging.info(
                    'Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Prob {prob.val:.4f} ({prob.avg:.4f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, prob=prob, top1=top1, top5=top5)
                )

    if print_log:
        logging.info(
            ' * Prob {prob.avg:.3f} Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
            .format(prob=prob, top1=top1, top5=top5)
        )

    # создаем директории и сохраняем    
    out_dir.mkdir(exist_ok=True, parents=True)

    np.save(str(out_dir / 'diag_probs'), diag_probs)
    np.save(str(out_dir / 'diag_probs2'), diag_probs2)
    np.save(str(out_dir / 'diag_corrs'), diag_corrs)
    np.save(str(out_dir / 'diag_preds'), diag_preds)

    return prob.avg, top1.avg, diag_probs


def validate_save(val_loader, mean, std, args):
    import matplotlib.pyplot as plt
    for i, (input, target) in enumerate(val_loader):
        img = (255 * np.clip(input[0, ...].data.cpu().numpy() * np.array(std)[:, None, None] + mean[:, None, None], 0,
                             1)).astype('uint8').transpose((1, 2, 0))
        plt.imsave(args.out_dir / '%05d.png' % i, img)


def agreement(output0, output1):
    pred0 = output0.argmax(dim=1, keepdim=False)
    pred1 = output1.argmax(dim=1, keepdim=False)
    agree = pred0.eq(pred1)
    agree = 100. * torch.mean(agree.type(torch.FloatTensor).to(output0.device))
    return agree
