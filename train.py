import torch
import logging
from augmentation import transformation
from layers import NASConv1d
from memory_cost_profiler import profile_memory_cost


class AverageMeter(object):
    """
	Computes and stores the average and current value
	Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
	"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

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
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)
    target = torch.argmax(target, dim=1)
    # print(target)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    # correct = target

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train_wo_arch(model, dataloader, optimizer, criterion):
    objs = AverageMeter()
    top1 = AverageMeter()

    model.train()
    for param in model.weight_params():
        param.requires_grad = True
    for param in model.arch_params():
        param.requires_grad = False

    for step, batch in enumerate(dataloader):
        x, y = batch[0].to(model.device), batch[1].to(model.device)
        x.unsqueeze(1)

        pred_gumbel = model(x, sampling=True, mode='gumbel')
        loss = criterion(pred_gumbel, y)
        model.reset_switches()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prec1 = accuracy(pred_gumbel, y)
        n = x.shape[0]
        objs.update(loss.item(), n)
        top1.update(prec1[0].item(), n)

        if step % 100 == 0:
            logging.info('TRAIN wo_Arch Step: %04d Objs_W: %f R1: %f', step, objs.avg, top1.avg)
    return top1.avg


def train_w_arch(model, train_loader, valid_loader, optimizer_w, optimizer_a, criterion):
    objs = AverageMeter()
    top1 = AverageMeter()

    model.train()

    for step, batch in enumerate(train_loader):
        for param in model.weight_params():
            param.requires_grad = True
        for param in model.arch_params():
            param.requires_grad = False

        x, y = batch[0].to(model.device), batch[1].to(model.device)
        x.unsqueeze(1)
        pred_gumbel, _ = model(x, sampling=True, mode='gumbel')
        loss_gumbel = criterion(pred_gumbel, y)

        pred_random, _ = model(x, sampling=True, mode='random')
        loss_random = criterion(pred_random, y)

        loss = loss_gumbel + loss_random

        optimizer_w.zero_grad()
        loss.backward()
        optimizer_w.step()

        if step % 2 == 0:
            try:
                x_val, y_val = next(valid_loader)
            except:
                val_queue_iter = iter(valid_loader)
                x_val, y_val = next(val_queue_iter)

            x_val = x_val.to(model.device, non_blocking=True)
            y_val = y_val.to(model.device, non_blocking=True)

            for param in model.weight_params():
                param.requires_grad = False
            for param in model.arch_params():
                param.requires_grad = True

            pred, mem_total = model(x_val, sampling=False)
            print(mem_total)
            loss_a = criterion(pred, y_val)
            loss_m = 0
            objs.update(loss_a.item())

            prec1 = accuracy(pred, y_val)
            top1.update(prec1[0].item())

            optimizer_a.zero_grad()
            loss_a.backward()
            optimizer_a.step()

        if step % 100 == 0:
            logging.info('TRAIN w_Arch Step: %04d Objs_W: %f R1: %f', step, objs.avg, top1.avg)
    return top1.avg
