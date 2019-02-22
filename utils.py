import argparse
import logging
import os

import torch

from config import device


def parse_args():
    parser = argparse.ArgumentParser(description='train DMN+')
    # general
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--end-epoch', type=int, default=256, help='training epoch size.')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint')
    args = parser.parse_args()
    return args


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def get_logger():
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s \t%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

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


def save_checkpoint(epoch, epochs_since_improvement, model, optimizer, acc, is_best):
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'acc': acc,
             'model': model,
             'optimizer': optimizer}
    filename = 'checkpoint.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_checkpoint.tar')


def get_mask(targets):
    batch_size, max_target_len = targets.size()
    mask = torch.ones_like(targets, device=device, dtype=torch.uint8)
    for i in range(batch_size):
        for j in range(max_target_len, 0, -1):
            t = j - 1
            if targets[i, t] == 0:
                mask[i, t] = 0
            else:
                break
    return mask


def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    # print('inp.size(): ' + str(inp.size()))
    # print('target.size(): ' + str(target.size()))
    # print('mask.size(): ' + str(mask.size()))
    crossEntropy = -torch.log(torch.gather(input=inp.squeeze(1), dim=1, index=target.view(-1, 1)))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()


def get_loss(model, images, questions, targets, mask):
    outputs, loss = model.forward(images, questions, targets, mask)
    print('outputs.size(): ' + str(outputs.size()))
    print('targets.size(): ' + str(targets.size()))
    print('outputs: ' + str(outputs))
    print('targets: ' + str(targets))

    reg_loss = 0
    for param in model.parameters():
        reg_loss += 0.001 * torch.sum(param * param)

    corrects = (outputs.data == targets.data).float()
    acc = corrects.masked_select(mask).mean()
    return loss + reg_loss, acc
