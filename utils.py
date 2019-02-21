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


def maskNLLLoss(outputs, targets):
    print('outputs.size(): ' + str(outputs.size()))
    print('targets.size(): ' + str(targets.size()))
    batch_size, max_target_len = targets.size()

    mask = torch.ones_like(targets)
    for i in range(batch_size):
        for t in range(max_target_len, 0, -1):
            if targets[i, t] == 0:
                mask[i, t] = 0
            else:
                break

    loss = 0
    n_totals = 0
    for t in range(0, max_target_len):
        nTotal = mask.sum()
        crossEntropy = -torch.log(torch.gather(input=outputs, dim=1, index=targets.view(-1, 1)))
        mask_loss = crossEntropy.masked_select(mask).mean()
        mask_loss = mask_loss.to(device)
        loss += mask_loss
        n_totals += nTotal

    return loss, n_totals
