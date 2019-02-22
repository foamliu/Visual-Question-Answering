import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader

from config import device, hidden_size, print_freq
from data_gen import MsCocoVqaDataset, pad_collate
from models import DMNPlus
from utils import parse_args, get_logger, AverageMeter, save_checkpoint, get_loss, get_mask


def train(dset, model, optim, epoch, logger):
    dset.set_mode('train')
    train_loader = DataLoader(dset, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate)

    model.train()

    losses = AverageMeter()
    accs = AverageMeter()

    for i, data in enumerate(train_loader):
        optim.zero_grad()
        images, questions, answers = data
        images = Variable(images.float().to(device))
        questions = Variable(questions.long().to(device))
        answers = Variable(answers.long().to(device))
        mask = Variable(get_mask(answers))

        loss, acc = get_loss(model, images, questions, answers, mask)
        loss.backward()

        # Keep track of metrics
        losses.update(loss.item())
        accs.update(acc)

        if i % print_freq == 0:
            logger.info(
                '[Epoch {}][{}/{}] [Training]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {accs.val:.3f} ({accs.avg:.3f})'.format(epoch,
                                                                  i,
                                                                  len(train_loader),
                                                                  loss=losses,
                                                                  accs=accs
                                                                  ))
        optim.step()


def valid(dset, model, epoch, logger):
    dset.set_mode('val')
    valid_loader = DataLoader(dset, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate)

    model.eval()

    losses = AverageMeter()
    accs = AverageMeter()

    for batch_idx, data in enumerate(valid_loader):
        images, questions, answers = data
        images = Variable(images.float().to(device))
        questions = Variable(questions.long().to(device))
        answers = Variable(answers.long().to(device))

        loss, acc = get_loss(model, images, questions, answers)

        # Keep track of metrics
        losses.update(loss.item())
        accs.update(acc)

    logger.info('[Epoch {}] [Validate] Accuracy : {:.4f}'.format(epoch, accs.avg))
    return accs.avg


def train_net(args):
    torch.manual_seed(7)
    np.random.seed(7)

    checkpoint = args.checkpoint
    logger = get_logger()

    dset = MsCocoVqaDataset()
    vocab_size = len(dset.QA.VOCAB)

    # Initialize / load checkpoint
    if checkpoint is None:
        start_epoch = 0
        epochs_since_improvement = 0
        model = DMNPlus(hidden_size, vocab_size, num_hop=3, qa=dset.QA)
        # model = nn.DataParallel(model)
        optim = torch.optim.Adam(model.parameters())

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        model = checkpoint['model']
        optim = checkpoint['optimizer']

    model.to(device)

    best_acc = 0

    for epoch in range(start_epoch, args.end_epoch):
        train(dset, model, optim, epoch, logger)

        valid_acc = valid(dset, model, epoch, logger)
        is_best = valid_acc > best_acc
        best_acc = max(valid_acc, best_acc)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, model, optim, best_acc, is_best)


def main():
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()
