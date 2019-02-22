import random

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
import cv2 as cv
import numpy as np
from config import batch_size, device
from data_gen import MsCocoVqaDataset
from data_gen import pad_collate
from utils import ensure_folder

if __name__ == '__main__':
    checkpoint = 'BEST_checkpoint.tar'
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    model.eval()

    dset = MsCocoVqaDataset()
    dset.set_mode('val')
    valid_loader = DataLoader(dset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate)

    chosen_samples = range(len(dset))
    _ids = random.sample(chosen_samples, 10)

    _pred_ids = []

    for i, data in enumerate(valid_loader):
        imgs, questions, targets = data
        imgs = Variable(imgs.to(device))
        questions = Variable(questions.to(device))
        targets = Variable(targets.to(device))

        max_target_len = targets.size()[1]
        outputs = model.forward(imgs, questions, max_target_len)
        preds = F.softmax(outputs, dim=-1)
        _, pred_ids = torch.max(preds, dim=1)
        _pred_ids += list(pred_ids.cpu().numpy())

    print('len(_pred_ids): ' + str(len(_pred_ids)))

    ensure_folder('images')

    for i, id in enumerate(_ids):
        img = dset[id][0]
        img = img * 128 + 127.5
        question = dset[id][1]
        question = ''.join([dset.QA.IVOCAB[id] for id in question]).replace('<EOS>', '')
        target = dset[id][2]
        target = ''.join([dset.QA.IVOCAB[id] for id in target]).replace('<EOS>', '')

        pred = _pred_ids[id]
        pred = ''.join([dset.QA.IVOCAB[id] for id in pred]).replace('<EOS>', '')

        img = img.astype(np.uint8)
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        filename = 'images/{}_img.png'.format(i)
        cv.imwrite(filename, img)

        print('提问：' + question)
        print('标准答案：' + target)
        print('电脑抢答：' + pred)
        print()
