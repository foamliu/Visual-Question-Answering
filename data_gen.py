import pickle

import cv2 as cv
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from config import im_size, pickle_file


class adict(dict):
    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self


def pad_collate(batch):
    max_question_len = float('-inf')
    max_answer_len = float('-inf')
    for elem in batch:
        _, question, answer = elem
        max_question_len = max_question_len if max_question_len > len(question) else len(question)
        max_answer_len = max_answer_len if max_answer_len > len(answer) else len(answer)

    for i, elem in enumerate(batch):
        image, question, answer = elem
        question = np.pad(question, (0, max_question_len - len(question)), 'constant', constant_values=0)
        answer = np.pad(answer, (0, max_answer_len - len(answer)), 'constant', constant_values=0)
        batch[i] = (image, question, answer)
    return default_collate(batch)


# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class MsCocoVqaDataset(Dataset):
    def __init__(self, mode='train'):
        self.mode = mode
        self.QA = adict()
        with open(pickle_file, 'rb') as file:
            data = pickle.load(file)

        self.QA.VOCAB = data['VOCAB']
        self.QA.IVOCAB = data['IVOCAB']
        self.train = data['train']
        self.val = data['val']
        
        if mode == 'train':
            self.transformer = data_transforms['train']

    def set_mode(self, mode):
        self.mode = mode

    def __len__(self):
        if self.mode == 'train':
            return len(self.train[0])
        elif self.mode == 'val':
            return len(self.val[0])

    def __getitem__(self, index):
        if self.mode == 'train':
            images, questions, answers = self.train
            prefix = 'data/train2014/COCO_train2014_0000'

        else:  # self.mode == 'val':
            images, questions, answers = self.val
            prefix = 'data/val2014/COCO_val2014_0000'

        image_id = int(images[index])
        image_id = '{:08d}'.format(image_id)
        filename = prefix + image_id + '.jpg'
        img = cv.imread(filename)
        img = cv.resize(img, (im_size, im_size))
        img = transforms.ToPILImage()(img)
        img = self.transformer(img)

        question = questions[index]
        answer = answers[index]

        return img, question, answer


if __name__ == '__main__':
    dset_train = MsCocoVqaDataset()
    train_loader = DataLoader(dset_train, batch_size=2, shuffle=True, collate_fn=pad_collate)
    for batch_idx, data in enumerate(train_loader):
        images, questions, answers = data
        print('answers.size(): ' + str(answers.size()))
        break
    print(len(dset_train.QA.VOCAB))
