import json
import pickle

import jieba
from tqdm import tqdm

from config import qa_json, pickle_file


class adict(dict):
    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self


def seg_line(line):
    return list(jieba.cut(line))


def get_unindexed_qa(raw_data):
    data = []

    for item in raw_data:
        question = item['Question']
        answer = item['Answer']
        image_id = item['image_id']
        data.append({'Q': question, 'A': answer, 'I': image_id})
    return data


def get_indexed_qa(raw_data):
    print('get indexed qa...')
    unindexed = get_unindexed_qa(raw_data)
    questions = []
    answers = []
    images = []
    for qa in tqdm(unindexed):
        question = seg_line(qa['Q']) + ['<EOS>']
        for token in question:
            build_vocab(token)
        question = [QA.VOCAB[token] for token in question]

        answer = seg_line(qa['A']) + ['<EOS>']
        for token in answer:
            build_vocab(token)
        answer = [QA.VOCAB[token] for token in answer]

        images.append(qa['I'])
        questions.append(question)
        answers.append(answer)
    return images, questions, answers


def build_vocab(token):
    if not token in QA.VOCAB:
        next_index = len(QA.VOCAB)
        QA.VOCAB[token] = next_index
        QA.IVOCAB[next_index] = token


if __name__ == '__main__':
    with open(qa_json, 'r') as file:
        raw_data = json.load(file)

    QA = adict()
    QA.VOCAB = {'<PAD>': 0, '<EOS>': 1, '<SOS>': 2}
    QA.IVOCAB = {0: '<PAD>', 1: '<EOS>', 2: '<SOS>'}
    data = dict()
    data['VOCAB'] = QA.VOCAB
    data['IVOCAB'] = QA.IVOCAB
    data['train'] = get_indexed_qa(raw_data['train'])
    data['val'] = get_indexed_qa(raw_data['val'])
    with open(pickle_file, 'wb') as file:
        pickle.dump(data, file)
