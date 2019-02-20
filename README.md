# Visual Question Answering

![apm](https://img.shields.io/apm/l/vim-mode.svg)

In VQA, an algorithm needs to answer text-based questions about images. This is an PyTorch implementation of DMN+ model on MSCOCO VQA dataset.

## Dependency
- Python 3.6
- PyTorch 1.0

## Dataset
1. MSCOCO: images are obtained from [MS COCO website](https://visualqa.org/download.html). 
    - 82,783 training images, 
    - 40,504 validation images and, 
    - 81,434 testing images.

2. Baidu: chinese questions and answers are obtained from [Baidu Research](http://idl.baidu.com/FM-IQA.html).
    - 164,735 training questions and answers, 
    - 75,206 validation questions and answers.


Simply create a "data" folder then run:
```bash
$ wget http://images.cocodataset.org/zips/train2014.zip
$ wget http://images.cocodataset.org/zips/val2014.zip
$ wget http://images.cocodataset.org/zips/test2015.zip
$ wget http://research.baidu.com/Public/uploads/5ac9e10bdd572.gz
$ tar -xvzf 5ac9e10bdd572.gz
```

## Usage

### Data wraggling
Extract and pro_processing training data：
```bash
$ python extract.py
$ python pre_process.py
```

### Train
```bash
$ python train.py
```

### Demo
Download pretrained [model](https://github.com/foamliu/Reading-Comprehension/releases/download/v1.0/BEST_checkpoint.tar) under "models" folder then run:

```bash
$ python demo.py
```
