# Visual Question Answering

![apm](https://img.shields.io/apm/l/vim-mode.svg)

In VQA, an algorithm needs to answer text-based questions about images.

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