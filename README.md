# Visual Question Answering

![apm](https://img.shields.io/apm/l/vim-mode.svg)

This is an PyTorch implementation of DMN+ model on MSCOCO VQA dataset. In VQA, an algorithm needs to answer text-based questions about images. 

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

### Data wrangling
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

图片|问答|图片|问答|
|---|---|---|---|
|![image](https://github.com/foamliu/Visual-Question-Answering/raw/master/images/0_img.png)|提问：这是在哪里？<br>标准答案：这是在海边。<br>电脑抢答：这是在海边。|![image](https://github.com/foamliu/Visual-Question-Answering/raw/master/images/1_img.png)|提问：撑伞的女人穿什么颜色的衣服？<br>标准答案：红色。<br>电脑抢答：白色。|
|![image](https://github.com/foamliu/Visual-Question-Answering/raw/master/images/2_img.png)|提问：这个人在做什么？<br>标准答案：在吃东西。<br>电脑抢答：这个人在|![image](https://github.com/foamliu/Visual-Question-Answering/raw/master/images/3_img.png)|提问：这是一只鸟吗？<br>标准答案：是的。<br>电脑抢答：是的。|
|![image](https://github.com/foamliu/Visual-Question-Answering/raw/master/images/4_img.png)|提问：这是哪里？<br>标准答案：这是洗脸池和沐浴间。<br>电脑抢答：这是卫生间。|![image](https://github.com/foamliu/Visual-Question-Answering/raw/master/images/5_img.png)|提问：小孩在什么地方？<br>标准答案：在床上。<br>电脑抢答：床上。|
|![image](https://github.com/foamliu/Visual-Question-Answering/raw/master/images/6_img.png)|提问：这只猫是什么颜色？<br>标准答案：黑色。<br>电脑抢答：这只是黑色的。|![image](https://github.com/foamliu/Visual-Question-Answering/raw/master/images/7_img.png)|提问：他在干什么？<br>标准答案：他在弹吉他。<br>电脑抢答：玩游戏。|
|![image](https://github.com/foamliu/Visual-Question-Answering/raw/master/images/8_img.png)|提问：这里有几个人？<br>标准答案：3个人。<br>电脑抢答：这里有一个人。|![image](https://github.com/foamliu/Visual-Question-Answering/raw/master/images/9_img.png)|提问：大象在干什么？<br>标准答案：大象在吃东西。<br>电脑抢答：在在。。|






## Reference
1. [Dynamic Memory Network for Visual and Textual Question Answering](https://arxiv.org/abs/1603.01417).