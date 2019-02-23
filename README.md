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
|![image](https://github.com/foamliu/Visual-Question-Answering/raw/master/images/0_img.png)|提问：这是花吗？<br>标准答案：这是花。<br>电脑抢答：是的。|![image](https://github.com/foamliu/Visual-Question-Answering/raw/master/images/1_img.png)|提问：这个人在干什么？<br>标准答案：在吃饭。<br>电脑抢答：这个人在玩滑板|
|![image](https://github.com/foamliu/Visual-Question-Answering/raw/master/images/2_img.png)|提问：他是爱玩滑板吗？<br>标准答案：是的。<br>电脑抢答：是的。|![image](https://github.com/foamliu/Visual-Question-Answering/raw/master/images/3_img.png)|提问：这是什么交通工具？<br>标准答案：公共汽车。<br>电脑抢答：火车。|
|![image](https://github.com/foamliu/Visual-Question-Answering/raw/master/images/4_img.png)|提问：微波炉是白色的吗？<br>标准答案：是的。<br>电脑抢答：是的。|![image](https://github.com/foamliu/Visual-Question-Answering/raw/master/images/5_img.png)|提问：这是什么动物？<br>标准答案：这是绵羊。<br>电脑抢答：这是。|
|![image](https://github.com/foamliu/Visual-Question-Answering/raw/master/images/6_img.png)|提问：他们的制服是什么颜色的？<br>标准答案：黑色的。<br>电脑抢答：这是是白色的|![image](https://github.com/foamliu/Visual-Question-Answering/raw/master/images/7_img.png)|提问：这是哪里？<br>标准答案：这是英国。<br>电脑抢答：这是。|
|![image](https://github.com/foamliu/Visual-Question-Answering/raw/master/images/8_img.png)|提问：这是什么食物？<br>标准答案：披萨。<br>电脑抢答：这是。|![image](https://github.com/foamliu/Visual-Question-Answering/raw/master/images/9_img.png)|提问：图中有几个人物？<br>标准答案：1个。<br>电脑抢答：1个。|






## Reference
1. [Dynamic Memory Network for Visual and Textual Question Answering](https://arxiv.org/abs/1603.01417).