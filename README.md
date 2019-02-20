# Visual Question Answering

![apm](https://img.shields.io/apm/l/vim-mode.svg)

In VQA, an algorithm needs to answer text-based questions about images.

## Dependency
- Python 3.6
- PyTorch 1.0

## Dataset
1. 82,783 training images, 40,504 validation images and 81,434 testing images (images are obtained from [MS COCO website] (https://visualqa.org/download.html))
2. 443,757 questions for training, 214,354 questions for validation and 447,793 questions for testing
3. 4,437,570 answers for training and 2,143,540 answers for validation (10 per question)

```bash
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
wget http://images.cocodataset.org/zips/test2015.zip
```