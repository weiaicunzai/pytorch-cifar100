# pytorch-cifar-100

practice on cifar-100 using pytorch


inspired by [this](https://github.com/kuangliu/pytorch-cifar) repo

## Requirements
- python3.5
- pytorch3.1
- tensorflow1.4
- cuda8.0
- cudnnv5

## Usage

### 1. enter directory
```bash
$ cd pytorch-cifar-100
```

### 2. run tensorbard
```bash
$ tensorboard --logdir='runs' --port=6006
```
### 3. train the model
```bash
$ python train.py
```
### 4. training
I train model for 140 epoch
set learning rate at:
- epoch < 60, lr = 0.1
- epoch < 100, lr = 0.01
- epoch < 140, lr = 0.001

I found that training more epoch when lr = 0.1 can improve
my model prformance by %1 or %2, but add more epoch at lr = 0.01
or lr = 0.001 own't make much difference.So I decide to train my
model for more epoch whye lr = 0.1

### 5. results

|dataset|network|params|top1 err|top5 err|
|:---:|:---:|:---:|:---:|:---:|
|cifar100|resnet101|42.7M|22.22|5.61|