# pytorch-cifar-100

practice on cifar-100 using pytorch


inspired by [this](https://github.com/kuangliu/pytorch-cifar) repo

## requirements
- python3.5
- pytorch3.1
- tensorflow1.4
- cuda8.0
- cudnnv5

## usage

### 1. enter directory
```bash
$ cd pytorch-cifar-100
```

### 2. run tensorbard
```bash
$ mkdir runs
$ tensorboard --logdir='runs' --port=6006
```
### 3. train the model
```bash
$ python train.py
```
