# Pytorch-cifar100

In this work, I use a novel approach to address GPU memory constraints in the implementation of the ResNet50 model for image classification on the CIFAR-100 dataset. By employing the Group Pruner technique, I selectively prune redundant filters in ResNet50, significantly reducing its memory usage while preserving comparable accuracy. Additionally, I leverage knowledge distillation to transfer the knowledge from the base model (ResNet50) to a smaller ResNet18 model, enabling us to obtain a memory-efficient architecture without sacrificing performance. My experimental results demonstrate that the combination of Group Pruner and knowledge distillation provides an effective solution to the challenges of deep CNN architecture deployment on resource-limited platforms, making it feasible to utilize sophisticated models like ResNet50 in real-world applications with memory constraints.

## Results
The result I can get from a certain model, since I use the same hyperparameters to train all the networks, some networks might not get the best result from these hyperparameters, you could try yourself by finetuning the hyperparameters to get
better result.

|Dataset|Network|Params|Inference time (ms)|Runtime Memory on CPU (ms)||Runtime Memory on CUDA (ms)|Top-1 Error|Top-5 Error|Checkpoint path|
|:-----:|:-----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|Cifar100|Resnet50 (base model)|23705252|167|15.933|6.481|21.24|5.33|?|
|Cifar100|Resnet50 (pruned model)|14959153|146|15.538|5.456|27.04|8.05|?|
|Cifar100|Resnet18 (KD from Resnet50)|11220132|61|7.019|2.735|23.95|7.01|?|


## Requirements

This is my experiment eviroment
- Ubuntu 20.04.6 LTS
- Python 3.8
- PyTorch 1.13.1+cu117
- CuDNN 8500

## Usage

### 1. Enter directory
```bash
$ cd pytorch-cifar100
```

### 2. Dataset
I will use cifar100 dataset from torchvision since it's more convenient, but I also kept the sample code for writing your own dataset module in dataset folder, as an example for people don't know how to write it.

### 3. Train the model
You need to specify the net you want to train using arg -net
```bash
$ python train.py -net resnet50 -gpu
```

sometimes, you might want to use warmup training by set ```-warm``` to 1 or 2, to prevent network
diverge during early training phase.

The supported net args are:
```
squeezenet
mobilenet
mobilenetv2
shufflenet
shufflenetv2
vgg11
vgg13
vgg16
vgg19
densenet121
densenet161
densenet201
googlenet
inceptionv3
inceptionv4
inceptionresnetv2
xception
resnet18
resnet34
resnet50
resnet101
resnet152
preactresnet18
preactresnet34
preactresnet50
preactresnet101
preactresnet152
resnext50
resnext101
resnext152
attention56
attention92
seresnet18
seresnet34
seresnet50
seresnet101
seresnet152
nasnet
wideresnet
stochasticdepth18
stochasticdepth34
stochasticdepth50
stochasticdepth101
```
Normally, the weights file with the best accuracy would be written to the disk with name suffix 'best'(default in checkpoint folder).


### 4. Test the model
Test the model using test.py
```bash
$ python test.py -net resnet50 -weights path_to_resnet50_weights_file 
```
For this work, you can easily test the model by the trained base resnet50 model by using
```bash
$ python test.py -net resnet50 -weights checkpoint/resnet50/Saturday_29_July_2023_05h_48m_46s/resnet50-200-regular.pth
```


### 5. Train the pruned model
Train the pruned model using train_prune.py. Considering an exist resnet50_weights_file is sparse model
```bash
$ python train_prune.py -net resnet50 -sl_weights path_to_sparse_resnet50_weights_file
```
For this work, you can train the model by the sparse resnet50 model by using
```bash
$ python train_prune.py -net resnet50 -sl_weights checkpoint/resnet50/Saturday_29_July_2023_05h_48m_46s/resnet50-200-regular.pth
```


### 6. Test the pruned model
Train the pruned model using train.py. Considering an exist resnet50_weights_file is sparse model
```bash
$ python test_prune.py -net resnet50 -sl_weights path_to_sparse_resnet50_weights_file -weights path_to_optimized_resnet50_weights_file
```
For this work, you can easily test the model by the trained pruned resnet50 model by using
```bash
$ python test_prune.py -net resnet50 -sl_weights checkpoint/resnet50/Saturday_29_July_2023_05h_48m_46s/resnet50-200-regular.pth -weights checkpoint/resnet50/Sunday_30_July_2023_11h_34m_52s/resnet50-200-regular.pth
```
### 7. Train the KD model
Train the pruned model using train_prune.py. Considering an exist resnet50_weights_file is sparse model
```bash
$ python train_KD.py -net-teacher resnet50 -teacher-weights path_to_teacher_resnet50_weights_file -net-student resnet18
```
For this work, you can train the model by the teacher resnet50 model by using
```bash
$ python train_KD.py -net-teacher resnet50 -teacher-weights checkpoint/resnet50/Saturday_29_July_2023_05h_48m_46s/resnet50-200-regular.pth -net-student resnet18
```

