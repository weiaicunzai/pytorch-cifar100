import torchvision
import xgboost as xgb
import numpy as np
import random

import sys
import os
# 如果报自己定义的模块报错module not found，则加入这句
sys.path.append(os.getcwd())
""" 训练XGBoost
"""

import argparse

from matplotlib import pyplot as plt

import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from conf import settings
from utils import get_network


def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    # cifar100_test = CIFAR100Test(path, transform=transform_test)
    cifar100_test = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform_test)
    print(f"测试集大小为{len(cifar100_test)}")
    print(f"测试集的文件中原先就是被打乱了的，查看第一个数据{cifar100_test[0]}"
          f"\n图片尺寸为{cifar100_test[0][0].shape}")
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader


def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns:
        train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    # cifar100_training = CIFAR100Train(path, transform=transform_train)
    cifar100_training = torchvision.datasets.CIFAR100(root='../data', train=True, download=True,
                                                      transform=transform_train)
    print(f"训练集大小为{len(cifar100_training)}")
    print(f"训练集的文件中原先就是被打乱了的，查看第一个数据{cifar100_training[0]}, "
          f"\n图片尺寸为{cifar100_training[0][0].shape}")
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader


def train():
    '''训练XGBoost
    '''
    trainDataSetLoader = cifar100_training_loader
    batch_num = len(trainDataSetLoader)
    # 训练模型
    start_time = time.time()
    global model_XGBoost  # 为了更新模型
    for batch_index, (images, labels) in enumerate(trainDataSetLoader):

        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()
        images = net(images)  # 映射出特征向量
        additional_tree = 1  # Number of boosting iterations, 1次boost迭代中，有x个类别就产生x棵树
        ###############调整超参数
        params1 = {
            'tree_method': 'exact',  #
            # XGBoost最后会进行softmax，输出概率最大的下标/标签,而非multi:softprob输出概率矩阵
            'objective': 'multi:softprob',  # 多分类问题,必须得指定这些参数，不然得出的predict会为小数导致十分的不准确
            'num_class': 100,  # 类别数,
            'seed': XGBoostSeed,
            'eta': 0.1
        }

        dtrain = xgb.DMatrix(images, label=labels)
        if batch_index != 0:  # 在原来的基础上继续训练，增量训练
            model_XGBoost = xgb.train(params1, dtrain, num_boost_round=additional_tree, xgb_model=model_XGBoost)
        else:  # 初始化时不能引用空的的xgb_model
            model_XGBoost = xgb.train(params1, dtrain, num_boost_round=additional_tree)
        print("-+-" * 25)
        print(f"当前树有 {len(model_XGBoost.get_dump())} 棵，迭代次数：{batch_index + 1}/{batch_num}")
        # 查看每棵树的具体节点
        # for leaf in model_XGBoost.get_dump():
        #     print(leaf)
        print("-+-" * 25)

    '''需要分批训练（CNN或者XGBoost都是），不然内存会爆满
    train_data = cifar100_training_loader.dataset

    # Warning，列表过大就死机，内存爆满了
    # train_images = torch.tensor([train_data_item[0].numpy() for train_data_item in train_data])

    train_images = [train_data_item[0] for train_data_item in train_data]
    # train_images = torch.cat(train_images, dim=0) # 采用这种堆叠的方式不会增加维度，也就是通道数都加起来了，(3*5000,)
    
    train_images = torch.stack(train_images, dim=0) # 这种才能扩维度，(5000, 3)

    train_labels = [train_data_item[1] for train_data_item in train_data]
    # 9.4GB的剩余内存还是爆满了
    train_X = net(train_images) # 直接使用就可以了    
    model_XGBoost.fit(train_X, train_labels)
    '''
    print(f"XGBoost训练模型使用时间为：{time.time() - start_time} s")


def test():
    '''测试XGBoost模型
    '''
    correct_1, correct_5 = 0, 0
    # 测试模型
    for n_iter, (images, labels) in enumerate(cifar100_test_loader):
        print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(cifar100_test_loader)))
        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()
        images = net(images)  # 映射出特征向量

        # 在测试集上进行预测,使用xgb库提供的DMatrix的格式可以加快运算速度
        images = xgb.DMatrix(images)
        y_pred = model_XGBoost.predict(images)

        # 计算准确率
        y_pred = torch.tensor(y_pred)
        _, y_pred = y_pred.topk(5, 1, largest=True, sorted=True)  # 从大到小并且排序，k=5,dim=1
        labels = labels.view(labels.size(0), -1).expand_as(y_pred)  # 从(16,)变为(16,5),其中一行元素全都相同，方便对比预测值
        correct = y_pred.eq(labels).float()
        # compute top 5
        correct_5 += correct[:, :5].sum()
        # compute top1
        correct_1 += correct[:, :1].sum()
    print("result is as follows:")
    best_accuracy = correct_1 / len(cifar100_test_loader.dataset)
    print(f"The best accuracy is {best_accuracy}")
    print("Top 1 err: ", 1 - best_accuracy)  # 累计10000条测试数据的偏差，给1次猜测机会
    print("Top 5 err: ", 1 - correct_5 / len(cifar100_test_loader.dataset))  # 20个大类，每个大类中有5个相似的小类别，所以给了5次猜测的机会

    print("此时XGBoost的森林如下")
    print("-+-" * 25)
    print(f"当前树有 {len(model_XGBoost.get_dump())} 棵")
    # 查看每棵树的具体节点
    # for leaf in model_XGBoost.get_dump():
    #     print(leaf)
    print("-+-" * 25)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    args = parser.parse_args()

    # 固定各种随机种子
    torch.manual_seed(42)
    XGBoostSeed = 42
    # random.seed(42)
    # np.random.seed(42)

    # XGBoost使用CNN模型提供的变量
    net = get_network(args)
    print(net)

    if args.gpu:
        net.load_state_dict(torch.load(args.weights))
    else:
        net.load_state_dict(torch.load(args.weights, map_location='cpu'))
    net.eval()  # 不启用 BatchNormalization 和 Dropout更新参数，此时使用的是训练好的值。

    with torch.no_grad():
        # data preprocessing:
        cifar100_training_loader = get_training_dataloader(
            settings.CIFAR100_TRAIN_MEAN,
            settings.CIFAR100_TRAIN_STD,
            num_workers=4,
            batch_size=args.b,
            shuffle=True
        )
        cifar100_test_loader = get_test_dataloader(
            settings.CIFAR100_TRAIN_MEAN,
            settings.CIFAR100_TRAIN_STD,
            # settings.CIFAR100_PATH,
            num_workers=4,
            batch_size=args.b,
        )
        # 初始化XGBoost分类器
        model_XGBoost = xgb.XGBClassifier()
        train()  # 训练
        test()  # 测试
