import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-dataset', type=str, default='cifar100', help='dataset type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-seed', type=int, default=0, help='random seed')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')

    parser.add_argument('-orig-augs', action='store_true', help='is use orig augs')

    parser.add_argument('-multiply-data', type=int, default=1, help='multiply the number of datasets')
    parser.add_argument('-x2-epoch', action='store_true', help='double the number of epochs')

    parser.add_argument('-use-distil-aug', action='store_true', help='is use distil augmentation loss')
    parser.add_argument('-distil-aug-weight', default=1.0, type=float, help='cross loss weight')
    parser.add_argument('-prob-aug', default=1.0, type=float, help='')
    parser.add_argument('-mode-aug', default="pad", type=str, help='')

    parser.add_argument('-use-cross-loss', action='store_true', help='is use cross samples loss')
    parser.add_argument('-cross-loss-start-epoch', default=0, type=int,
                        help='milestone for start to use cross samples loss')
    parser.add_argument('-only-correct-cross-loss', action='store_true', help='is use cross samples loss')
    parser.add_argument('-cross-loss-weight', default=1.0, type=float, help='cross loss weight')

    parser.add_argument('-use-avg-cross-loss', action='store_true', help='is use cross samples loss')
    parser.add_argument('-avg-cross-loss-start-epoch', default=0, type=int,
                        help='milestone for start to use cross samples loss')
    parser.add_argument('-avg-cross-loss-weight', default=1.0, type=float, help='avg cross loss weight')

    parser.add_argument('-bp-filt-size', type=int, default=None, help='')

    parser.add_argument('-teacher', type=str, default='', help='name of folder with model')
    parser.add_argument('-distil-function', default='l2', type=str, help='distillation function')
    parser.add_argument('-distil-weight', default=0.0, type=float, help='distillation loss weight')
    parser.add_argument('-soft-temper', default=1, type=int, help='soft logits temperature')

    args = parser.parse_args()

    return args
