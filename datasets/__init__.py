from .cifar100 import get_cifar100_dataloaders
from .tiny_imagenet import get_tiny_imagenet_dataloaders

__all__ = ["cifar100", "tiny_imagenet"]


def get_dataloaders(args):

    if args.dataset == "cifar100":
        return get_cifar100_dataloaders(args)
    elif args.dataset == "tiny_imagenet":
        return get_tiny_imagenet_dataloaders(args)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
