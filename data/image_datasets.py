"""
Using same transormations as "Bayesian Deep Learning via Subnetwork Inference"
Based on code from https://github.com/edaxberger/subnetwork-inference
"""

import torch
import numpy as np


from PIL import Image
from torchvision import transforms, datasets

DATA_PATH = "Image"

DATA_META = {
    'MNIST': {
       'channels': 1,
       'classes':  10
    },
    'Fashion': {
        'channels': 1,
        'classes':  10
    },
    'SVHN': {
        'channels': 3,
        'classes':  10
    },
    'CIFAR10': {
        'channels': 3,
        'classes':  10
    }
}

def get_image_loader(dataset, batch_size, workers, data_path=DATA_PATH, gpu=False, distributed=False):

    assert dataset in DATA_META.keys()

    if dataset == 'MNIST':
        train_transform = transforms.Compose([
            transforms.RandomCrop(28, padding=2),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ])

        test_trainsform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ])
        train_dataset = datasets.MNIST(root=data_path,
                                       train=True,
                                       download=True,
                                       transform=train_transform)
        
        test_dataset = datasets.MNIST(root=data_path,
                                      train=False,
                                      download=True,
                                      transform=test_trainsform)

    elif dataset == 'Fashion':
        train_transform = transforms.Compose([
            transforms.RandomCrop(28, padding=2),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.2860,), std=(0.3530,))
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.2860,), std=(0.3530,))
        ])

        train_dataset = datasets.FashionMNIST(root=data_path,
                                              train=True,
                                              download=True,
                                              transform=train_transform)
        test_dataset = datasets.FashionMNIST(root=data_path,
                                            train=False,
                                            download=True,
                                            transform=test_transform)
       

    elif dataset == 'SVHN':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])

        train_dataset = datasets.SVHN(root=data_path,
                                      split='train',
                                      download=True,
                                      transform=train_transform)

        test_dataset = datasets.SVHN(root=data_path,
                                     split='test',
                                     download=True,
                                     transform=test_transform)
      

    elif dataset == 'CIFAR10':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])

        train_dataset = datasets.CIFAR10(root=data_path,
                                         train=True,
                                         download=True,
                                         transform=train_transform)

        test_dataset = datasets.CIFAR10(root=data_path,
                                        train=False,
                                        download=True,
                                        transform=test_transform)
       
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        sampler = None
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=sampler is None,
                                               num_workers=workers,
                                               pin_memory=gpu,
                                               sampler=sampler)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=workers,
                                              pin_memory=gpu)

    info = {
        'n_train': len(train_dataset),
        'n_test': len(test_dataset),
        'n_channels': DATA_META[dataset]['channels'],
        'n_classes' : DATA_META[dataset]['classes']
    }
   
    return train_loader, test_loader, info


def rotate_load_dataset(dataset, angle, batch_size, workers, data_path=DATA_PATH, gpu=False):
    assert dataset in DATA_META.keys()

    if dataset == 'MNIST':
        test_trainsform = transforms.Compose([
            transforms.RandomRotation([angle, angle], expand=False, center=None),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ])
        
        test_dataset = datasets.MNIST(root=data_path,
                                      train=False,
                                      download=True,
                                      transform=test_trainsform)

    elif dataset == 'Fashion':
        test_transform = transforms.Compose([
            transforms.RandomRotation([angle, angle], expand=False, center=None),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.2860,), std=(0.3530,))
        ])

        test_dataset = datasets.FashionMNIST(root=data_path,
                                            train=False,
                                            download=True,
                                            transform=test_transform)
       

    elif dataset == 'SVHN':
        test_transform = transforms.Compose([
            transforms.RandomRotation([angle, angle], expand=False, center=None),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])

        test_dataset = datasets.SVHN(root=data_path,
                                     split='test',
                                     download=True,
                                     transform=test_transform)
      

    elif dataset == 'CIFAR10':
        test_transform = transforms.Compose([
            transforms.RandomRotation([angle, angle], expand=False, center=None),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])

        test_dataset = datasets.CIFAR10(root=data_path,
                                        train=False,
                                        download=True,
                                        transform=test_transform)
       

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=workers,
                                              pin_memory=gpu)

    info = {
        'n_test': len(test_dataset),
        'rotation': angle,
        'n_channels': DATA_META[dataset]['channels'],
        'n_classes' : DATA_META[dataset]['classes']
    }
   
    return test_loader, info
