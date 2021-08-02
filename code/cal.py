# -*- coding: utf-8 -*-
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Created on Sat Sep 19 20:55:56 2015

@author: liangshiyu
"""

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from torchvision.transforms import Compose, Normalize, ToTensor

from calData import testDataIn, testDataOut
from calMetric import metric


def test(nnName, dataName, CUDA_DEVICE, epsilon, temperature):
    # loading neural netowork
    device = (
        torch.device(f"cuda:{CUDA_DEVICE}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    net1 = torch.load("../models/{}.pth".format(nnName), map_location=device)

    # loading data sets
    transform = Compose(
        [
            ToTensor(),
            Normalize(
                (125.3 / 255, 123.0 / 255, 113.9 / 255),
                (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0),
            ),
        ]
    )

    if nnName == "densenet10" or nnName == "wideresnet10":
        testset = CIFAR10(
            root="../data", train=False, download=True, transform=transform
        )
        testloaderIn = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
    elif nnName == "densenet100" or nnName == "wideresnet100":
        testset = CIFAR100(
            root="../data", train=False, download=True, transform=transform
        )
        testloaderIn = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
    else:
        raise NotImplementedError

    if dataName == "Uniform" or dataName == "Gaussian":
        testloaderOut = testloaderIn
    else:
        testsetout = ImageFolder(f"../data/{dataName}", transform=transform)
        testloaderOut = DataLoader(
            testsetout, batch_size=1, shuffle=False, num_workers=2
        )

    # loading training params
    criterion = CrossEntropyLoss()
    if dataName == "Uniform" or dataName == "Gaussian":
        N = 10000
    elif dataName == "iSUN":
        N = 8925
    else:
        N = 1100

    testDataIn(net1, device, criterion, testloaderIn, epsilon, temperature, N)
    testDataOut(net1, device, criterion, testloaderOut, epsilon, temperature, N)
    metric(nnName, dataName)
