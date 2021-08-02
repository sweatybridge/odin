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

import time

import torch


def postprocess(inputs, outputs, net1, temper, noiseMagnitude1, criterion):
    # Calculating the confidence of the output, no perturbation added here, no temperature scaling used
    score = outputs.detach().softmax(dim=1).max(dim=1)

    # Calculating the perturbation we need to add, that is,
    # the sign of gradient of cross entropy loss w.r.t. input
    # Using temperature scaling
    scaled = outputs / temper
    labels = score.indices
    loss = criterion(scaled, labels)
    loss.backward()

    # Normalizing the gradient to binary in {-1, 1}
    gradient = torch.ge(inputs.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2
    # print(gradient.shape) = torch.Size([1, 3, 36, 36])
    # Normalizing the gradient to the same space of image
    gradient[0][0] /= 63.0 / 255.0
    gradient[0][1] /= 62.1 / 255.0
    gradient[0][2] /= 66.7 / 255.0
    # Adding small perturbations to images
    tempInputs = torch.add(inputs.data, -noiseMagnitude1, gradient)
    tempInputs.requires_grad_(False)

    with torch.no_grad():
        outputs = net1(tempInputs) / temper
        odin = outputs.softmax(dim=1).max(dim=1)

    # Calculating the confidence after adding perturbations
    return score.values[0].item(), odin.values[0].item()


def testDataIn(net1, device, criterion, testloader10, noiseMagnitude1, temper, N):
    t0 = time.time()
    f1 = open("./softmax_scores/confidence_Base_In.txt", "w")
    g1 = open("./softmax_scores/confidence_Our_In.txt", "w")
    print("Processing in-distribution images")
    ########################################In-distribution###########################################
    for j, data in enumerate(testloader10):
        if j < 1000:
            continue

        inputs, _ = data
        inputs.to(device)
        inputs.requires_grad_()
        outputs = net1(inputs)

        # Calculating the confidence of the output, with and without perturbation
        score, odin = postprocess(
            inputs, outputs, net1, temper, noiseMagnitude1, criterion
        )

        f1.write(f"{temper}, {noiseMagnitude1}, {score}\n")
        g1.write(f"{temper}, {noiseMagnitude1}, {odin}\n")

        if j % 100 == 99:
            print(
                "{:4}/{:4} images processed, {:.1f} seconds used.".format(
                    j + 1 - 1000, N - 1000, time.time() - t0
                )
            )
            t0 = time.time()

        if j == N - 1:
            break

    f1.close()
    g1.close()


def testDataOut(net1, device, criterion, testloader, noiseMagnitude1, temper, N):
    t0 = time.time()
    f2 = open("./softmax_scores/confidence_Base_Out.txt", "w")
    g2 = open("./softmax_scores/confidence_Our_Out.txt", "w")
    print("Processing out-of-distribution images")
    ###################################Out-of-Distributions#####################################
    for j, data in enumerate(testloader):
        if j < 1000:
            continue

        inputs, _ = data
        inputs.to(device)
        inputs.requires_grad_()
        outputs = net1(inputs)

        # Calculating the confidence of the output, with and without perturbation
        score, odin = postprocess(
            inputs, outputs, net1, temper, noiseMagnitude1, criterion
        )
        f2.write(f"{temper}, {noiseMagnitude1}, {score}\n")
        g2.write(f"{temper}, {noiseMagnitude1}, {odin}\n")

        if j % 100 == 99:
            print(
                "{:4}/{:4} images processed, {:.1f} seconds used.".format(
                    j + 1 - 1000, N - 1000, time.time() - t0
                )
            )
            t0 = time.time()

        if j == N - 1:
            break

    f2.close()
    g2.close()
