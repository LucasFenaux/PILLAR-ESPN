#!/usr/bin/env python3

from math import prod

import torch

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import crypten
from crypten.cuda import CUDALongTensor


def zeros(size, bitlength=None, generator=None, device=torch.device('cpu'), **kwargs):
    rand_element = torch.zeros(size, device=device).to(torch.int64)
    if rand_element.is_cuda:
        return CUDALongTensor(rand_element)
    return rand_element


def ones(size, bitlength=None, generator=None, device=torch.device('cpu'), **kwargs):
    rand_element = torch.ones(size, device=device).to(torch.int64)
    if rand_element.is_cuda:
        return CUDALongTensor(rand_element)
    return rand_element


def elements(size):
    return prod(list(size))


def generate_random_ring_element(size, ring_size=(2 ** 64), generator=None, **kwargs):
    """Helper function to generate a random number from a signed ring"""
    if generator is None:
        device = kwargs.get("device", torch.device("cpu"))
        device = torch.device("cpu") if device is None else device
        device = torch.device(device) if isinstance(device, str) else device
        generator = crypten.generators["local"][device]
    # TODO (brianknott): Check whether this RNG contains the full range we want.
    rand_element = torch.randint(
        -(ring_size // 2),
        (ring_size - 1) // 2,
        size,
        generator=generator,
        dtype=torch.long,
        **kwargs,
    )
    if rand_element.is_cuda:
        return CUDALongTensor(rand_element)
    return rand_element


def generate_kbit_random_tensor(size, bitlength=None, generator=None, **kwargs):
    """Helper function to generate a random k-bit number"""
    if bitlength is None:
        bitlength = torch.iinfo(torch.long).bits
    if bitlength == 64:
        return generate_random_ring_element(size, generator=generator, **kwargs)
    if generator is None:
        device = kwargs.get("device", torch.device("cpu"))
        device = torch.device("cpu") if device is None else device
        device = torch.device(device) if isinstance(device, str) else device
        generator = crypten.generators["local"][device]
    rand_tensor = torch.randint(
        0, 2 ** bitlength, size, generator=generator, dtype=torch.long, **kwargs
    )
    if rand_tensor.is_cuda:
        return CUDALongTensor(rand_tensor)
    return rand_tensor
