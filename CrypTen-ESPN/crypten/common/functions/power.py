#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

import crypten
import crypten.communicator as comm
from crypten.config import cfg
from ..tensor_types import is_tensor

__all__ = ["norm", "polynomial", "pos_pow", "pow", "crypten_poly",
           "espn_poly", "espn_pow",
           "honeybadger_poly"]

from ...cuda import CUDALongTensor


def pow(self, p, **kwargs):
    """
    Computes an element-wise exponent `p` of a tensor, where `p` is an
    integer.
    """
    if isinstance(p, float) and int(p) == p:
        p = int(p)

    if not isinstance(p, int):
        raise TypeError(
            "pow must take an integer exponent. For non-integer powers, use"
            " pos_pow with positive-valued base."
        )
    if p < -1:
        return self.reciprocal().pow(-p)
    elif p == -1:
        return self.reciprocal()
    elif p == 0:
        # Note: This returns 0 ** 0 -> 1 when inputs have zeros.
        # This is consistent with PyTorch's pow function.
        return self.new(torch.ones_like(self.data))
    elif p == 1:
        return self.clone()
    elif p == 2:
        return self.square()
    elif p % 2 == 0:
        return self.square().pow(p // 2)
    else:
        x = self.square().mul_(self)
        return x.pow((p - 1) // 2)


def pos_pow(self, p):
    """
    Approximates self ** p by computing: :math:`x^p = exp(p * log(x))`

    Note that this requires that the base `self` contain only positive values
    since log can only be computed on positive numbers.

    Note that the value of `p` can be an integer, float, public tensor, or
    encrypted tensor.
    """
    if isinstance(p, int) or (isinstance(p, float) and int(p) == p):
        return self.pow(p)
    return self.log().mul_(p).exp()


def crypten_poly(self, coeffs, func="mul"):
    """Computes a polynomial function on a tensor with given coefficients,
    `coeffs`, that can be a list of values or a 1-D tensor.

    Coefficients should be ordered from the order 1 (linear) term first,
    ending with the highest order term. (Constant is not included).
    """
    # Compute terms of polynomial using exponentially growing tree
    terms = crypten.stack([self, self.square()])
    while terms.size(0) < coeffs.size(0):
        highest_term = terms.index_select(
            0, torch.tensor(terms.size(0) - 1, device=self.device)
        )
        new_terms = getattr(terms, func)(highest_term)
        terms = crypten.cat([terms, new_terms])

    # Resize the coefficients for broadcast
    terms = terms[: coeffs.size(0)]
    for _ in range(terms.dim() - 1):
        coeffs = coeffs.unsqueeze(1)

    # Multiply terms by coefficients and sum
    return terms.mul(coeffs).sum(0)


def get_ncr(degree, scale, coeffs):
    nCr = []
    for k in range(degree + 1):
        nCr += [coeffs[k] * scale ** (degree + 1 - k)]
        for i in range(k):
            nCr += [nCr[-1] * (k - i) / (i + 1)]
    nCr = torch.tensor(nCr).unsqueeze(1)
    return nCr


def espn_pow(self, k):
    """Perform element-wise exponentiation by constant k"""
    assert comm.get().get_world_size() == 2, "Exponentation only supported for two parties"

    shape = self._tensor.shape
    self._tensor = self._tensor.reshape((-1,))

    nCr = [1]
    for i in range(k):
        nCr += [nCr[-1] * (k - i) / (i + 1)]
    nCr = torch.tensor(nCr, device=self.device).unsqueeze(1)

    a_s = torch.stack([self.share ** i for i in range(k + 1)])
    a = crypten.cryptensor(a_s, precision=0, src=0, device=self.device, ptype=crypten.mpc.arithmetic)

    b_s = torch.stack([self.share ** (k - i) for i in range(k + 1)])
    b = crypten.cryptensor(b_s, precision=0, src=1, device=self.device, ptype=crypten.mpc.arithmetic)
    scale = self.encoder.scale

    ans = (a * b * nCr).sum(dim=0) / (scale ** (k - 1))
    ans.encoder = self.encoder
    ans._tensor = ans._tensor.reshape(shape)
    return ans


def espn_poly(self, coeffs):
    """Perform element-wise polynomial evaluation using fast_pow"""
    assert comm.get().get_world_size() == 2, "Exponentation only supported for two parties"
    # coeffs = torch.cat([torch.tensor([0.0], device=self.device), coeffs])
    """Info and preprocessed"""
    # with Timer("-----------\nPreparation"):
    shape = self.share.shape
    self.share = self.share.reshape((-1,))
    degree = coeffs.size(0) - 1
    scale = self.encoder.scale
    nCr = get_ncr(degree, scale, coeffs).to(self.device)

    """Power computation"""
    # with Timer("Powers"):
    # powers = [torch.ones_like(self.share), self.share]
    # powers += [torch.pow(self.share, i) for i in range(2, degree + 1)]
    exps = torch.tensor(range(degree + 1), device=self.device).unsqueeze(1)
    powers = self.share.unsqueeze(0).repeat(degree + 1, 1)
    powers = powers ** exps

    """Power replication"""
    # with Timer("Replication"):
    # ab_s = []
    # for k in range(degree + 1):
    #     ab_s += powers[rank*k:(k+1)*(1-rank):1-2*rank]
    # # [powers[(rank * (k - 2 * i) + i)] for i in range(k + 1)]
    # ab_s = torch.stack(ab_s)
    # a_s = ab_s
    # b_s = ab_s
    a_s = []
    if self.is_cuda:
        for k in range(degree + 1):
            a_s += powers[:k + 1].tensor()
        a_s = CUDALongTensor(torch.stack(a_s))
    else:
        for k in range(degree + 1):
            a_s += powers[:k + 1]
        a_s = torch.stack(a_s)

    b_s = []
    if self.is_cuda:
        for k in range(degree + 1):
            b_s += torch.flip(powers[:k + 1].tensor(), dims=(0,))
        b_s = CUDALongTensor(torch.stack(b_s))
    else:
        for k in range(degree + 1):
            b_s += torch.flip(powers[:k + 1], dims=(0,))
        b_s = torch.stack(b_s)

    """Resharing"""
    # with Timer("Resharing"):
    a = crypten.cryptensor(a_s, precision=0, src=0, device=self.device, ptype=crypten.mpc.arithmetic)
    b = crypten.cryptensor(b_s, precision=0, src=1, device=self.device, ptype=crypten.mpc.arithmetic)

    """Final Aggregation"""
    # with Timer("Aggregation"):
    ans = (a * b * nCr).sum(dim=0) / scale ** degree

    """Postprocessing"""
    ans.encoder = self.encoder
    ans._tensor = ans._tensor.reshape(shape)
    self._tensor = self._tensor.reshape(shape)
    return ans


def honeybadger_poly(self, coeffs):
    k = coeffs.size(0) - 1
    scale = self.encoder.scale
    kpows = self._tensor.honeybadger_pows(k)
    coeffs_t = torch.tensor([coeffs[i] * scale ** (k - i) for i in range(1, k + 1)], device=self.device)
    while len(coeffs_t.size()) < len(kpows.size()):
        coeffs_t = coeffs_t.unsqueeze(1)
    ans = (kpows * coeffs_t).sum(dim=0) / scale ** (k - 1)
    return ans + coeffs[0]


def polymath_poly(x, coeffs, func="mul"):
    pass


def polynomial(self, coeffs, func="mul"):
    # Coefficient input type-checking
    if isinstance(coeffs, list):
        coeffs = torch.tensor(coeffs, device=self.device)
    assert is_tensor(coeffs) or crypten.is_encrypted_tensor(
        coeffs
    ), "Polynomial coefficients must be a list or tensor"
    assert coeffs.dim() == 1, "Polynomial coefficients must be a 1-D tensor"

    # Handle linear case
    if coeffs.size(0) == 1:
        return self.mul(coeffs)

    # TODO: find a cleaner solution to handle empty coefficients
    if coeffs.size(0) == 0:
        terms = crypten.stack([self])
        terms = terms[: coeffs.size(0)]
        for _ in range(terms.dim() - 1):
            coeffs = coeffs.unsqueeze(1)
        return terms.mul(coeffs).sum(0)

    if cfg.functions.poly_method == "crypten":
        return self.crypten_poly(coeffs, func)
    elif cfg.functions.poly_method == "espn":
        return self.espn_poly(coeffs)
    elif cfg.functions.poly_method == "honeybadger":
        return self.honeybadger_poly(coeffs)
    elif cfg.functions.poly_method == "polymath":
        return polymath_poly(self, coeffs, func)
    else:
        raise NotImplementedError(f"{cfg.functions.poly_method} polynomial evaluation is not supported")


def norm(self, p="fro", dim=None, keepdim=False):
    """Computes the p-norm of the input tensor (or along a dimension)."""
    if p == "fro":
        p = 2

    if isinstance(p, (int, float)):
        assert p >= 1, "p-norm requires p >= 1"
        if p == 1:
            if dim is None:
                return self.abs().sum()
            return self.abs().sum(dim, keepdim=keepdim)
        elif p == 2:
            if dim is None:
                return self.square().sum().sqrt()
            return self.square().sum(dim, keepdim=keepdim).sqrt()
        elif p == float("inf"):
            if dim is None:
                return self.abs().max()
            return self.abs().max(dim=dim, keepdim=keepdim)[0]
        else:
            if dim is None:
                return self.abs().pos_pow(p).sum().pos_pow(1 / p)
            return self.abs().pos_pow(p).sum(dim, keepdim=keepdim).pos_pow(1 / p)
    elif p == "nuc":
        raise NotImplementedError("Nuclear norm is not implemented")
    else:
        raise ValueError(f"Improper value p ({p})for p-norm")
