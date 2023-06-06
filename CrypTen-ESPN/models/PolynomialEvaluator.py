import torch
import torch.nn as nn


class PolynomialEvaluator(nn.Module):
    def __init__(self, coeffs, EPS=1e-6):
        super(PolynomialEvaluator, self).__init__()
        self.len_coeffs = len(coeffs)
        self.coeffs = coeffs
        self.powers = None
        self.indices = [i for i in range(len(coeffs)) if abs(coeffs[i]) > EPS]
        self.reset()

    def reset(self):
        self.powers = [None for _ in range(self.len_coeffs)]

    def pow(self, x, i):
        if self.powers[i] is None:
            if i == 0:
                self.powers[i] = torch.ones_like(x)
            elif i == 1:
                self.powers[i] = x
            else:
                half = self.pow(x, i // 2)
                full = half * half
                if i % 2 == 1:
                    full *= x
                self.powers[i] = full
        return self.powers[i]

    def forward(self, x):
        ans = sum(self.coeffs[i] * self.pow(x, i) for i in self.indices)
        self.reset()
        return ans
