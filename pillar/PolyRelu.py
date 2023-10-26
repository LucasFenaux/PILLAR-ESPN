import pickle

import torch
import torch.nn as nn
import os
import numpy as np
from gekko import GEKKO
from utils import SmoothedValue, accuracy


def get_penalty(penalty_exp: int = 10, reg_coef: float = 0.001, reg_range: float = 4., reset: bool = True):
    activations = PolyRelu.buffer
    penalties = []
    for activation in activations:
        activation_norm = torch.pow(activation / reg_range, penalty_exp).flatten()
        activation_norm = torch.mean(torch.linalg.norm(activation_norm, ord=1, dim=0)) * reg_coef
        penalties.append(activation_norm)

    if len(penalties) == 0:
        penalty = 0.
    else:
        penalty = torch.mean(torch.stack(penalties))

    if reset:
        PolyRelu.reset()
    return penalty


def get_metrics(dataloader, model, input_range, device, metrics: list):
    count = 0
    total = 0
    max_val = 0
    unique_imgs_total = 0
    oor_mean = SmoothedValue()
    oor_std = SmoothedValue()
    nan_count = 0
    acc = SmoothedValue()
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            acc1, _ = accuracy(out, y, topk=(1, 5))
            acc.update(acc1, n=y.size(0))

            all_out_of_range_values = []

            activations = PolyRelu.buffer

            upper = input_range[1]
            lower = input_range[0]

            unflattened_activation = torch.cat(
                [x.detach().to(device).view(x.size()[0], -1) for x in activations], dim=1)

            within_upper = torch.where(upper > unflattened_activation, 1, 0)
            within_lower = torch.where(unflattened_activation > lower, 1, 0)
            in_range = within_upper * within_lower

            if "oor_percentage" in metrics or "num_oor_total" in metrics:
                count += torch.sum(in_range).item()
                total += unflattened_activation.size()[0] * unflattened_activation.size()[1]

            if "unique_image_with_oor_total" in metrics:
                num_oor_unique = (torch.ones(in_range.size()[0]) * in_range.size()[1]).to(device)
                num_oor_unique -= torch.count_nonzero(in_range, dim=1)
                unique_imgs_with_oor = torch.count_nonzero(num_oor_unique).item()
                unique_imgs_total += unique_imgs_with_oor

            if "oor_values" in metrics:
                out_of_range_indices = torch.where(in_range == 1, 0, 1)
                flattened_out_of_range_values = torch.flatten(unflattened_activation * out_of_range_indices)
                all_out_of_range_values.append(flattened_out_of_range_values[flattened_out_of_range_values.nonzero()])

            if "oor_max" in metrics:
                cur_max = torch.max(torch.abs(unflattened_activation))
                if cur_max > max_val:
                    max_val = cur_max

            all_oor_values = torch.abs(torch.cat(all_out_of_range_values).squeeze()).detach().cpu().numpy()
            arr_size = all_oor_values.size
            not_empty = False
            if isinstance(arr_size, int):
                if arr_size > 0:
                    not_empty = True
            else:
                not_empty = True

            if not_empty:
                # we make sure to replace all inf values with nans so the following computations run fine
                all_oor_values = np.where(all_oor_values == np.inf, np.nan, all_oor_values)
                oor_m = np.nanmean(all_oor_values)

                oor_mean.update(oor_m, n=x.size(0))

                oor_s = np.nanstd(all_oor_values)

                oor_std.update(oor_s, n=x.size(0))

                nan_count += np.count_nonzero(np.isnan(all_oor_values))

            PolyRelu.reset()

        metric_dict = {}
        if "oor_percentage" in metrics:
            if total == 0:
                metric_dict["oor_percentage"] = 0
            else:
                metric_dict["oor_percentage"] = (1 - count / total) * 100

        if "num_oor_total" in metrics:
            metric_dict["num_oor_total"] = total - count

        if "oor_max" in metrics:
            metric_dict["oor_max"] = max_val

        if "unique_image_with_oor_total" in metrics:
            metric_dict["unique_image_with_oor_total"] = unique_imgs_total

        if "oor_values" in metrics:
            try:
                metric_dict["oor_values"] = torch.cat(all_out_of_range_values)
            except RuntimeError:
                metric_dict["oor_values"] = torch.empty(0)

    if not isinstance(oor_mean, float):
        oor_mean = oor_mean.global_avg
    metric_dict["oor_mean"] = oor_mean
    if not isinstance(oor_std, float):
        oor_std = oor_std.global_avg
    metric_dict["oor_std"] = oor_std
    metric_dict["nan_count"] = nan_count

    metric_dict["Acc"] = acc.global_avg
    print(f"Out of range distribution information")
    if "oor_max" in metrics:

        print(f"OOR Max: {max_val:.3f}")
    if oor_mean and oor_std is not None:
        print(f"OOR Mean: {oor_mean:.3f}")
        print(f"OOR Std: {oor_std:.3f}")
    else:
        print("No OOR value so cannot compute mean or standard deviation")
    print(f"Nan count: {nan_count}")
    if "num_oor_total" in metrics:
        print(f"Num Oor Total: {metric_dict['num_oor_total']}")
    if "unique_image_with_oor_total" in metrics:
        print(f"Num Unique Images With Oor: {unique_imgs_total}")
    print(f"Acc: {acc.global_avg}")
    if "oor_percentage" in metrics:

        print(f"Out of Range %: {metric_dict['oor_percentage']}")

    return metric_dict


def real_relu(x):
    return np.maximum(0, x)


def real_sigmoid(x):
    x_tensor = torch.Tensor(x)
    return torch.sigmoid(x_tensor).cpu().numpy()


def generate_coeffs(activation_func, file_prefix, degree, rng, granularity=1e-2, quantized_coef=1,
                    crypto_precision=8):
    filename = f"{file_prefix}/d{degree}_r{rng}_g{granularity}_q{quantized_coef}_cp{crypto_precision}_coeffs.p"
    if os.path.exists(filename):
        #Load from file
        with open(filename, 'rb') as input_file:
            coeffs = pickle.load(input_file)
    else:
        ''' Modified from: 
        https://github.com/kvgarimella/sisyphus-ppml/blob/main/experiments/poly_regression/generate_poly_regression_coeffs.py'''
        steps = int(2 * rng / granularity)
        xs = np.linspace(-rng, rng, steps)
        ys = activation_func(xs)
        if quantized_coef:
            coeffs = quantized_poly_fit(xs, ys, degree,
                                        crypto_precision=crypto_precision)
        else:
            coeffs = np.polyfit(xs, ys, deg=degree)[::-1]
        with open(filename, 'wb') as out_file:
            pickle.dump(coeffs, out_file)
    return coeffs


'''Shifts the poly fit by the precision then fits with integer programing
This ensure we find the best coeffs under precision constraints'''
def quantized_poly_fit(xs, ys, degree, crypto_precision=8):
    X = []
    for i in range(degree+1):
        X.append(np.power(xs, i))
    X = np.vstack(X)
    X = X.T
    # X = np.trunc(np.multiply(X, 2 ** crypto_precision)).astype(int)
    ys = np.trunc(np.multiply(ys, 2 ** crypto_precision)).astype(int)

    m = GEKKO(remote=False)  # Initialize gekko
    m.options.SOLVER = 1  # APOPT is an MINLP solver

    #optional solver settings with APOPT
    # m.solver_options = ['minlp_maximum_iterations 500', \
    #                     # minlp iterations with integer solution
    #                     'minlp_max_iter_with_int_sol 10', \
    #                     # treat minlp as nlp
    #                     'minlp_as_nlp 0', \
    #                     # nlp sub-problem max iterations
    #                     'nlp_maximum_iterations 50', \
    #                     # 1 = depth first, 2 = breadth first
    #                     'minlp_branch_method 1', \
    #                     # maximum deviation from whole number
    #                     'minlp_integer_tol 0.05', \
    #                     # covergence tolerance
    #                     'minlp_gap_tol 0.01']

    # Initialize variables
    alpha = [m.Var(integer=True, lb=-2**(crypto_precision-1), ub=2**(crypto_precision-1)) for _ in range(len(X[0]))]
    result = []
    for i in range(len(X)):
        sum_val = 0
        for j in range(len(X[0])):
            sum_val += X[i][j] * alpha[j]
        result.append(m.Intermediate((sum_val - ys[i]) ** 2))
    m.Obj(sum(result))  # Objective
    m.solve(disp=False)  # Solve
    alpha = np.array([x.value[0] for x in alpha])
    return alpha / (2 ** crypto_precision)


class PolynomialEvaluator(nn.Module):
    def __init__(self, coeffs, EPS=1e-9):
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


class PolyRelu(nn.Module):
    buffer = []
    ids = []
    ordered_buffer = {}
    def __init__(self, coeffs, range=5., reg_range=5., alpha=0, clip: bool = False, regularize: bool = False,
                 monitor: bool = False):
        super().__init__()
        self.evaluator = PolynomialEvaluator(coeffs)
        self.clip = clip
        self.range = range
        self.reg_range = reg_range
        self.alpha = alpha
        self.regularize = regularize
        self.monitor = monitor
        self.input_count = -1

        if len(PolyRelu.ids) == 0:
            self.id = 0
            PolyRelu.ids.append(0)
        else:
            self.id = max(PolyRelu.ids) + 1
            PolyRelu.ids.append(self.id)

        PolyRelu.ordered_buffer[self.id] = []

    def forward(self, x):
        input_count = -1
        size = x.size()
        for i, s in enumerate(size):
            if i == 0:
                # batch_size
                pass
            elif i == 1:
                input_count = s
            else:
                input_count *= s
        self.input_count = input_count
        if self.regularize:
            PolyRelu.buffer.append(x)
        if self.monitor:
            PolyRelu.ordered_buffer[self.id].append(x)

        if self.clip and self.training:
            x = torch.clip(x, -self.range, self.range)

        ans = self.evaluator(x)
        return ans

    @classmethod
    def reset(cls):
        cls.buffer = []
        for module_id in cls.ids:
            cls.ordered_buffer[module_id] = []

    @classmethod
    def reset_ids(cls):
        cls.ids = []
        cls.ordered_buffer = {}


class MonitoredRelu(nn.Module):
    buffer = []
    ids = []
    ordered_buffer = {}

    def __init__(self, monitor: bool = True):
        super(MonitoredRelu, self).__init__()
        self.monitor = monitor
        self.evaluator = nn.ReLU()
        if len(MonitoredRelu.ids) == 0:
            self.id = 0
            MonitoredRelu.ids.append(0)
        else:
            self.id = max(MonitoredRelu.ids) + 1
            MonitoredRelu.ids.append(self.id)

        MonitoredRelu.ordered_buffer[self.id] = []

    def forward(self, x):
        if self.monitor:
            MonitoredRelu.buffer.append(x)
            MonitoredRelu.ordered_buffer[self.id].append(x)
        ans = self.evaluator(x)
        return ans

    @classmethod
    def reset(cls):
        cls.buffer = []
        for module_id in cls.ids:
            cls.ordered_buffer[module_id] = []

    @classmethod
    def reset_ids(cls):
        cls.ids = []
        cls.ordered_buffer = {}


def test(relu, x, coeffs):
    act = relu(x)
    act_naive = sum(x ** i * coeffs[i] for i in range(len(coeffs)))
    error = (act - act_naive).abs().sum()
    return error


def plot_relu(relu, x):
    import matplotlib.pyplot as plt
    plt.plot(x, relu(x))
    plt.savefig("relu.png")


if __name__ == "__main__":
    coeffs = generate_coeffs(real_relu, degree=5, file_prefix="./poly_coefs", rng=5.,
                             quantized_coef=True, crypto_precision=8)
    print(coeffs)
    coeffs = generate_coeffs(real_relu, degree=5, file_prefix="./poly_coefs", rng=5.,
                             quantized_coef=True, crypto_precision=10)
    print(coeffs)
