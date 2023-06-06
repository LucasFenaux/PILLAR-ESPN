#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import logging
import ntpath
import os
import shutil

import torch
import yaml

from crypten.config import cfg
from examples.multiprocess_launcher import MultiProcessLauncher

parser = argparse.ArgumentParser(description="CrypTen Multidataset Inference")
parser.add_argument(
    "--world_size",
    type=int,
    default=2,
    help="The number of parties to launch. Each party acts as its own process",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=1,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256)",
)
parser.add_argument(
    "--n-batches",
    default=None,
    type=int,
    metavar="N",
    help="num of batches to evaluate",
)
parser.add_argument(
    "--print-freq",
    "-p",
    default=1,
    type=int,
    metavar="N",
    help="print frequency (default: 1)",
)
parser.add_argument(
    "--model-location",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "--model-type",
    default="resnet32",
    type=str,
    choices=["resnet18", "resnet32", "resnet50", "resnet110",
             "vgg16_avg", "vgg16_avg_bn", "vgg16_max", "vgg16_max_bn",
             "minionn", "minionn_bn"],
    help="Model architecture",
)

parser.add_argument(
    "--dataset",
    default="cifar10",
    type=str,
    choices=["cifar10", "cifar100", "imagenet"],
    help="evaluation dataset",
)

parser.add_argument(
    "--config",
    default="configs/default.yaml",
    type=str,
    metavar="PATH",
    help="path to latest crypten config",
)

parser.add_argument('-d', '--delays', nargs='+', help='delays for experiments', type=float, default=[0.0, 0.05])

parser.add_argument(
    "--seed", default=0, type=int, help="seed for initializing training. "
)
parser.add_argument(
    "--multiprocess",
    default=False,
    action="store_true",
    help="Run example in multiprocess mode",
)
parser.add_argument(
    "--use-cuda",
    default=False,
    action="store_true",
    help="Run example on gpu",
)
parser.add_argument(
    "--resume",
    default=False,
    action="store_true",
    help="Resume training from latest checkpoint",
)
parser.add_argument(
    "--skip-plaintext",
    default=False,
    action="store_true",
    help="skip plaintext evaluation",
)
parser.add_argument(
    "--evaluate-separately",
    default=False,
    action="store_true",
    help="evaluate private model separately",
)


def _run_experiment(args):
    # only import here to initialize crypten within the subprocesses
    from examples.mpc_inference.mpc_inference import run_mpc_model
    # Only Rank 0 will display logs.
    level = logging.INFO
    rank = os.environ['RANK']
    if int(rank) != 0:
        level = logging.CRITICAL
    logging.getLogger().setLevel(level)
    # Device and config
    device_id = rank if torch.cuda.device_count() > 1 else 0
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() and args.use_cuda else "cpu")
    cfg.load_config(args.config)
    # Naming
    cfg_name = ntpath.basename(args.config).split(".")[0]
    model_name = args.model_type
    if args.resume:
        model_name = "_".join(args.model_location.split("/")[-3:-1])
    results_path = f"results/{args.dataset}/{model_name}_{device}"
    results = None
    aggregable_keys = ["comm_time", "run_time", "run_time_amortized", "run_time_95conf_lower", "run_time_95conf_upper"]
    for delay in args.delays:
        cfg.communicator.delay = delay
        _results = run_mpc_model(
            batch_size=args.batch_size,
            n_batches=args.n_batches,
            print_freq=args.print_freq,
            model_location=args.model_location,
            model_type=args.model_type,
            dataset=args.dataset,
            seed=args.seed,
            skip_plaintext=args.skip_plaintext,
            resume=args.resume,
            evaluate_separately=args.evaluate_separately,
            device=device
        )
        if results is None:
            results = _results
            for key in aggregable_keys:
                results[key] = [results[key]]
        else:
            for key in aggregable_keys:
                results[key].append(_results[key])
    results['delays'] = args.delays
    # with open(f"{results_path}/{cfg_name}_result_{rank}.yaml", "w") as f:
    #     yaml.dump(results, f)
    if int(rank) == 0:
        os.makedirs(results_path, exist_ok=True)
        shutil.copyfile(args.config, f"{results_path}/{cfg_name}.yaml")
        with open(f"{results_path}/{cfg_name}_result.yaml", "w") as f:
            yaml.dump(results, f)


def main(run_experiment):
    args = parser.parse_args()
    if args.multiprocess:
        launcher = MultiProcessLauncher(args.world_size, run_experiment, args)
        launcher.start()
        launcher.join()
        launcher.terminate()
    else:
        run_experiment(args)


if __name__ == "__main__":
    main(_run_experiment)
