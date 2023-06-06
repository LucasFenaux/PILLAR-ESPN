#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import random
import time
from datetime import timedelta

import numpy
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets, transforms
# from torchvision.models.resnet import resnet50
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import InterpolationMode

import crypten
import crypten.communicator as comm
from crypten.config import cfg
from datasets.cifar import CIFAR10
from examples.meters import AverageMeter
from examples.mpc_inference import presets
from models.PolynomialEvaluator import PolynomialEvaluator
from models.resnet import resnet18
from models.resnet_x import resnet32, resnet110, MiniONN
from models.vgg import ModulusNet_vgg16 as vgg16
from models.vgg import ModulusNet_vgg16_bn as vgg16_bn


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


def replace_relus(model, evaluator=None):
    if cfg.functions.relu_method != "poly":
        return
    if not isinstance(model, torch.nn.Module):
        return
    if evaluator is None:
        evaluator = PolynomialEvaluator(list(cfg.functions.relu_coeffs))
    if hasattr(model, "relu") and not isinstance(getattr(model, "relu"), torch.nn.Identity):
        setattr(model, "relu", evaluator)
    for module in model.children():
        replace_relus(module, evaluator)


def build_model(model_type: str = "resnet18", num_classes=None):
    if model_type == "resnet18":
        model = resnet18(num_classes=num_classes)
    elif model_type == "resnet32":
        model = resnet32(num_classes=num_classes, init_weights=False)
    elif model_type == "resnet50":
        model = resnet50(num_classes=num_classes, weights=ResNet50_Weights.DEFAULT)
    elif model_type == "resnet110":
        model = resnet110(num_classes=num_classes, init_weights=False)
    elif model_type == "minionn":
        model = MiniONN(num_classes=num_classes, init_weights=False, use_batch_norm=False)
    elif model_type == "minionn_bn":
        model = MiniONN(num_classes=num_classes, init_weights=False, use_batch_norm=True)
    elif model_type.startswith("vgg16"):
        pool = "avg" if "avg" in model_type else "max"
        if "bn" in model_type:
            model = vgg16_bn(num_classes=num_classes, pool=pool)
        else:
            model = vgg16(num_classes=num_classes, pool=pool)
    else:
        raise NotImplementedError
    return model


def get_dataset(datatset_name: str = "cifar10", batch_size=1):
    g = torch.Generator()
    g.manual_seed(0)
    if datatset_name == "cifar10":
        dataset = CIFAR10(root="../data/cifar10")
        val_loader = dataset.get_dataloader('valid',
                                            shuffle=False,
                                            batch_size=batch_size,
                                            num_workers=2,
                                            pin_memory=False,
                                            worker_init_fn=seed_worker,
                                            generator=g)
        num_classes = 10
    elif datatset_name == "cifar100":
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                             (0.2023, 0.1994, 0.2010))])
        dataset = datasets.CIFAR100(root="../data/cifar100", transform=transform, train=False,
                                    download=True)
        val_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=False,
            worker_init_fn=seed_worker,
            generator=g)
        num_classes = 100
    elif datatset_name == "imagenet":
        interpolation = InterpolationMode("bilinear")
        preprocessing = presets.ClassificationPresetEval(
            crop_size=224, resize_size=232, interpolation=interpolation
        )

        dataset = datasets.ImageFolder("../data/imagenet/val", preprocessing)

        val_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=False,
            worker_init_fn=seed_worker,
            generator=g)
        num_classes = 1000

    return val_loader, num_classes


def run_mpc_model(
        batch_size=1,
        n_batches=None,
        print_freq=10,
        model_location="",
        model_type="",
        dataset="",
        seed=None,
        skip_plaintext=False,
        resume=False,
        evaluate_separately=False,
        device=torch.device('cpu')
):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        numpy.random.seed(seed)

    val_loader, num_classes = get_dataset(dataset, batch_size)

    # create model
    model = build_model(model_type, num_classes)
    criterion = nn.CrossEntropyLoss()

    # optionally resume from a checkpoint
    if resume:
        if os.path.isfile(model_location):
            logging.info("=> loading checkpoint '{}'".format(model_location))
            checkpoint = torch.load(model_location)
            model.load_state_dict(checkpoint['model'])
            logging.info(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    model_location, checkpoint["epoch"]
                )
            )
        else:
            raise IOError("=> no checkpoint found at '{}'".format(model_location))

    model = model.to(device)
    input_size = get_input_size(val_loader)
    private_model = construct_private_model(input_size, model, model_type, num_classes, device=device)
    replace_relus(model)

    if not skip_plaintext:
        logging.info("===== Evaluating plaintext LeNet network =====")
        validate(val_loader, model, criterion, print_freq, device=device)
    if evaluate_separately:
        logging.info("===== Evaluating Private LeNet network =====")
        validate(val_loader, private_model, criterion, print_freq, device=device)
    logging.info("===== Validating side-by-side ======")
    return validate_side_by_side(val_loader, model, private_model, device, n_batches)


def validate_side_by_side(val_loader, plaintext_model, private_model, device, n_batches=None):
    """Validate the plaintext and private models side-by-side on each example"""
    # switch to evaluate mode
    plaintext_model.eval()
    private_model.eval()

    accuracy_plain = AverageMeter()
    accuracy_enc = AverageMeter()
    average_error = AverageMeter()
    match = AverageMeter()
    inference_time = AverageMeter()
    communication_time = AverageMeter()
    total_time = AverageMeter()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input = input.to(device)
            target = target.to(device)
            # compute output for plaintext
            output_plaintext = plaintext_model(input)
            # encrypt input and compute output for private
            # assumes that private model is encrypted with src=0
            input_encr = encrypt_data_tensor_with_src(input)
            comm.get().reset_communication_stats()
            start_time = time.monotonic()
            output_encr = private_model(input_encr)
            end_time = time.monotonic()
            # inspect(output_encr, "Output of Model -->", output_plaintext)
            # print(count_nans(output_plaintext))
            comm_stats = comm.get().get_communication_stats()

            delta = timedelta(seconds=end_time - start_time)
            time_diff = delta.seconds + (delta.microseconds / 1e6)
            average_time_diff = time_diff / input.size(0)
            inference_time.add(average_time_diff, input.size(0))
            total_time.add(time_diff, 1)

            communication_time.add(comm_stats['time'])
            output_encr = output_encr.get_plain_text()
            # Error
            error = (output_encr - output_plaintext).abs().sum().sum() / (
                    output_plaintext.shape[0] * output_plaintext.shape[1])

            average_error.add(error, (output_plaintext.shape[0] * output_plaintext.shape[1]))
            # Predictions
            pred_plain = output_plaintext.argmax(dim=1)
            pred_enc = output_encr.argmax(dim=1)
            # Accuracies
            acc_plain = accuracy(output_plaintext, target)
            accuracy_plain.add(acc_plain[0], input.size(0))
            acc_enc = accuracy(output_encr, target)
            accuracy_enc.add(acc_enc[0], input.size(0))
            # Match
            mtch = sum(pred_plain == pred_enc) * 100 / input.size(0)
            match.add(mtch, input.size(0))

            # log all info
            logging.info("==============================")
            logging.info(f"Example {i}\t target = {target}")
            logging.info(f"Example {i}\t plainP = {pred_plain}")
            logging.info(f"Example {i}\t encryP = {pred_enc}")
            logging.info("   ========================   ")
            logging.info("Enc Acc    {:.3f} ({:.3f})".format(acc_enc[0].item(), accuracy_enc.value().item()))
            logging.info("Pla Acc    {:.3f} ({:.3f})".format(acc_plain[0].item(), accuracy_plain.value().item()))
            logging.info("Match      {:.3f} ({:.3f})".format(mtch.item(), match.value().item()))
            logging.info("Error      {:.3f} ({:.3f})".format(error.item(), average_error.value().item()))
            logging.info("Runtime    {:.3f} ({:.3f})".format(average_time_diff, inference_time.value()))
            logging.info("Rounds     {:.3f} ({:.3f})".format(comm_stats['rounds'], comm_stats['rounds']))
            logging.info("Bytes      {:.3f} ({:.3f})".format(comm_stats['bytes'], comm_stats['bytes']))
            logging.info("Comtime    {:.3f} ({:.3f})".format(comm_stats['time'], communication_time.value()))
            logging.info("Runtime(T) {:.3f} ({:.3f})".format(time_diff, total_time.value()))
            # only use the first 1000 examples
            if n_batches is not None and i + 1 >= n_batches:
                break
    comm_stats.pop("time")
    _, mmh, mph = total_time.mean_confidence_interval()
    results = {
        "enc_acc": accuracy_enc.value().item(),
        "pla_acc": accuracy_plain.value().item(),
        "match": match.value().item(),
        "error": average_error.value().item(),
        "comm_time": communication_time.value(),
        "run_time": total_time.value(),
        "run_time_95conf_lower": mmh.item(),
        "run_time_95conf_upper": mph.item(),
        "run_time_amortized": inference_time.value(),
        "comm": comm_stats
    }
    return results


def get_input_size(val_loader):
    input, target = next(iter(val_loader))
    return input.size()


def construct_private_model(input_size, model, model_name, num_classes, device=torch.device('cpu')):
    """Encrypt and validate trained model for multi-party setting."""
    # get rank of current process
    rank = comm.get().get_rank()
    dummy_input = torch.empty(input_size, device=device)

    # party 0 always gets the actual model; remaining parties get dummy model
    if rank == 0:
        model_upd = model
    else:
        model_upd = build_model(model_name, num_classes)

    model_upd = model_upd.to(device)
    private_model = crypten.nn.from_pytorch(model_upd, dummy_input).to(device).encrypt(src=0)
    return private_model


def encrypt_data_tensor_with_src(input):
    """Encrypt data tensor for multi-party setting"""
    # get rank of current process
    rank = comm.get().get_rank()
    # get world size
    world_size = comm.get().get_world_size()

    if world_size > 1:
        # party 1 gets the actual tensor; remaining parties get dummy tensor
        src_id = 1
    else:
        # party 0 gets the actual tensor since world size is 1
        src_id = 0

    if rank == src_id:
        input_upd = input
    else:
        input_upd = torch.empty(input.size(), device=input.device)
    private_input = crypten.cryptensor(input_upd, src=src_id)
    return private_input


def validate(val_loader, model, criterion, print_freq=10, device=torch.device('cpu')):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.to(device)
            target = target.to(device)
            if isinstance(model, crypten.nn.Module) and not crypten.is_encrypted_tensor(
                    input
            ):
                input = encrypt_data_tensor_with_src(input)
            # compute output
            output = model(input)
            if crypten.is_encrypted_tensor(output):
                output = output.get_plain_text()
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.add(loss.item(), input.size(0))
            top1.add(prec1[0], input.size(0))
            top5.add(prec5[0], input.size(0))

            # measure elapsed time
            current_batch_time = time.time() - end
            batch_time.add(current_batch_time)
            end = time.time()

            if (i + 1) % print_freq == 0:
                logging.info(
                    "\nTest: [{}/{}]\t"
                    "Time {:.3f} ({:.3f})\t"
                    "Loss {:.4f} ({:.4f})\t"
                    "Prec@1 {:.3f} ({:.3f})   \t"
                    "Prec@5 {:.3f} ({:.3f})".format(
                        i + 1,
                        len(val_loader),
                        current_batch_time,
                        batch_time.value(),
                        loss.item(),
                        losses.value(),
                        prec1[0],
                        top1.value(),
                        prec5[0],
                        top5.value(),
                    )
                )

        logging.info(
            " * Prec@1 {:.3f} Prec@5 {:.3f}".format(top1.value(), top5.value())
        )
    return top1.value()


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
