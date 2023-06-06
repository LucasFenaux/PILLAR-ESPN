"""Originally from https://github.com/pytorch/vision/blob/main/references/classification/
Modified and adapted for our purposes"""

import datetime
import os
import time
import warnings

import presets
import torch
import torch.utils.data
import torchvision
import transforms
import utils
from PolyRelu import PolyRelu, get_penalty, get_metrics, generate_coeffs, real_relu
from sampler import RASampler
from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode
from torchvision import datasets
from resnet_x import resnet32, resnet110, ConvNet, MiniONN
from resnet import resnet18
import numpy as np
from vgg import ModulusNet_vgg16_bn


def train_and_regularize_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema=None,
                                   scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.3e}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value:.2e}"))
    metric_logger.add_meter("pen", utils.SmoothedValue(window_size=10, fmt="{value:.1e}"))

    header = f"Epoch: [{epoch}]"
    try:
        for i, (image, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
            start_time = time.time()
            image, target = image.to(device), target.to(device)
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                output = model(image)
                l = criterion(output, target)
                if epoch == 0 and not args.no_reg_warmup:
                    pen = get_penalty(4, args.reg_coef / 100, args.reg_range)
                    loss = l * 0. + pen
                elif epoch == 1 and not args.no_reg_warmup:
                    pen = get_penalty(6, args.reg_coef/50, args.reg_range)
                    loss = l + pen
                elif epoch == 2 and not args.no_reg_warmup:
                    pen = get_penalty(8, args.reg_coef/10, args.reg_range)
                    loss = l + pen
                elif epoch == 3 and not args.no_reg_warmup:
                    pen = get_penalty(10, args.reg_coef / 5, args.reg_range)
                    loss = l + pen
                else:
                    pen = get_penalty(args.penalty_exp, args.reg_coef, args.reg_range)
                    loss = l + pen

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                if args.clip_grad_norm is not None:
                    # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.clip_grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()

            if model_ema and i % args.model_ema_steps == 0:
                model_ema.update_parameters(model)
                if epoch < args.lr_warmup_epochs:
                    # Reset ema buffer to keep copying weights during warmup period
                    model_ema.n_averaged.fill_(0)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            batch_size = image.shape[0]
            metric_logger.update(loss=l.item(), lr=optimizer.param_groups[0]["lr"])

            metric_logger.meters["pen"].update(pen.item(), n=batch_size)
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))
        if np.isnan(metric_logger.meters["pen"].global_avg):
            print("Model died during prep time, you should consider either increasing the range, reducing "
                  "the regularization coefficient or lowering the learning rate")
            return False
        else:
            return True

    except Exception as e:
        print(f"Training epoch died due to {e}")
        return False


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema=None, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    for i, (image, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                # Reset ema buffer to keep copying weights during warmup period
                model_ema.n_averaged.fill_(0)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))

    if np.isnan(metric_logger.meters["loss"].global_avg):
        print("Model died during training due to a NaN loss")
        return False
    else:
        return True


def evaluate(model, criterion, data_loader, device, args, print_freq=100, log_suffix="", eval_batches=None):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    with torch.inference_mode():
        for i, (image, target) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            if eval_batches is not None and i > eval_batches:
                break
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)
            if args.use_poly and args.regularize: #and not args.quantize:
                loss += get_penalty(args.penalty_exp, args.reg_coef, args.reg_range)
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size
    # gather the stats from all processes

    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if (
            hasattr(data_loader.dataset, "__len__")
            and len(data_loader.dataset) != num_processed_samples
            and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f} "
          f"Loss: {metric_logger.loss.global_avg:.3f}")
    return metric_logger.acc1.global_avg, metric_logger.loss.global_avg


def _get_cache_path(filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def load_cifar_data(args):
    # Data loading code
    print("Loading data")

    print("Loading training data")
    st = time.time()

    auto_augment_policy = getattr(args, "auto_augment", None)

    transform_list = [
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
    if auto_augment_policy is not None:
        transform_list.insert(2, torchvision.transforms.AutoAugment(torchvision.transforms.AutoAugmentPolicy.CIFAR10))

    if args.dataset == "cifar10":
        dataset = datasets.CIFAR10(root=os.path.join(args.data_path, args.dataset),
                                   transform=torchvision.transforms.Compose(transform_list), train=True, download=True)
    elif args.dataset == "cifar100":
        dataset = datasets.CIFAR100(root=os.path.join(args.data_path, args.dataset),
                                    transform=torchvision.transforms.Compose(transform_list), train=True, download=True)
    else:
        raise NotImplementedError

    print("Took", time.time() - st)

    print("Loading validation data")

    if args.dataset == "cifar10":
        dataset_test = datasets.CIFAR10(root="./cifar10",
                                        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                  torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
                                        train=False, download=True)
    elif args.dataset == "cifar100":
        dataset_test = datasets.CIFAR100(root="./cifar100",
                                        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                  torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
                                        train=False, download=True)
    else:
        raise NotImplementedError

    print("Creating data loaders")
    if args.distributed:
        if hasattr(args, "ra_sampler") and args.ra_sampler:
            train_sampler = RASampler(dataset, shuffle=True, repetitions=args.ra_reps)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler


def load_data(traindir, valdir, args):
    # Data loading code
    print("Loading data")
    val_resize_size, val_crop_size, train_crop_size = (
        args.val_resize_size,
        args.val_crop_size,
        args.train_crop_size,
    )
    interpolation = InterpolationMode(args.interpolation)

    print("Loading training data")
    st = time.time()
    cache_path = _get_cache_path(traindir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print(f"Loading dataset_train from {cache_path}")
        dataset, _ = torch.load(cache_path)
    else:
        auto_augment_policy = getattr(args, "auto_augment", None)
        random_erase_prob = getattr(args, "random_erase", 0.0)
        ra_magnitude = args.ra_magnitude
        augmix_severity = args.augmix_severity
        dataset = torchvision.datasets.ImageFolder(
            traindir,
            presets.ClassificationPresetTrain(
                crop_size=train_crop_size,
                interpolation=interpolation,
                auto_augment_policy=auto_augment_policy,
                random_erase_prob=random_erase_prob,
                ra_magnitude=ra_magnitude,
                augmix_severity=augmix_severity,
            ),
        )
        if args.cache_dataset:
            print(f"Saving dataset_train to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset, traindir), cache_path)
    print("Took", time.time() - st)

    print("Loading validation data")
    cache_path = _get_cache_path(valdir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print(f"Loading dataset_test from {cache_path}")
        dataset_test, _ = torch.load(cache_path)
    else:
        if args.weights and args.test_only:
            weights = torchvision.models.get_weight(args.weights)
            preprocessing = weights.transforms()
        else:
            preprocessing = presets.ClassificationPresetEval(
                crop_size=val_crop_size, resize_size=val_resize_size, interpolation=interpolation
            )

        dataset_test = torchvision.datasets.ImageFolder(
            valdir,
            preprocessing,
        )
        if args.cache_dataset:
            print(f"Saving dataset_test to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_test, valdir), cache_path)

    print("Creating data loaders")
    if args.distributed:
        if hasattr(args, "ra_sampler") and args.ra_sampler:
            train_sampler = RASampler(dataset, shuffle=True, repetitions=args.ra_reps)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler


def main(args):
    if args.seed is not None:
        utils.set_seeds(args.seed)

    if args.output_dir:
        if os.path.exists(os.path.join(args.output_dir, "best_model.pth")) and args.no_overwrite and not args.resume:
            print("There already exists an output folder with models that would be overwritten if training went forward"
                  "but the argument no_overwrite is true so training is aborted")
            return
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    if args.dataset == "cifar10" or args.dataset == "cifar100":
        dataset, dataset_test, train_sampler, test_sampler = load_cifar_data(args)
    else:
        train_dir = os.path.join(args.data_path, "train")
        val_dir = os.path.join(args.data_path, "val")
        dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, args)

    collate_fn = None
    num_classes = len(dataset.classes)
    mixup_transforms = []
    if args.mixup_alpha > 0.0:
        mixup_transforms.append(transforms.RandomMixup(num_classes, p=1.0, alpha=args.mixup_alpha))
    if args.cutmix_alpha > 0.0:
        mixup_transforms.append(transforms.RandomCutmix(num_classes, p=1.0, alpha=args.cutmix_alpha))
    if mixup_transforms:
        mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)

        def collate_fn(batch):
            return mixupcutmix(*default_collate(batch))

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
        prefetch_factor=args.prefetch_factor
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=True
    )

    print("Creating model")
    if args.model == "resnet50":
        model = torchvision.models.resnet50(weights=args.weights, num_classes=num_classes)
    elif args.model == "resnet18":
        if args.dataset == "imagenet":
            model = torchvision.models.resnet18(weights=args.weights, num_classes=num_classes)
        elif args.dataset == "cifar10" or "cifar100":
            model = resnet18(num_classes=num_classes)
        else:
            raise NotImplementedError
    elif args.model == "vgg11":
        model = torchvision.models.vgg11(weights=args.weights, num_classes=num_classes)
    elif args.model == "vgg16":
        model = torchvision.models.vgg16(weights=args.weights, num_classes=num_classes)
        module = nn.AvgPool2d(kernel_size=2, stride=2)
        utils.strip(model, "maxpool", module, nn.MaxPool2d)
    elif args.model == "vgg16bnmax":
        model = ModulusNet_vgg16_bn(pool="max", num_class=num_classes)
    elif args.model == "vgg16bnavg":
        model = ModulusNet_vgg16_bn(pool="avg", num_class=num_classes)
    elif args.model == "minionn":
        model = MiniONN(use_batch_norm=False, to_quant=False)
    elif args.model == "minionnbn":
        model = MiniONN(use_batch_norm=True, to_quant=False)
    elif args.model == "resnet32":
        model = resnet32(num_classes=num_classes)
    elif args.model == "resnet110":
        model = resnet110(num_classes=num_classes)
    elif args.model == "convnet":
        model = ConvNet(num_classes=num_classes)
    else:
        raise NotImplementedError
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()

    print(f'[Network] Total number of parameters : {num_params:.3e}')
    if args.use_poly:
        RANGE = args.range
        CLIP = args.clip
        ALPHA = args.alpha
        if not os.path.exists("./poly_coefs"):
            os.mkdir(os.path.join("./poly_coefs"))
        coeffs = generate_coeffs(real_relu, degree=args.degree, file_prefix="./poly_coefs", rng=args.range,
                                 quantized_coef=not args.no_coef_quant, crypto_precision=args.crypto_precision)
        print(coeffs)
        module = PolyRelu(coeffs, range=RANGE, alpha=ALPHA, regularize=args.regularize) if not args.clip else \
            PolyRelu(coeffs, range=RANGE, clip=CLIP, regularize=args.regularize)
        utils.strip(model, "relu", module, nn.ReLU)

    model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
    if args.transformer_embedding_decay is not None:
        for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
            custom_keys_weight_decay.append((key, args.transformer_embedding_decay))
    parameters = utils.set_weight_decay(
        model,
        args.weight_decay,
        norm_weight_decay=args.norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
    )

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
        )
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )

    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    model_ema = None
    best_acc = 0.
    best_acc_no_nan = 0.
    best_ema_acc = 0.
    best_ema_acc_no_nan = 0.
    if args.model_ema:
        # Decay adjustment that aims to keep the decay independent from other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and ommit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        try:
            best_acc = checkpoint["best_acc"]
        except:
            # no best_acc saved
            print("Couldn't find best_acc in checkpoint")
            best_acc = 0.
        try:
            best_acc_no_nan = checkpoint["best_acc_no_nan"]
        except:
            # no best_acc saved
            print("Couldn't find best_acc_no_nan in checkpoint")
            best_acc_no_nan = 0.
        if model_ema:
            model_ema.load_state_dict(checkpoint["model_ema"])
            try:
                best_ema_acc = checkpoint["best_ema_acc"]
            except:
                print("Couldn't find best_ema_acc in checkpoint")
                best_ema_acc = 0.
            try:
                best_ema_acc_no_nan = checkpoint["best_ema_acc_no_nan"]
            except:
                print("Couldn't find best_ema_acc in checkpoint")
                best_ema_acc_no_nan = 0.
        if scaler:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if model_ema:
            evaluate(model_ema, criterion, data_loader_test, args=args, device=device, log_suffix="EMA")

        evaluate(model, criterion, data_loader_test, args=args, device=device)

        if args.use_poly:
            print("Model loaded metrics")

            if args.dataset == "imagenet":
                device = torch.device("cpu")

            get_metrics(dataloader=data_loader_test, model=model, device=device, input_range=(-args.range, args.range),
                        metrics=["oor_percentage", "num_oor_total", "unique_image_with_oor_total", 'oor_max',
                                 'oor_values'])
            if model_ema:
                print("Model loaded EMA model metrics")
                checkpoint = torch.load(os.path.join(args.output_dir, f"best_ema_model.pth"), map_location="cpu")
                model_ema.load_state_dict(checkpoint["model_ema"])
                get_metrics(dataloader=data_loader_test, model=model_ema, device=device,
                            input_range=(-args.range, args.range),
                            metrics=["oor_percentage", "num_oor_total", "unique_image_with_oor_total", 'oor_max',
                                     'oor_values'])
        return

    print("Start training")
    start_time = time.time()
    has_not_died = True

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        if args.regularize:
            has_not_died = train_and_regularize_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args,
                                                          model_ema, scaler)
        else:
            has_not_died = train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema, scaler)
        if not has_not_died:
            break
        lr_scheduler.step()
        acc, loss = evaluate(model, criterion, data_loader_test, args=args, device=device)
        if acc > best_acc:
            best_acc = acc
            if args.output_dir:
                checkpoint = {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "args": args,
                    "best_acc": best_acc,
                    "best_acc_no_nan": best_acc_no_nan
                }
                if model_ema:
                    checkpoint["model_ema"] = model_ema.state_dict()
                    checkpoint["best_ema_acc"] = best_ema_acc
                    checkpoint["best_ema_acc_no_nan"] = best_ema_acc_no_nan
                if scaler:
                    checkpoint["scaler"] = scaler.state_dict()
                utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"best_model.pth"))

        if acc > best_acc_no_nan:
            if not np.isnan(loss) and loss < 10:   # some arbitrary value where we it might still blow up in encrypted
                best_acc_no_nan = acc
                if args.output_dir:
                    checkpoint = {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "args": args,
                        "best_acc": best_acc,
                        "best_acc_no_nan": best_acc_no_nan
                    }
                    if model_ema:
                        checkpoint["model_ema"] = model_ema.state_dict()
                        checkpoint["best_ema_acc"] = best_ema_acc
                        checkpoint["best_ema_acc_no_nan"] = best_ema_acc_no_nan
                    if scaler:
                        checkpoint["scaler"] = scaler.state_dict()
                    utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"best_model_no_nan.pth"))

        if model_ema:
            ema_acc, ema_loss = evaluate(model_ema, criterion, data_loader_test, args=args, device=device, log_suffix="EMA")
            if ema_acc > best_ema_acc:
                best_ema_acc = ema_acc
                if args.output_dir:
                    checkpoint = {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "args": args,
                        "best_acc": best_acc,
                        "best_acc_no_nan": best_acc_no_nan
                    }
                    checkpoint["model_ema"] = model_ema.state_dict()
                    checkpoint["best_ema_acc"] = best_ema_acc
                    checkpoint["best_ema_acc_no_nan"] = best_ema_acc_no_nan
                    if scaler:
                        checkpoint["scaler"] = scaler.state_dict()
                    utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"best_ema_model.pth"))
            if ema_acc > best_ema_acc_no_nan:
                if not np.isnan(ema_loss) and loss < 10:
                    best_ema_acc_no_nan = ema_acc
                    if args.output_dir:
                        checkpoint = {
                            "model": model_without_ddp.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "epoch": epoch,
                            "args": args,
                            "best_acc": best_acc,
                            "best_acc_no_nan": best_acc_no_nan
                        }
                        checkpoint["model_ema"] = model_ema.state_dict()
                        checkpoint["best_ema_acc"] = best_ema_acc
                        checkpoint["best_ema_acc_no_nan"] = best_ema_acc_no_nan
                        if scaler:
                            checkpoint["scaler"] = scaler.state_dict()
                        utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"best_ema_model_no_nan.pth"))

        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
                "best_acc": best_acc,
                "best_acc_no_nan": best_acc_no_nan
            }
            if model_ema:
                checkpoint["model_ema"] = model_ema.state_dict()
                checkpoint["best_ema_acc"] = best_ema_acc
                checkpoint["best_ema_acc_no_nan"] = best_ema_acc_no_nan
            if scaler:
                checkpoint["scaler"] = scaler.state_dict()
            if args.save_every_epoch:
                utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

    if args.use_poly:
        if args.dataset == "imagenet":
            device = torch.device("cpu")
        if has_not_died:
            print("Last model metrics")
            get_metrics(dataloader=data_loader_test, model=model, device=device, input_range=(-args.range, args.range),
                        metrics=["oor_percentage", "num_oor_total", "unique_image_with_oor_total", 'oor_max', 'oor_values'])
        if os.path.exists(os.path.join(args.output_dir, "best_ema_model_no_nan.pth")):
            print("Best EMA model metrics")
            checkpoint = torch.load(os.path.join(args.output_dir, f"best_ema_model_no_nan.pth"), map_location="cpu")
            model_ema.load_state_dict(checkpoint["model_ema"])
            get_metrics(dataloader=data_loader_test, model=model_ema, device=device, input_range=(-args.range, args.range),
                        metrics=["oor_percentage", "num_oor_total", "unique_image_with_oor_total", 'oor_max', 'oor_values'])
        if os.path.exists(os.path.join(args.output_dir, f"best_model_no_nan.pth")):
            print("Best model metrics")
            checkpoint = torch.load(os.path.join(args.output_dir, f"best_model_no_nan.pth"), map_location="cpu")
            model_without_ddp.load_state_dict(checkpoint["model"])
            get_metrics(dataloader=data_loader_test, model=model, device=device, input_range=(-args.range, args.range),
                        metrics=["oor_percentage", "num_oor_total", "unique_image_with_oor_total", 'oor_max', 'oor_values'])
    else:
        print("Testing")
        acc, loss = evaluate(model, criterion, data_loader_test, args=args, device=device)
        print(f"Final accuracy: {acc}")
    print("Best accuracy")
    print(best_acc)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    parser.add_argument("--data-path", type=str, help="dataset path")
    parser.add_argument("--model", default="resnet50", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=32, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=90, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=8, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--bias-weight-decay",
        default=None,
        type=float,
        help="weight decay for bias parameters of all layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--transformer-embedding-decay",
        default=None,
        type=float,
        help="weight decay for embedding parameters for vision transformer models (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing"
    )
    parser.add_argument("--mixup-alpha", default=0.0, type=float, help="mixup alpha (default: 0.0)")
    parser.add_argument("--cutmix-alpha", default=0.0, type=float, help="cutmix alpha (default: 0.0)")
    parser.add_argument("--lr-scheduler", default="steplr", type=str, help="the lr scheduler (default: steplr)")
    parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument(
        "--lr-warmup-method", default="constant", type=str, help="the warmup method (default: constant)"
    )
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--lr-min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy (default: None)")
    parser.add_argument("--ra-magnitude", default=9, type=int, help="magnitude of auto augment policy")
    parser.add_argument("--augmix-severity", default=3, type=int, help="severity of augmix policy")
    parser.add_argument("--random-erase", default=0.0, type=float, help="random erasing probability (default: 0.0)")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument(
        "--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters"
    )
    parser.add_argument(
        "--model-ema-steps",
        type=int,
        default=32,
        help="the number of iterations that controls how often to update the EMA model (default: 32)",
    )
    parser.add_argument(
        "--model-ema-decay",
        type=float,
        default=0.99998,
        help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",
    )
    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )
    parser.add_argument(
        "--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)"
    )
    parser.add_argument(
        "--val-resize-size", default=256, type=int, help="the resize size used for validation (default: 256)"
    )
    parser.add_argument(
        "--val-crop-size", default=224, type=int, help="the central crop size used for validation (default: 224)"
    )
    parser.add_argument(
        "--train-crop-size", default=224, type=int, help="the random crop size used for training (default: 224)"
    )
    parser.add_argument("--clip-grad-norm", default=None, type=float, help="the maximum gradient norm (default None)")
    parser.add_argument("--ra-sampler", action="store_true", help="whether to use Repeated Augmentation in training")
    parser.add_argument(
        "--ra-reps", default=3, type=int, help="number of repetitions for Repeated Augmentation (default: 3)"
    )
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument(
        "--use-poly",
        dest="use_poly",
        help="Use polynomial approximation of relu",
        action="store_true",
    )
    parser.add_argument(
        "--clip",
        dest="clip",
        help="Use clipping (default is annealed switching)",
        action="store_true",
    )
    parser.add_argument(
        "--coeffs", default="coeffs.p", type=str, help="the polynomial coefficients file"
    )
    parser.add_argument("-d", "--degree", default=6, type=int, help="degree of polynomial")
    parser.add_argument("-r", "--range", default=5.0, type=float, help="range of polynomial")
    parser.add_argument("--reg_range", default=4.8, type=float, help="range used for regularization")
    parser.add_argument("-c", "--reg_coef", default=0.001, type=float, help="the coefficient for regularization")
    parser.add_argument("--penalty_exp", default=10, type=int, help="the exponent for the penalty function")
    parser.add_argument(
        "--regularize",
        dest="regularize",
        help="Use regularization ",
        action="store_true",
    )
    parser.add_argument("--dataset", type=str, default="imagenet")
    parser.add_argument("--alpha", default=0., type=float, help="range of polynomial")
    parser.add_argument("--prefetch-factor", dest="prefetch_factor", default=2, type=int,
                        help="Number of batches loaded in advance by each worker. 2 means there will be a total of "
                             "2 * num_workers batches prefetched across all workers.")
    parser.add_argument("--no-overwrite", action="store_true", help="do not run if models are already in the output"
                                                                    "folder",
                        dest="no_overwrite")
    parser.add_argument("--save-every-epoch", action="store_true", dest="save_every_epoch")
    parser.add_argument("--no-coef-quant", action="store_true", dest="no_coef_quant")
    parser.add_argument("--no-reg-warmup", action="store_true", dest="no_reg_warmup")
    parser.add_argument("--seed", default=None, type=int, help="randomness seed, if None, no seed is set")
    parser.add_argument("--crypto-precision", dest="crypto_precision", default=8, type=int,
                        help="number of bits to use for quantization, in particular quantized coefficient generation")
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
