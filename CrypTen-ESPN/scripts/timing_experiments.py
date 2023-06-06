import os

config_folder = "configs"
batch_size = 1
repeats = 20

# [LAN, MIDDLE, WAN]
delays = ["0.000125", "0.025", "0.05"]

datasets = ["cifar10", "cifar100", "imagenet"]

models = {
    "cifar10": ["resnet110", "minionn_bn", "vgg16_avg_bn", "resnet18"],
    "cifar100": ["resnet32", "vgg16_avg_bn", "resnet18"],
    "imagenet": ["resnet50"]
}

configs = ["espn12.yaml", "default12.yaml", "crypten12.yaml", "honeybadger12.yaml"]

device_commands = ["--use-cuda"]

base_command = f"python3 examples/mpc_inference/launcher.py --multiprocess --world_size 2 " \
               f" --skip-plaintext " \
               f" --batch-size {batch_size} " \
               f" --n-batches {repeats}"

base_command += " --delays " + " ".join(delays)

for dataset in datasets:
    for model in models[dataset]:
        cmd = base_command + f" --dataset {dataset}" \
                             f" --model-type {model} "
        for device_cmd in device_commands:
            for config in configs:
                cmd += f" --config {config_folder}/{config} {device_cmd}"
                os.system(cmd)
