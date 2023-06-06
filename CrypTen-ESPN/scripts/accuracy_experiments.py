import os

config_folder = "configs"
models_folder = "/home/paper_models"
batch_size = 100

delays = ["0"]

datasets = ["cifar10", "cifar100", "imagenet"]

models = {
    "cifar10": ["resnet110", "minionn_bn", "vgg16_avg_bn", "resnet18"],
    "cifar100": ["resnet32", "vgg16_avg_bn", "resnet18"],
    "imagenet": ["resnet50"]
}

configs = ["honeybadger12.yaml"]  # In case of Poly
# configs = ["default12.yaml"]  # In case of CryptGPU

device_commands = ["--use-cuda"]

runs = range(1, 6)

base_command = f"python3 examples/mpc_inference/launcher.py --multiprocess --world_size 2 " \
               f" --skip-plaintext " \
               f" --batch-size {batch_size} "

base_command += " --delays " + " ".join(delays)

for dataset in datasets:
    for model in models[dataset]:
        for run in runs:
            cmd = None
            for model_folder in os.listdir(f"{models_folder}/{dataset}"):
                if model_folder.startswith(f"{model}"):
                    model_file = f"{models_folder}/{dataset}/{model_folder}/run_{run}/best_model.pth"
                    print(model_file)
                    cmd = base_command + f" --dataset {dataset}" \
                                         f" --model-type {model} " \
                                         f"--resume " \
                                         f"--model-location {model_file}"
                    break
            if cmd is not None:
                for device_cmd in device_commands:
                    for config in configs:
                        cmd += f" --config {config_folder}/{config} {device_cmd}"
                        os.system(cmd)
