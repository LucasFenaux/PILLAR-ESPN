import os.path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy import interpolate

# models = ["resnet18", "resnet32", "resnet50", "resnet110", "minionn", "vgg16_bn"]
# datasets = ["cifar10", "cifar100"]
datasets = ["cifar10", "cifar100", "imagenet"]
configs = ["default12", "crypten12", "honeybadger12", "espn12", "gforce", "coinn"]
delay_range = np.arange(0, 0.055, 0.005)
colors = ["r", "g", "k", "m", "b"]
styles = ["-", "--"]
devices = ["cpu", "cuda:0"]

for dataset in datasets:
    models = os.listdir(f"results/{dataset}")
    for _model in models:
        model = "_".join(_model.split("_")[:-4])
        device = "cpu" if _model.endswith("cpu") else "cuda"
        plot_label = f"{dataset}_{model}_{device}"
        # create a plot
        for c, config in enumerate(configs):
            label = f"{config}_{device}"
            results_file = f"results/{dataset}/{_model}/{config}_result.yaml"
            if os.path.exists(results_file):
                with open(results_file, "r") as f:
                    results = yaml.safe_load(f)
                fmt_str = colors[c]
                fmean = interpolate.interp1d(results['delays'], results['run_time'], fill_value='extrapolate')
                flow = interpolate.interp1d(results['delays'], results['run_time_95conf_lower'],
                                            fill_value='extrapolate')
                fupp = interpolate.interp1d(results['delays'], results['run_time_95conf_upper'],
                                            fill_value='extrapolate')

                plt.plot(2000 * delay_range, fmean(delay_range), fmt_str, label=label)
                plt.fill_between(2000 * delay_range, flow(delay_range), fupp(delay_range),
                                 color=colors[c], alpha=0.3)
            else:
                print(results_file + " is not found")
        # save plots and reset
        plt.title(f"Performance of {model} on {dataset}")
        plt.xlabel("Artificial roundtrip delay in ms")
        plt.ylabel("Total inference runtime in seconds")
        plt.legend()
        plt.savefig("plots/" + plot_label + ".png")
        plt.close()
