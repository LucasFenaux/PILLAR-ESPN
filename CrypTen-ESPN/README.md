# CrypTen-ESPN

## Setup
If you did not install the environment from the parent repo
```bash
conda env create -f env.yml
conda activate pillar_espn
pip install -r requirements.txt
```

Otherwise, just activate the conda environment
```bash
conda activate pillar_espn
```
## Timing Experiments
For running the timing experiments, you don't need any model files, we just instantiate a random model and run inferences.
```bash
export PYTHONPATH=$PWD
python scripts/timing_experiments.py
```
You can edit the parameters at the top of [timing_experiments.py](https://github.com/D-Diaa/CrypTen-ESPN/blob/main/scripts/timing_experiments.py) to select different models, datasets, batch sizes, number of repeats per experiment and simulated delays.
```python
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

```
You can also edit the configs (default is BinaryShares Relu, crypten is CrypTen Polynomial Evaluation). The rest are as described in the paper.
```python
configs = ["espn12.yaml", "default12.yaml", "crypten12.yaml", "honeybadger12.yaml"]
```
## Accuracy Experiments
Similary, you can run all the accuracy experiments:
```bash
export PYTHONPATH=$PWD
python scripts/accuracy_experiments.py
```
However, for this you will need to provide a models_folder at the top of [the file](https://github.com/D-Diaa/CrypTen-ESPN/blob/main/scripts/accuracy_experiments.py):
```python
models_folder = "/home/paper_models"
```
Where models_folder is a path to all the models following this format:
```python
model_file = f"{models_folder}/{dataset}/{model_folder}/run_{run}/best_model.pth"
```

Alternatively, you can run the models one by one in the following way:
```shell
python examples/mpc_inference/launcher.py --multiprocess --world_size 2 \
--skip-plaintext \
--batch-size 100 \
--delays 0 \
--dataset {dataset} \
--model-type {model} \
--resume \
--model-location {model_file} \
--config {path_to_config} \
--use-cuda
```
Where {model} is any of
```python
["resnet110", "minionn_bn", "vgg16_avg_bn", "resnet18", "resnet32", "resnet50"]
```
## Results
All the results will be in "results" folder, you can use [plots.ipynb](https://github.com/D-Diaa/CrypTen-ESPN/blob/main/notebooks/plots.ipynb) to generate all the timing plots. You can inspect the config_result.yaml to see the accuracy, match-rate and different statistics about communication rounds, bytes, etc.