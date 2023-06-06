# PILLAR

## Reproducing the paper figures

To reproduce the figures related to PILLAR in the paper, follow these steps. Given a figure k (from 3, 4, 5, 11):
First run the associated bash script

```bash
cd paper_exp
bash figure_k.sh
```

Then run the cells in the associated python notebook <code>figure_k.ipynb</code>.

## Reproducing the paper results

To reproduce the accuracy results from the paper, we provided training scripts that train the models and then the 
associated testing scripts to extract the results.
We have 3 datasets: CIFAR10, CIFAR100 and ImageNet. For ImageNet, make sure it is already downloaded and at the location
that <code>dataset_path</code> point to in the bash scripts (feel free to change that for your own purposes).

### Training PolyReLU models
#### CIFAR10

To train our polynomial approximated relu models on CIFAR10, you can run:

```bash
gpu=0  # the gpu device to use to run the experiment
model=1  # which of the models you want to train
cd training_scripts
bash cifar10_polyrelu_runs.sh "${gpu}" "${model}"
```

Each model architecture is associated with a model number for the script in the following way:

| Bash script Model # | Model Architecture | Output Directory            |
|---------------------|--------------------|-----------------------------|
| 1                   | ResNet18           | cifar10_resnet18_polyrelu   |
| 2                   | ResNet110          | cifar10_resnet110_polyrelu  |
| 3                   | MinionnBN          | cifar10_minionnbn_polyrelu  |
| 4                   | VGG16BNAVG         | cifar10_vgg16bnavg_polyrelu |

For the original relu model architectures (labeled as CryptGPU's models in the paper), the same association applies 
with the corresponding script:

```bash
gpu=0  # the gpu device to use to run the experiment
model=1  # which of the models you want to train
cd training_scripts
bash cifar10_relu_runs.sh "${gpu}" "${model}"
```

To get a summary of the results (without all the training logging), one can <code>cd</code> into the
<code>testing_scripts</code> directory and run the associated scripts:

```bash
gpu=0  # the gpu device to use to run the experiment
model=1  # which of the models you want to test
cd testing_scripts
bash cifar10_polyrelu_runs.sh "${gpu}" "${model}"  # for the PolyReLU model
bash cifar10_relu_runs.sh "${gpu}" "${model}"  # for the ReLU model
```

#### CIFAR100

To train our polynomial approximated relu models on CIFAR100, you can run:

```bash
gpu=0  # the gpu device to use to run the experiment
model=1  # which of the models you want to train
cd training_scripts
bash cifar100_polyrelu_runs.sh "${gpu}" "${model}"
```

Each model architecture is associated with a model number for the script in the following way:

| Bash script Model # | Model Architecture | Output Directory             |
|---------------------|--------------------|------------------------------|
| 1                   | ResNet18           | cifar100_resnet18_polyrelu   |
| 2                   | ResNet32           | cifar100_resnet32_polyrelu   |
| 3                   | VGG16BNAVG         | cifar100_vgg16bnavg_polyrelu |


For the original relu model architectures (labeled as CryptGPU's models in the paper), the same association applies 
with the corresponding script:

```bash
gpu=0  # the gpu device to use to run the experiment
model=1  # which of the models you want to train
cd training_scripts
bash cifar100_relu_runs.sh "${gpu}" "${model}"
```

To get a summary of the results (without all the training logging), one can <code>cd</code> into the
<code>testing_scripts</code> directory and run the associated scripts:

```bash
gpu=0  # the gpu device to use to run the experiment
model=1  # which of the models you want to test
cd testing_scripts
bash cifar100_polyrelu_runs.sh "${gpu}" "${model}"  # for the PolyReLU model
bash cifar100_relu_runs.sh "${gpu}" "${model}"  # for the ReLU model
```


#### ImageNet

For ImageNet, the script is a little different as we only trained one model due to the time it takes to train said model.
We trained our model using 32 CPU cores @3.7GHz and 1TB of RAM with four NVIDIA A100 with 80GB of memory. To get results
as close to the paper as possible, we recommend training the model with 4 GPUs as well. The script only works with 4 
GPUs but can easily be tweaked to use more/less by changing:

```bash
export CUDA_VISIBLE_DEVICES=$1,$2,$3,$4
```

to use as few/as many as necessary and then updating <code>--nproc_per_node=</code> to reflect the correct number of 
GPUs to use.
It is worth noting that due to how Pytorch's distributed training works, the number of GPUs used directly affects the 
effective batch size. We have in our case an effective batch size of 4x256 = 1024. Changing the gpu size/batch size 
without updating the other can therefore change the results.

```bash
export OMP_NUM_THREADS=4
```
Sets the number of threads to be used per-gpu process and can be tuned depending on your system.

Our script should be run in the following way:

```bash
bash imagenet_resnet50.sh 0 1 2 3
```

and outputs its results in the following directory: <code>imagenet_resnet50</code>.

Note: Even with good gpus and cpus, the ImageNet training can take quite a while (multiple days).

One can get a summary of the metrics and results using the following script located in the <code>testing_scripts</code>
directory:

```bash
gpu=0
cd testing_scripts
bash imagenet_resnet50.sh "${gpu}"
```

The testing script itself only uses a singular gpu. It is worth noting that due to the higher memory usage (up to 
200-300GB of RAM at any given time), the metrics are computed using CPUs rather than GPUs (our GPUs did not have quite
that much RAM). Therefore, the computations are very slow, and it takes us about ~8h to run to completion. This is due
to some  metrics being computed over the entire ImageNet dataset and therefore needing to store all the 
intermediate values.