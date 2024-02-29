## PILLAR-ESPN
This repo includes the implementation code for the paper:
***Fast and Private Inference of Deep Neural Networks by Co-designing Activation Functions***

## The environment for both PILLAR and CrypTen-ESPN

This environment contains all the packages required to run the code for both PILLAR and CrypTen-ESPN.

```bash
conda env create -f env.yml
conda activate pillar_espn
pip install -r requirements.txt
```

### PILLAR
For all model training related scripts and code, see the <code>pillar</code> directory (and its README.md).

### CrypTen-ESPN
For all secure inference related scripts and code, see the <code>CrypTen-ESPN</code> directory (and its README.md).

### GFORCE
For all the gforce (related work we compare against) scripts and code, see the <code>gforce</code> directory (and its
README.md). 
The instructions for the environment and packages for gforce are in its README.md.


### Implementation Details

We implement PILLAR using [GEKKO](https://gekko.readthedocs.io/en/latest/)'s mixed integer linear programming solver for the quantization-aware polynomial fitting. We then cache these polynomial coefficients and use them for all models and datasets. 
We train all our models using PyTorch and implement a custom activation function module for the polynomial approximations.
We call this the PolyReLU layer and use it to replace all ReLU layers. 
This module takes the cached coefficients and computes the output of the polynomial approximation for forward passes.
When training, we compute the regularization penalty by taking a snapshot of the inputs passed to the PolyReLU layers and computing our modified loss function.
After training, we export the model as an [ONNX](https://onnxruntime.ai/) model which can be inputted to CrypTen.

We implement the Polynomial Evaluation Protocol for 2-party additive secret-sharing with both ESPN and HoneyBadger in the CrypTen interface.
We add configuration parameters that allow the user to specify which type of activation function they would like from ReLU or PolyReLU. Similarly, the polynomial evaluation method is parametrized in the config file between ESPN, HoneyBadger, and the default CrypTen Polynomial Evaluation.
We follow CrypTen's default trusted first party provider, which assumes a first party will generate and distribute all needed pre-computed values from Beaver triplets to binary shares, etc.
To ensure we accurately measure the online phase of the inference, we move some additional operations to the pre-computation phase.
Specifically, random-number generation in the Pseudo-Random-Zero-Sharing (PRZs) are fixed to zero to simulate being computed in the offline phase (for timing only).

For ESPN, we implement the idea directly into CrypTen's polynomial evaluation module.
For HoneyBadger, we modify the Trusted First-Party Provider to provide the additional sets of random numbers and exponents needed for their protocol. We then implement the idea directly into the polynomial module of ArithmeticSharedTensors. We use HoneyBadger's proposed dynamic programming method with the memory optimization technique of keeping only the current and previous iterations in memory.

When running our experiments on the GPU, we observed overflows not present in the CPU version of CrypTen. Upon inspection, we found that the default number of blocks (4) set in CryptGPU is not adequate for our use-case. Increasing this parameter to 5 fixed the overflow issue completely.

### Datasets
We use a collection of common benchmark datasets used in the related work we compare to.
Each dataset represents an image classification task of varying difficulty.
We use the PyTorch version of each dataset with the pre-determined splits into training and test sets.
- CIFAR-10/100 is a collection of $60,000$ images of size 32x32 pixels. CIFAR-10 contains 10 classes, each with 6,000 images (CIFAR-100 has 100 classes each with 600 images). In total, this gives 50,000 training images and 10,000 test images.
- ImageNet contains 1.4 million images of size 256x256 pixels. There are 1000 different classes, each with around 1000 images on average. The training set has 1.2 million images, the validation set contains 50,000 images and 100,000 test images.

### Architectures
We choose the following architectures to facilitate comparisons COINN and GForce. This covers models of depth 7 to 110 layers with the number of trainable parameters ranging from 0.2 to 23 million.
- MiniONN is a smaller convolutional model tailored for secure inference. It contains six convolutional layers and one fully connected layer.
-  VGG-16 is a classic deep convolutional network commonly evaluated in the literature. We use the version evaluated in GForce, which changes the fully-connected layer from 4096 to 512 neurons.
-  ResNet-18, 32, 50, 110. ResNets are convolutional neural networks that introduce skip connections between layers that connect non-adjacent layers. They are the most accurate of all the models we consider. We use various sizes following related work.  
