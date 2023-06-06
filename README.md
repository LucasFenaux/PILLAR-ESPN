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