# UncertaintyRegression
The code for "Simple and Effective Approaches for Uncertainty Prediction in Facial Action Unit Intensity Regression" at FG 2020

All building blocks are part of [this repository](https://bitbucket.org/twoertwein/python-tools/src/master/):

* [DWAR](https://bitbucket.org/twoertwein/python-tools/src/e98707bbb102f775b1c9632bf8a0c72af83c9af1/ml/neural.py#lines-620)
* [GP-VFE](https://bitbucket.org/twoertwein/python-tools/src/e98707bbb102f775b1c9632bf8a0c72af83c9af1/ml/neural.py#lines-736)
* [MLP ensemble](https://bitbucket.org/twoertwein/python-tools/src/e98707bbb102f775b1c9632bf8a0c72af83c9af1/ml/neural.py#lines-1066)
* [Loss Attenuation](https://bitbucket.org/twoertwein/python-tools/src/e98707bbb102f775b1c9632bf8a0c72af83c9af1/ml/neural.py#lines-183)
* [MLP Model](https://bitbucket.org/twoertwein/python-tools/src/e98707bbb102f775b1c9632bf8a0c72af83c9af1/ml/neural.py#lines-976) for dropout, U-MLP, and the Multi-Task MLP


## Installation
```sh
poetry add git+https://github.com:twoertwein/UncertaintyRegression.git
poetry run pip install torchvision==0.9.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html
```

## Usage
Run the grid-search for a primary model to predict facial action units (AU) intensities. Some primary models can also estimate their uncertainty.
```sh
python train.py --method dropout --workers 4 --dataset mnist
```
Train a secondary model to estimate the uncertainty of the primary dropout model.
```sh
python train.py --uncertainty umlp --workers 4 --dataset mnist
```
