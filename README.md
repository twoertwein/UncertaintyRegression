# UncertaintyRegression
The code for ["Simple and Effective Approaches for Uncertainty Prediction in Facial Action Unit Intensity Regression"](https://par.nsf.gov/biblio/10169266-simple-effective-approaches-uncertainty-prediction-facial-action-unit-intensity-regression) at FG 2020

All building blocks are part of [this repository](https://bitbucket.org/twoertwein/python-tools/src/master/):

* [DWAR](https://bitbucket.org/twoertwein/python-tools/src/ddd6ea310f5eecf3c2998c67fe8a11b8c36e6810/python_tools/ml/nonparametric.py#lines-13)
* [GP-VFE](https://bitbucket.org/twoertwein/python-tools/src/ddd6ea310f5eecf3c2998c67fe8a11b8c36e6810/python_tools/ml/nonparametric.py#lines-264)
* [MLP ensemble](https://bitbucket.org/twoertwein/python-tools/src/ddd6ea310f5eecf3c2998c67fe8a11b8c36e6810/python_tools/ml/neural.py#lines-1450)
* [Loss Attenuation](https://bitbucket.org/twoertwein/python-tools/src/ddd6ea310f5eecf3c2998c67fe8a11b8c36e6810/python_tools/ml/neural.py#lines-679)
* [MLP Model](https://bitbucket.org/twoertwein/python-tools/src/ddd6ea310f5eecf3c2998c67fe8a11b8c36e6810/python_tools/ml/neural.py#lines-959) for dropout, U-MLP, and the Multi-Task MLP


## Installation
```sh
git clone git@github.com:twoertwein/UncertaintyRegression.git
cd UncertaintyRegression
poetry update
poetry run pip install torchvision --extra-index-url https://download.pytorch.org/whl/cu116
```

## Usage
Run the grid-search for a primary model to predict facial action units (AU) intensities. Some primary models can also estimate their uncertainty.
```sh
python train.py --method dropout --workers 4 --dataset mnist
```
Train a secondary model to estimate the uncertainty of the primary dropout model.
```sh
python train_secundary.py --uncertainty umlp --workers 4 --dataset mnist
```
