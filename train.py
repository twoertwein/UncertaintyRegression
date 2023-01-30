#!/usr/bin/env python3
import argparse
import shutil
from functools import partial
from pathlib import Path

import numpy as np
from python_tools.generic import namespace_as_string
from python_tools.ml import metrics
from python_tools.ml.default.neural_models import EnsembleModel, MLPModel
from python_tools.ml.default.transformations import (
    DefaultTransformations,
    revert_transform,
    set_transform,
)
from python_tools.ml.evaluator import evaluator

from dataloader import BP4D_PLUS, DISFA, MNIST


def train(partitions: dict[str, DISFA], folder: Path, args: argparse.Namespace) -> None:
    params = {"interval": True, "metric_max": True, "y_names": np.array(["intensity"])}

    model = MLPModel(device="cuda", **params)
    grid_search = {
        "epochs": [5000],
        "early_stop": [50],
        "lr": [0.01, 0.001, 0.0001, 0.00001],
        "dropout": [0.0, 0.5],
        "layers": [0, 1, 2, 3],
        "activation": [{"name": "ReLU"}],
        "attenuation": [""],
        "sample_weight": [True],
    }
    if args.method == "gp":
        grid_search["final_activation"] = [
            {"name": "gpvfe", "embedding_size": 2, "inducing_points": 2000}
        ]
    else:
        grid_search["final_activation"] = [{"name": "linear"}]
    if args.method == "attenuation":
        grid_search["attenuation"] = ["gaussian"]
    elif args.method == "dropout":
        grid_search["dropout"] = [0.5]
    elif args.method == "ensemble":
        model = EnsembleModel(device="cuda", **params)
        for key in ("layers", "activation", "dropout"):
            grid_search[f"model_{key}"] = grid_search.pop(key)
            model.parameters.pop(key)

    model.parameters.update(grid_search)

    models, parameters, model_transform = model.get_models()

    apply_transformation = partial(
        combine_transformations, model_transform=model_transform
    )

    transform = DefaultTransformations(**params)

    transforms = tuple([{}] * len(partitions))
    kwargs = {
        "parallel": "local",
        "n_workers": args.workers,
        "workers": args.workers,
    }
    print(folder)
    evaluator(
        models=models,
        partitions=partitions,
        parameters=parameters,
        folder=folder,
        metric_fun=metrics.interval_metrics,
        metric="ccc",
        metric_max=params["metric_max"],
        learn_transform=transform.define_transform,
        apply_transform=apply_transformation,
        revert_transform=revert_transform,
        transform_parameter=transforms,
        **kwargs,
    )


def combine_transformations(data, transform, model_transform=None):
    data = set_transform(data, transform)
    data.add_transform(model_transform, optimizable=True)
    return data


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser()
    au_flags = [
        "1",
        "2",
        "4",
        "5",
        "6",
        "9",
        "10",
        "12",
        "14",
        "15",
        "17",
        "20",
        "25",
        "26",
    ]
    for name in au_flags + ["transfer"]:
        parser.add_argument(
            f"--{name}", action="store_const", const=True, default=False
        )
    parser.add_argument(
        "--method", choices=["dropout", "attenuation", "gpvfe", "ensemble"]
    )
    parser.add_argument("--dataset", choices=["disfa", "bp4d_plus", "mnist", "mnisti"])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.transfer:
        assert args.dataset == "bp4d_plus"
    arg_aus = []
    for au in au_flags:
        if getattr(args, au):
            arg_aus.append(int(au))

    # choose dataloader
    folds = 1
    aus = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]

    def backend(au, fold, name):
        return DISFA(au, ifold=fold, name=name).get_loader()

    if args.dataset == "bp4d_plus":
        aus = [6, 10, 12, 14, 17]

        def backend(au, fold, name):
            return BP4D_PLUS(au, ifold=fold, name=name).get_loader()

    elif args.dataset.startswith("mnist"):
        aus = [6]

        def backend(au, fold, name):
            return MNIST(
                au, ifold=fold, name=name, imbalance=args.dataset == "mnisti"
            ).get_loader()

    if args.transfer:
        aus = [6, 12, 17]
        assert args.dataset == "bp4d_plus"

        def backend(au, fold, name):
            if name == "test":
                return DISFA(au, ifold=fold, name=name).get_loader()
            return BP4D_PLUS(au, ifold=fold, name=name).get_loader()

    # run on subset of AUs
    if arg_aus:
        aus = [au for au in aus if au in arg_aus]

    for au in aus:
        print("AU", au)
        folder = Path(namespace_as_string(args, exclude=("workers",)) + f"_au={au}")

        if args.transfer:
            # copy BP4D+ models
            args.transfer = False
            folder_bp4d_plus = Path(
                namespace_as_string(args, exclude=("workers",)) + f"_au={au}"
            )
            args.transfer = True
            if not folder.is_dir():
                shutil.copytree(folder_bp4d_plus, folder)

        data = {
            i: {
                name: backend(au, i, name)
                for name in ("training", "validation", "test")
            }
            for i in range(folds)
        }
        train(data, folder, args)
