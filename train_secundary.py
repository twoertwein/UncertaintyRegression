#!/usr/bin/env python3
import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
from python_tools import caching
from python_tools.generic import namespace_as_string
from python_tools.ml.data_loader import DataLoader
from python_tools.ml.pytorch_tools import dict_to_batched_data

from train import train


def get_data(training: Path, transformation: Path):
    results = {
        key: caching.read_pickle(
            training.parent / training.name.replace("training", key)
        )[0]
        for key in ("training", "evaluation", "test")
    }

    # meta data
    x_names = ["guess"] + [
        f"embedding_{i}" for i in range(results["training"]["meta_embedding"].shape[1])
    ]
    meta_data = {"y_names": np.array(["loss"]), "x_names": np.array(x_names)}

    # apply transformation
    transformation = caching.read_pickle(transformation)[0][0]["y"]
    for dataset in results:
        for key in ("y", "y_hat"):
            results[dataset][key] = (
                results[dataset][key] - transformation["mean"]
            ) / transformation["std"]

    # generate data
    datasets = {}
    for key, data in results.items():
        dataset = {
            "x": np.concatenate([data["y_hat"], data["meta_embedding"]], axis=1),
            "y": np.abs(data["y_hat"] - data["y"]),
            "meta_id": data["meta_id"],
            "meta_frame": data["meta_frame"],
            "meta_y_hat": data["y_hat"],
            "meta_y": data["y"],
        }

        datasets[key] = DataLoader(
            dict_to_batched_data(dataset), properties=deepcopy(meta_data)
        )

    return {0: datasets}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["disfa", "bp4d_plus", "mnist", "mnisti"])
    parser.add_argument("--uncertainty", choices=["umlp", "dwar"], default="umlp")
    parser.add_argument("--method", choices=[""], default="")
    parser.add_argument("--au", type=int, default=6)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    # find best model
    primary_folder = Path(f"method=dropout_dataset={args.dataset}_au={args.au}")
    path = next(primary_folder.glob("*_training_predictions.pickle"))

    # get data
    data = get_data(path, primary_folder / "partition_0.pickle")

    # train
    folder = Path(namespace_as_string(args, exclude=("workers",)))
    train(data, folder, args)
