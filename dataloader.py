from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np
import pandas as pd
from python_tools import caching, generic
from python_tools.ml.data_loader import DataLoader
from python_tools.ml.pytorch_tools import dict_to_batched_data
from python_tools.ml.split import stratified_splits

prefix = Path("/projects/")
if not prefix.is_dir():
    prefix = Path("/pool01/")
DISFA_FOLDER = prefix / "dataset_original/DISFA/"
BP4D_PLUS_FOLDER = prefix / "dataset_original/BP4D_plus/AUCoding/AU_INT/"
assert DISFA_FOLDER.is_dir()


class DISFA:
    def __init__(
        self,
        au: str,
        ifold: int = 0,
        name: str = "training",
    ) -> None:
        if not isinstance(au, str):
            au = str(au)

        self.n_folds: int = 5
        self.au: str = au
        self.ifold: int = ifold
        self.name: str = name

    def get_loader(self) -> DataLoader:
        subject = self.get_list_of_subjects()[0]
        property_dict = {
            "X_names": [
                x
                for x in self.get_features(subject).columns
                if x.startswith("p_") or x.startswith("hog")
            ],
            "Y_names": ["intensity"],
        }
        for key in property_dict:
            property_dict[key] = np.asarray(property_dict[key])

        # DataLoader will use a copy of it
        self.property_dict = property_dict

        return DataLoader([x for x in self], property_dict=deepcopy(property_dict))

    def __iter__(self) -> Iterator[Dict[str, List[np.ndarray]]]:
        """
        Returns a list of training, evaluation, test folds

        Yields multiple items for evaluation and testing
        """
        x_names = self.property_dict["X_names"]
        # load labels for all subjects
        subjects = self.get_subjects_for_fold()

        for subject in subjects:
            # load openface and combine with labels
            data = self.get_au(subject)[0].join(self.get_features(subject), how="inner")

            # create folds
            data = {
                "X": data.loc[:, x_names].values.astype(np.float32),
                "Y": data["intensity"].values[:, None].astype(np.float32),
                "meta_id": data["subject"].values[:, None],
                "meta_frame": data.index.values[:, None],
            }

            data = dict_to_batched_data(data, batch_size=-1)  # TODO
            while data:
                yield data.pop()

    def get_au(self, subject: str) -> Tuple[pd.DataFrame, List[Path]]:
        """
        Returns a Pandas DataFrame with the specified AU.
        """
        data = pd.read_csv(
            DISFA_FOLDER / f"ActionUnit_Labels/{subject}/{subject}_au{self.au}.txt",
            header=None,
            names=["frame", "intensity"],
            index_col="frame",
            dtype=int,
        )
        return (data, [Path("DISFA") / f"LeftVideo{subject}_comp.hdf"])

    def get_features(self, subject: str) -> pd.DataFrame:
        """
        Loads (and extracts) the openface features.
        """
        au_data, files = self.get_au(subject)
        openface = caching.read_hdfs(files[0])["df"]
        openface = openface.set_index("frame")
        columns = [
            x for x in openface.columns if x.startswith("hog") or x.startswith("p_")
        ]
        openface = openface.loc[:, columns]
        openface = openface - openface.median()
        openface = openface.loc[
            np.intersect1d(openface.index, au_data.index).tolist(), :
        ]
        openface["subject"] = int(subject[2:])
        return openface

    def get_list_of_subjects(self) -> List[str]:
        """
        Returns the list of subject ids.
        """
        files = DISFA_FOLDER.glob("Videos_LeftCamera/*.avi")
        return sorted(file.name[9:14] for file in files)

    def get_subjects_for_fold(self) -> List[str]:
        # load labels for all subjects
        subjects = self.get_list_of_subjects()
        data = [self.get_au(subject)[0] for subject in subjects]

        # generate split
        assert self.n_folds >= 4
        groups = np.array(
            [au_intensities["intensity"].mean() for au_intensities in data]
        )
        fold_index = stratified_splits(groups, np.ones(self.n_folds))

        # determine relevant subjects
        test = fold_index == self.ifold
        evaluation = fold_index == ((self.ifold + 1) % self.n_folds)
        training = ~test & ~evaluation
        fold = {"training": training, "evaluation": evaluation, "test": test}
        subject_index = fold[self.name]
        assert subject_index.sum() > 0, f"{self.name} {self.ifold} {self.au}"

        return [subject for subject, isin in zip(subjects, subject_index) if isin]


class MNIST(DISFA):
    def __init__(self, *args, imbalance=False, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.mnist = caching.read_pickle(Path("mnist"))
        for key in self.mnist:
            self.mnist[key] = list(self.mnist[key])
            self.mnist[key][0] = self.mnist[key][0].reshape(
                self.mnist[key][0].shape[0], -1
            )
            self.mnist[key][1] = self.mnist[key][1][:, None]

            # remove numbers above 5
            index = self.mnist[key][1][:, 0] <= 5
            self.mnist[key][0] = self.mnist[key][0][index, :].astype(float)
            self.mnist[key][1] = self.mnist[key][1][index, :].astype(float)

            # imbalance
            # 0: 55% --> 100%
            # 1: 8% --> 15%
            # 2: 14% --> 26%
            # 3: 18% --> 33%
            # 4: 5% --> 9%
            # 5: <1% --> 1%
            if imbalance:
                reference = (self.mnist[key][1] == 0).sum()
                for level, percentage in zip(
                    [1, 2, 3, 4, 5], [0.15, 0.26, 0.33, 0.09, 0.01]
                ):
                    index = np.where(self.mnist[key][1] == level)[0]
                    rng = np.random.RandomState(0)
                    keep = rng.choice(
                        index, size=int(reference * percentage), replace=False
                    )
                    remove = np.setdiff1d(index, keep)
                    self.mnist[key][0][remove] = float("NaN")
                    self.mnist[key][1][remove] = float("NaN")
                index = ~np.isnan(self.mnist[key][0]).any(axis=1)
                self.mnist[key][0] = self.mnist[key][0][index]
                self.mnist[key][1] = self.mnist[key][1][index]

        # training/test subjects
        self.n_training = int(np.floor(self.mnist["training"][1].size / 1006))
        self.n_test = int(np.floor(self.mnist["test"][1].size / 1006))

    def get_list_of_subjects(self) -> List[str]:
        # simulate that each person has 1006 data points
        return np.arange(self.n_training + self.n_test, dtype=int).astype(str).tolist()

    def get_features(self, subject: str) -> pd.DataFrame:
        # from training or testing
        data = self.mnist["training"][0]
        subject = int(subject)
        if subject >= self.n_training:
            data = self.mnist["test"][0]
            subject -= self.n_training

        # get block
        data = data[subject * 1006 : (subject + 1) * 1006, :]

        # wrap in pandas
        data = pd.DataFrame(data)
        data.columns = "hog_" + data.columns.astype(str)
        data["subject"] = subject
        data["frame"] = np.arange(subject * 1006, subject * 1006 + data.shape[0])
        data = data.set_index("frame")

        return data

    def get_subjects_for_fold(self) -> List[str]:
        if self.name == "test":
            return np.arange(self.n_training, self.n_training + self.n_test).astype(str)

        # determine split of training/development set
        subjects = np.arange(self.n_training, dtype=int).astype(str)
        groups = []
        for subject in subjects:
            groups.append(self.get_au(subject)[0]["intensity"].std())
        groups = np.asarray(groups)
        index = stratified_splits(groups, np.array([2, 1]))

        if self.name == "training":
            subjects = subjects[index == 0]
        else:
            subjects = subjects[index == 1]
        return subjects.tolist()

    def get_au(self, subject) -> Tuple[pd.DataFrame, List[Path]]:
        # from training or testing
        data = self.mnist["training"][1]
        subject = int(subject)
        if subject >= self.n_training:
            data = self.mnist["test"][1]
            subject -= self.n_training

        # get block
        data = data[subject * 1006 : (subject + 1) * 1006, :].astype(int)

        # wrap in pandas
        data = pd.DataFrame({"intensity": data.flatten()})
        data["frame"] = np.arange(subject * 1006, subject * 1006 + data.shape[0])
        data = data.set_index("frame")
        return data, []


class BP4D_PLUS(DISFA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.au == "6":
            self.au = "06"

        self.n_folds = 5
        self.dataset = Path("BP4D_plus")
        self.folder = BP4D_PLUS_FOLDER

    def get_au(self, subject: str) -> Tuple[pd.DataFrame, List[Path]]:
        videos, files = self.get_labeled_videos(subject)
        data = []
        for ifile, file in enumerate(files):
            data.append(
                pd.read_csv(file, header=None, names=["frame", "intensity"], dtype=int)
            )
            data[-1] = data[-1].set_index("frame")
            data[-1] = data[-1].loc[data[-1]["intensity"] != 9, :]
            data[-1].index += ifile * 50000
        data = pd.concat(data)

        return data, videos

    def get_labeled_videos(self, subject: str) -> Tuple[List[Path], List[Path]]:
        labels = self.folder.glob(f"AU{self.au}/{subject}*.csv")
        videos = list(self.dataset.glob(f"{subject}*.hdf"))
        # delete labels
        new_labels = []
        for label in labels:
            name = "_".join(label.name.split("_")[:2])
            if [1 for x in videos if name in x.name]:
                new_labels.append(label)
        # delete videos
        new_videos = []
        for video in videos:
            name = generic.basename(video)
            if [1 for x in new_labels if name in x.name]:
                new_videos.append(video)
        return sorted(new_videos), sorted(new_labels)

    def get_features(self, subject: str) -> pd.DataFrame:
        au_data, files = self.get_au(subject)
        data = []
        for ifile, file in enumerate(files):
            file = file.name.split(".")[0]
            openface = caching.read_hdfs(self.dataset / f"{file}.hdf")["df"]
            openface = openface.set_index("frame")
            columns = [
                x for x in openface.columns if x.startswith("hog") or x.startswith("p_")
            ]
            openface = openface.loc[:, columns]
            openface = openface.astype(np.float32)
            openface.index += ifile * 50000
            openface.loc[np.intersect1d(openface.index, au_data.index).tolist(), :]
            data.append(openface)
        data = pd.concat(data)
        data -= data.median()
        subject_id = int(subject[1:])
        if subject[0] == "M":
            subject_id *= -1
        data["subject"] = subject_id
        return data

    def get_list_of_subjects(self) -> List[str]:
        files = self.dataset.glob("*.hdf")
        return sorted({file.name.split("_", 1)[0] for file in files})
