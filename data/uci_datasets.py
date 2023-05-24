import numpy as np
import zipfile
import yaml

from pathlib import Path
from urllib.request import urlretrieve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .base import RegressionDataset

UCI_DATA = "UCI_for_sharing"
UCI_DATA_ZIP = "UCI_for_sharing.zip"
UCI_DATA_URL = "https://javierantoran.github.io/assets/datasets/UCI_for_sharing.zip"

STANDARD = "standard"
GAP = "gap"
DATA = "data"

DATA_FILE = "data.txt"
FEATURE_INDICES_FILE = "index_features.txt"
TARGET_INDICES_FILE = "index_target.txt"

UCI_METADATA = "uci_meta.yaml"


class UCIData:
    def __init__(self, data_path):
        """
        Initialize UCIData. Download the data if it does not exist.

        Parameters
        ----------
        data_path : str
            Path to the data directory.

        Returns
        -------
        None
        """
        self.data_path = Path(data_path)
        self.uci_data_path = self.data_path.joinpath(UCI_DATA)

        uci_data_zip = self.data_path.joinpath(UCI_DATA_ZIP)
        if not self.uci_data_path.exists():
            urlretrieve(
                UCI_DATA_URL,
                filename=uci_data_zip,
            )
        with zipfile.ZipFile(uci_data_zip, "r") as zip_ref:
            zip_ref.extractall(self.data_path)

    def get_metadata(self):
        uci_meta = yaml.safe_load(self.data_path.joinpath(UCI_METADATA).read_text())
        return uci_meta

    def get_dataloaders(self, dataset, batch_size, seed, val_size, split_index, gap):
        """Get train, validation and test dataloaders for a given dataset.
        Parameters
        ----------
        dataset : str
            Name of the dataset.
        batch_size : int
            Batch size.
        seed : int
            Seed for the random number generator.
        val_size : float
            Size of the validation set.
        split_index : int
            Index of the split.
        gap : bool
            Whether to use the gap version of the dataset.

        Returns
        -------
        tuple: (torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader)
            A tuple of train, validation, and test dataloaders, where each dataloader is an instance of
            torch.utils.data.DataLoader.
        """
        train_dataset, val_dataset, test_dataset = self.get_datasets(
            dataset=dataset,
            seed=seed,
            val_size=val_size,
            split_index=split_index,
            gap=gap,
        )

        return (
            DataLoader(train_dataset, shuffle=True, batch_size=batch_size),
            DataLoader(val_dataset, shuffle=True, batch_size=batch_size),
            DataLoader(test_dataset, shuffle=True, batch_size=batch_size),
        )

    def get_datasets(self, dataset, seed, val_size, split_index, gap):
        """
        Get train, validation and test datasets for a given dataset.

        Parameters
        ----------
        dataset : str
            Name of the dataset.
        seed : int
            Seed for the random number generator.
        val_size : float
            Size of the validation set.
        split_index : int
            Index of the split.
        gap : bool
            Whether to use the gap version of the dataset.

        Returns
        -------
        tuple: (RegressionDataset, RegressionDataset, RegressionDataset)
            A tuple of train, validation, and test datasets, where each dataset is an instance of RegressionDataset.

        """
        data_path = (
            self.uci_data_path.joinpath(STANDARD).joinpath(dataset).joinpath("data")
        )
        if not gap:
            data_index_path = (
                self.uci_data_path.joinpath(STANDARD).joinpath(dataset).joinpath(DATA)
            )
        else:
            data_index_path = (
                self.uci_data_path.joinpath(GAP).joinpath(dataset).joinpath(DATA)
            )

        data = np.loadtxt(data_path.joinpath(DATA_FILE))
        feature_indices = np.loadtxt(
            fname=data_path.joinpath(FEATURE_INDICES_FILE),
            dtype=int,
        )
        target_indices = np.loadtxt(
            fname=data_path.joinpath(TARGET_INDICES_FILE),
            dtype=int,
        )
        train_indices = np.loadtxt(
            data_index_path.joinpath(f"index_train_{split_index}.txt"), dtype=int
        )
        test_indices = np.loadtxt(
            data_index_path.joinpath(f"index_test_{split_index}.txt"), dtype=int
        )

        train_data = data[train_indices, :]
        test_data = data[test_indices, :]

        X_train = train_data[:, feature_indices]
        y_train = train_data[:, target_indices]

        X_test = test_data[:, feature_indices]
        y_test = test_data[:, target_indices]

        X_scaler = StandardScaler().fit(X_train)
        y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))

        X_train = X_scaler.transform(X_train)
        y_train = y_scaler.transform(y_train.reshape(-1, 1))

        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=val_size,
            random_state=seed,
        )

        X_test = X_scaler.transform(X_test)
        y_test = y_scaler.transform(y_test.reshape(-1, 1))

        return (
            RegressionDataset(X_train, y_train),
            RegressionDataset(X_val, y_val),
            RegressionDataset(X_test, y_test),
        )


if __name__ == "__main__":
    data = UCIData(".")
    train_dataloader, val_dataloader, test_dataloader = data.get_dataloaders(
        dataset="wine", batch_size=32, seed=42, val_size=0.15, split_index=0, gap=True
    )
    print(train_dataloader.dataset.X.shape)
    print(val_dataloader.dataset.X.shape)
    print(test_dataloader.dataset.X.shape)
