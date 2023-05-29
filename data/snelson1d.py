import numpy as np

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .base import RegressionDataset

INPUTS = "train_inputs"
OUTPUTS = "train_outputs"


class Snelson1D:
    def __init__(self, data_path):
        """Snelson1D dataset.
        Parameters
        ----------
        data_path : str
            Path to the data.
        """
        self.data_path = Path(data_path)
        inputs_path = self.data_path.joinpath(INPUTS)
        outputs_path = self.data_path.joinpath(OUTPUTS)

        X = np.loadtxt(inputs_path).reshape(-1, 1)
        y = np.loadtxt(outputs_path).reshape(-1, 1)

        X_scaler = MinMaxScaler(feature_range=(-2, 2)).fit(X)
        y_scaler = MinMaxScaler().fit(y)
        self.X = X_scaler.transform(X).reshape(-1)
        self.y = y_scaler.transform(y).reshape(-1)

    def data(self):
        return self.X, self.y

    def get_dataloaders(self, batch_size, val_size, random_state=9):
        """Get train, validation and test dataloaders for a snelson1d dataset.
        Parameters
        ----------
        batch_size : int
            Batch size.
        val_size : float
            Size of the validation set.
        Returns
        -------
        tuple: (torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader)
            A tuple of train, validation, and test dataloaders, where each dataloader is an instance of
            torch.utils.data.DataLoader.
        """
        train_dataset, val_dataset, test_dataset = self.get_datasets(val_size=val_size)

        return (
            DataLoader(train_dataset, shuffle=True, batch_size=batch_size),
            DataLoader(val_dataset, shuffle=True, batch_size=batch_size),
            DataLoader(test_dataset, shuffle=True, batch_size=batch_size),
        )

    def get_datasets(self, val_size=0.15, random_state=9):
        """Get train, validation and test dataloaders for a snelson1d dataset.
        Parameters
        ----------
        val_size : float
            Size of the validation set.
        random_state : int
            Random state.
        Returns
        -------
        tuple:  A tuple of train, validation, and test datasets, where each dataloader is an instance of
           RegressionDataset.
        """
        X_train, y_train, X_test, y_test = self.between_split()
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=val_size,
            random_state=random_state,
            shuffle=True,
        )
        return (
            RegressionDataset(X_train, y_train),
            RegressionDataset(X_val, y_val),
            RegressionDataset(X_test, y_test),
        )

    def between_split(self, ranges=[(-5, -1.5), (-0.5, 0.5), (1.5, 5)]):
        """Split the data into train and test sets based on the ranges provided.
        Parameters
        ----------
        ranges : list
            List of tuples of ranges for splitting the data.
        Returns
        -------
        tuple: (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
            A tuple of train inputs, train outputs, test inputs, and test outputs.
        """

        sorted_indices = np.argsort(self.X)
        X = self.X[sorted_indices]
        y = self.y[sorted_indices]
        conditions = []
        for l, r in ranges:
            conditions.append(np.all([X > l, X < r], axis=0))
        bl = np.any(conditions, axis=0)
        test_indices = np.nonzero(bl)
        train_indices = np.nonzero(~bl)
        return (
            X[train_indices].reshape(-1, 1),
            y[train_indices].reshape(-1, 1),
            X[test_indices].reshape(-1, 1),
            y[test_indices].reshape(-1, 1),
        )


if __name__ == "__main__":
    data = Snelson1D("SPGP")
    train, val, test = data.get_datasets()
    print(train.X.shape, train.y.shape)
    print(val.X.shape, val.y.shape)
    print(test.X.shape, test.y.shape)
