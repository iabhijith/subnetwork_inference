import torch
from torch.utils.data import Dataset


class RegressionDataset(Dataset):
    def __init__(self, X, y) -> None:
        """A dataset for regression tasks.
        Parameters
        ----------
        X : numpy.ndarray
            Input data.
        y : numpy.ndarray
            Target data.
        """
        super().__init__()
        self.X = torch.from_numpy(X).double()
        self.y = torch.from_numpy(y).double()
        self.len = self.X.shape[0]

    def __getitem__(self, index: int) -> tuple:
        """Get an item from the dataset."""
        return self.X[index], self.y[index]

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return self.len
