import torch
from torch.utils.data import Dataset


class RegressionDataset(Dataset):
    def __init__(self, X, y) -> None:
        self.X = torch.from_numpy(X).double()
        self.y = torch.from_numpy(y).double()
        self.len = self.X.shape[0]

    def __getitem__(self, index: int) -> tuple:
        return self.X[index], self.y[index]

    def __len__(self) -> int:
        return self.len
