import torch
from laplace import Laplace
from torch.nn.utils import parameters_to_vector

from .base import ScoreBasedSubnetMask


class KronckerFactoredEigenSubnetMask(ScoreBasedSubnetMask):
    """Subnetwork mask identifying the parameters with the largest marginal variances
    (estimated using a diagonal Laplace approximation over all model parameters).

    Parameters
    ----------
    model : torch.nn.Module
    n_params_subnet : int
        number of parameters in the subnetwork (i.e. number of top-scoring parameters to select)
    diag_laplace_model : `laplace.baselaplace.DiagLaplace`
        diagonal Laplace model to use for variance estimation
    """

    def __init__(self, model, n_params_subnet, kron_laplace_model):
        super().__init__(model, n_params_subnet)
        self.kron_laplace_model = kron_laplace_model

    def compute_param_scores(self, train_loader):
        if train_loader is None:
            raise ValueError("Need to pass train loader for subnet selection.")

        self.kron_laplace_model.fit(train_loader)
        # Make it efficient by combining the eigenvalues of individual blocks
        K = self.kron_laplace_model.posterior_precision.to_matrix()
        K = torch.linalg.inv(K)
        return torch.linalg.eigvalsh(K)
