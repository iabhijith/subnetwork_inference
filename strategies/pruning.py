import torch
from laplace import Laplace
from torch.nn.utils import parameters_to_vector

from .base import ScoreBasedSubnetMask


class OBDSubnetMask(ScoreBasedSubnetMask):
    """Subnetwork mask identifying the parameters with the largest saliencies as calculated using optimal brain damage (OBD).

    Parameters
    ----------
    model : torch.nn.Module
    n_params_subnet : int
        number of parameters in the subnetwork (i.e. number of top-scoring parameters to select)
    diag_laplace_model : `laplace.baselaplace.DiagLaplace`
        diagonal Laplace model to use for diagonal hessian estimation
    """

    def __init__(self, model, n_params_subnet, diag_laplace_model):
        super().__init__(model, n_params_subnet)
        self.diag_laplace_model = diag_laplace_model

    def compute_param_scores(self, train_loader):
        if train_loader is None:
            raise ValueError("Need to pass train loader for subnet selection.")
        # Train the diagonal Laplace model
        self.diag_laplace_model.fit(train_loader)
        # Compute the parameter saliencies using (Optimal Brain Damage)
        saliencies = 0.5 * self.parameter_vector.square() * self.diag_laplace_model.H
        return saliencies

    
class SPRSubnetMask(ScoreBasedSubnetMask):
    """Subnetwork mask identifying the parameters with SPR.

    Parameters
    ----------
    model : torch.nn.Module
    n_params_subnet : int
        number of parameters in the subnetwork (i.e. number of top-scoring parameters to select)
    diag_laplace_model : `laplace.baselaplace.DiagLaplace`
        diagonal Laplace model to use for diagonal hessian estimation
    """

    def __init__(self, model, n_params_subnet, diag_laplace_model):
        super().__init__(model, n_params_subnet)
        self.diag_laplace_model = diag_laplace_model

    def compute_param_scores(self, train_loader):
        if train_loader is None:
            raise ValueError("Need to pass train loader for subnet selection.")
        # Train the diagonal Laplace model
        self.diag_laplace_model.fit(train_loader)
        # Compute the parameter saliencies using (SPR)
        V = 1/self.diag_laplace_model.H
        saliencies = self.parameter_vector.abs() + torch.sqrt(V)
        return saliencies

class MNSubnetMask(ScoreBasedSubnetMask):
    """Subnetwork mask identifying the parameters with the largest saliencies as calculated using MN.

    Parameters
    ----------
    model : torch.nn.Module
    n_params_subnet : int
        number of parameters in the subnetwork (i.e. number of top-scoring parameters to select)
    diag_laplace_model : `laplace.baselaplace.DiagLaplace`
        diagonal Laplace model to use for diagonal hessian estimation
    """

    def __init__(self, model, n_params_subnet, diag_laplace_model):
        super().__init__(model, n_params_subnet)
        self.diag_laplace_model = diag_laplace_model

    def compute_param_scores(self, train_loader):
        if train_loader is None:
            raise ValueError("Need to pass train loader for subnet selection.")
        # Train the diagonal Laplace model
        self.diag_laplace_model.fit(train_loader)
        # Compute the parameter saliencies using MN
        V = 1/self.diag_laplace_model.H
        saliencies = self.parameter_vector.square() + V
        return saliencies