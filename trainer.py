import torch
import numpy as np
import copy
import math
import logging
import torch.nn.functional as F

from torch import nn
from torch.nn.utils import parameters_to_vector
from torch.distributions.normal import Normal
from laplace import Laplace, marglik_training
from laplace.curvature import BackPackGGN

from configuration.config import TrainerConfig

log = logging.getLogger(__name__)


class NegativeLogLikelihood(nn.Module):
    """Negative log likelihood loss for regression tasks.
    Parameters
    ----------
    output_dims : int
        Number of output dimensions.
    sigma : float
        Fixed initialization for the standard deviation. If None it is optimized.
    """

    def __init__(self, output_dims=1, sigma=None):
        super(NegativeLogLikelihood, self).__init__()
        if sigma is None:
            self.log_sigma = nn.Parameter(torch.zeros(output_dims))
        else:
            self.log_sigma = nn.Parameter(
                torch.ones(output_dims) * np.log(sigma), requires_grad=False
            )

    def forward(self, mu, y):
        sigma = self.log_sigma.exp().clamp(min=1e-4)
        dist = Normal(mu, sigma)
        return -dist.log_prob(y)

    def sigma(self):
        """Get the standard deviation.
        Returns
        -------
        sigma: float
            The standard deviation."""
        return self.log_sigma.exp().clamp(min=1e-4).item()


class ModelTrainer:
    def __init__(self, config: TrainerConfig, device: torch.device) -> None:
        """A trainer for models.
        Parameters
        ----------
        config : TrainerConfig
            The trainer configuration.
        device : torch.device
            The device to use for training.
        """
        self.config = config
        self.device = device

    def train(self, model, train_dataloader, val_dataloader):
        """Train a model. The model is trained using SGD with momentum and weight decay for the epochs specified in the configuration.
        After each epoch the validation loss is computed and the model with the lowest validation loss is returned. The training is early stopped
        if the validation loss does not improve for the number of epochs specified in the configuration.
        Parameters
        ----------
        model : torch.nn.Module
            The model to train.
        train_dataloader : torch.utils.data.DataLoader
            The training dataloader.
        val_dataloader : torch.utils.data.DataLoader
            The validation dataloader.
        Returns
        -------
        best_model: torch.nn.Module
            The trained model.
        best_sigma: float
            The standard deviation of the model.
        """
        criteria = NegativeLogLikelihood().to(self.device)
        params = list(model.parameters()) + list(criteria.parameters())
        optimizer = torch.optim.SGD(
            params, lr=self.config.lr, momentum=0.9, weight_decay=1e-4
        )
        best_val_nll = math.inf
        best_epoch = 0
        best_model = copy.deepcopy(model)
        best_sigma = 1.0
        for i in range(self.config.epochs):
            epoch_err = 0.0
            epoch_nll = 0.0
            count = 0
            model.train()
            for X, y in train_dataloader:
                X = X.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                out = model(X)
                loss = criteria(out, y).mean()
                loss.backward()
                optimizer.step()
                batch_size = X.shape[0]
                epoch_err += (
                    F.mse_loss(out, y, reduction="mean").sqrt().item() * batch_size
                )
                epoch_nll += loss * batch_size
                count += batch_size

            epoch_nll = epoch_nll / count
            epoch_err = epoch_err / count

            val_nll, val_err = self.evaluate(model, criteria.sigma(), val_dataloader)
            if val_nll < best_val_nll:
                best_val_nll = val_nll
                best_epoch = i
                best_model = copy.deepcopy(model)
                best_sigma = criteria.sigma()

            if i - best_epoch > self.config.patience:
                # log.info(f"Early stopping at epoch {i}")
                break

        return best_model, best_sigma

    def evaluate(self, model, sigma, dataloader):
        """Evaluate a model on a dataset.
        Parameters
        ----------
        model : torch.nn.Module
            The model to evaluate.
        sigma : float
            The standard deviation of the likelihood.
        dataloader : torch.utils.data.DataLoader
            The dataloader.
        Returns
        -------
        nll : float
            The negative log likelihood.
        err : float
            The root mean squared error.
        """
        model.eval()
        criteria = NegativeLogLikelihood(sigma=sigma).to(self.device)
        err = 0.0
        nll = 0.0
        count = 0
        for X, y in dataloader:
            X = X.to(self.device)
            y = y.to(self.device)
            batch_size = X.shape[0]
            out = model(X)
            loss = criteria(out, y).mean()
            err += F.mse_loss(out, y, reduction="mean").sqrt().item() * batch_size
            nll += loss.item() * batch_size
            count += batch_size

        nll = nll / count
        err = err / count

        return nll, err

    def train_la_posthoc(
        self,
        model,
        dataloader,
        subset_of_weights,
        hessian_structure,
        sigma_noise,
        prior_mean,
        val_dataloader,
        prior_precisions=None,
        subnetwork_indices=None,
    ):
        """Train a model using Laplace approximation.
        Parameters
        ----------
        model : torch.nn.Module
            The model to train.
        dataloader : torch.utils.data.DataLoader
            The training dataloader.
        subset_of_weights : str
            The subset of weights to train which can be either 'all' or 'subnetwork' or 'last_layer'.
        hessian_structure : list
            Approximate heissian structure. It can take values 'diagonal' or 'kron' or 'full'.
        sigma_noise : float
            The standard deviation of the noise.
        prior_mean : float
            The prior mean.
        val_dataloader : torch.utils.data.DataLoader, optional
            The validation dataloader.
        prior_precisions : list, optional
            The prior precisions for cross-validation. The default is None.
        subnetwork_indices : list, optional
            The indices of the subnetwork. Required only for 'subnetwork' subset_of_weights. The default is None.

        Returns
        -------
        best_model: torch.nn.Module
            The trained model.
        best_prior_precision: float
            The prior precision of the model.
        """
        if prior_precisions is None:
            prior_precisions = [10]  # Tuned the prior precision for the datasets.
        best_la_nll = np.inf
        best_prior_precision = prior_precisions[0]
        for prior_precision in prior_precisions:
            try:
                model_copy = copy.deepcopy(model)
                model_copy.train()
                if subnetwork_indices is None:
                    la = Laplace(
                        model=model_copy,
                        likelihood="regression",
                        subset_of_weights=subset_of_weights,
                        hessian_structure=hessian_structure,
                        sigma_noise=sigma_noise,
                        prior_precision=torch.tensor(
                            prior_precision, dtype=torch.float64
                        ),
                        prior_mean=prior_mean,
                    )
                else:
                    la = Laplace(
                        model=model_copy,
                        likelihood="regression",
                        subset_of_weights=subset_of_weights,
                        hessian_structure=hessian_structure,
                        sigma_noise=sigma_noise,
                        prior_precision=torch.tensor(
                            prior_precision, dtype=torch.float64
                        ),
                        prior_mean=prior_mean,
                        subnetwork_indices=subnetwork_indices,
                    )

                la.fit(dataloader)
                la_nll = self.evaluate_la(la, val_dataloader)
                if la_nll < best_la_nll:
                    best_la_nll = la_nll
                    best_prior_precision = prior_precision
            except Exception as e:
                continue
        return la, best_prior_precision

    def evaluate_la(self, la, dataloader):
        """Evaluate a Laplace approximation on a dataset.
        Parameters
        ----------
        la : Laplace
            The Laplace approximation.
        dataloader : torch.utils.data.DataLoader
            The dataloader.
        Returns
        -------
        nll : float
            The negative log likelihood.
        """
        ll = 0.0
        count = 0
        for X, y in dataloader:
            X = X.to(self.device)
            y = y.to(self.device)
            f_mu, f_var = la(x=X)
            f_sigma = torch.sqrt(f_var)
            pred_std = torch.sqrt(f_sigma**2 + la.sigma_noise**2)
            ll += self.log_likelihood(y, f_mu, pred_std)
            count += X.shape[0]
        return -ll / count

    def log_likelihood(self, y, mu, std):
        dist = Normal(mu.squeeze(), std.squeeze())
        log_probs = dist.log_prob(y.squeeze())
        return log_probs.squeeze().sum().item()
