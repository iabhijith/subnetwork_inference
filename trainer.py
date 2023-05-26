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
        return self.log_sigma.exp().clamp(min=1e-4).item()


class ModelTrainer:
    def __init__(self, config: TrainerConfig) -> None:
        self.config = config

    def train(self, model, train_dataloader, val_dataloader):
        model.train()
        criteria = NegativeLogLikelihood()
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
            model.train()
            for X, y in train_dataloader:
                optimizer.zero_grad()
                theta = parameters_to_vector(model.parameters())
                out = model(X)
                loss = criteria(out, y).mean()  # + (0.5 * (10 * theta) @ theta)
                loss.backward()
                optimizer.step()
                epoch_err += F.mse_loss(out, y, reduction="mean").sqrt() * X.shape[0]
                epoch_nll += loss * X.shape[0]

            epoch_nll = epoch_nll / len(train_dataloader.dataset)
            epoch_err = epoch_err / len(train_dataloader.dataset)

            val_nll, val_err, count = self.evaluate(
                model, criteria.sigma(), val_dataloader
            )
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
        model.eval()
        criteria = NegativeLogLikelihood(sigma=sigma)
        err = 0.0
        nll = 0.0
        count = 0
        for X, y in dataloader:
            batch_size = X.shape[0]
            out = model(X)
            loss = criteria(out, y).mean()
            err += F.mse_loss(out, y, reduction="mean").sqrt() * batch_size
            nll += loss.item() * batch_size
            count += batch_size

        nll = nll / count
        err = err / count

        return nll, err, count

    def train_la_posthoc(
        self,
        model,
        dataloader,
        subset_of_weights,
        hessian_structure,
        sigma_noise,
        prior_mean,
        subnetwork_indices=None,
        val_dataloader=None,
    ):
        prior_precisions = [10]
        best_la_nll = np.inf
        best_prior_precision = prior_precisions[0]
        for prior_precision in prior_precisions:
            try:
                model_copy = copy.deepcopy(model)
                model_copy.train()
                if subnetwork_indices is None:
                    la = Laplace(
                        model=model,
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
                        model=model,
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
                print(e)
                continue
        return la, best_prior_precision

    def train_la_marglik(self, model, train_dataloader, hessian_structure):
        model.train()
        la, model, margliks, losses = marglik_training(
            model=model,
            train_loader=train_dataloader,
            likelihood="regression",
            hessian_structure=hessian_structure,
            backend=BackPackGGN,
            n_epochs=self.config.la.epochs,
            optimizer_kwargs={"lr": self.config.la.lr},
        )
        return la

    def evaluate_la(self, la, dataloader):
        ll = 0.0
        count = 0
        for X, y in dataloader:
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
