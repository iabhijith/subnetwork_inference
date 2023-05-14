import torch
import numpy as np
import copy
import math
import logging

from torch import nn
from torch.nn.utils import parameters_to_vector
from torch.distributions.normal import Normal
from laplace import Laplace, marglik_training
from laplace.curvature import BackPackGGN

from configuration.config import TrainerConfig

log = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, config: TrainerConfig) -> None:
        self.config = config

    def train(self, model, delta, train_dataloader, val_dataloader):
        model.train()
        log_sigma = nn.Parameter(-torch.ones(1, requires_grad=True) / 2)
        params = list(model.parameters())
        params.append(log_sigma)
        optimizer = torch.optim.Adam(params, lr=self.config.lr)
        for i in range(self.config.epochs):
            epoch_loss = 0.0
            epoch_nll = 0.0
            model.train()
            for X, y in train_dataloader:
                optimizer.zero_grad()
                theta = parameters_to_vector(model.parameters())
                out = model(X).squeeze()
                loss = -Normal(out, log_sigma.exp()).log_prob(y.squeeze()).sum() + (
                    0.5 * (delta * theta) @ theta
                )
                loss.backward()
                optimizer.step()
                epoch_loss += loss.cpu().item() / len(train_dataloader)
                epoch_nll += (
                    -Normal(out.detach(), log_sigma.detach().exp()).log_prob(y).sum()
                )
            epoch_nll = epoch_nll / len(train_dataloader.dataset)

        val_nll = self.evaluate_map(model, log_sigma.detach(), val_dataloader)
        return model, log_sigma, val_nll

    def evaluate_map(self, model, log_sigma, dataloader):
        model.eval()
        val_nll = 0.0
        with torch.no_grad():
            for X, y in dataloader:
                out = model(X).squeeze()
                # print(X.shape, y.shape)
                # print(-Normal(out.detach(), log_sigma.detach().exp()).log_prob(y))
                val_nll += (
                    -Normal(out.detach(), log_sigma.detach().exp()).log_prob(y).sum()
                )
                print(val_nll)

            val_nll = val_nll / len(dataloader.dataset)

        return val_nll

    def train_map(self, model, dataloader, val_dataloader):
        delta_grid = np.logspace(-6, -1, 12)
        best_model = copy.deepcopy(model)
        best_nll = math.inf
        best_delta = delta_grid.min()
        for delta in delta_grid:
            model, log_sigma, val_nll = self.train(
                model, delta, dataloader, val_dataloader
            )
            log.info(f"MAP model with delta={delta} has a validation NLL={val_nll}")
            if val_nll < best_nll:
                best_model = copy.deepcopy(model)
                best_mse = val_nll
                best_delta = delta
        log.info(
            f"Best MAP model found with delta={best_delta} and validation Nll={val_nll}"
        )
        return best_model, log_sigma, best_delta

    def train_la_posthoc(
        self,
        model,
        dataloader,
        subset_of_weights,
        hessian_structure,
        sigma_noise,
        prior_precision,
        prior_mean,
        subnetwork_indices=None,
    ):
        model.train()
        if subnetwork_indices is None:
            la = Laplace(
                model=model,
                likelihood="regression",
                subset_of_weights=subset_of_weights,
                hessian_structure=hessian_structure,
                sigma_noise=sigma_noise,
                prior_precision=prior_precision,
                prior_mean=prior_mean,
            )
        else:
            la = Laplace(
                model=model,
                likelihood="regression",
                subset_of_weights=subset_of_weights,
                hessian_structure=hessian_structure,
                sigma_noise=sigma_noise,
                prior_precision=prior_precision,
                prior_mean=prior_mean,
                subnetwork_indices=subnetwork_indices,
            )

        la.fit(dataloader)
        return la

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

    def fit_sigma_noise(self, model, sigma, dataloader):
        sigma = torch.tensor(sigma, dtype=torch.float64, requires_grad=True)
        model.eval()
        optimiser = torch.optim.Adam([sigma], lr=self.config.map_tuning.lr)
        for epochs in range(self.config.map_tuning.epochs):
            for X, y in dataloader:
                optimiser.zero_grad()
                f_map = model(X)
                loss = -Normal(f_map, sigma).log_prob(y).mean()
                loss.backward()
                optimiser.step()

        return sigma

    def fit_sigma_noise_la(self, model, log_sigma):
        log_sigma = torch.tensor(log_sigma, dtype=torch.float64, requires_grad=True)
        optimiser = torch.optim.Adam([log_sigma], lr=self.config.la_tuning.lr)
        for epochs in range(self.config.la_tuning.epochs):
            optimiser.zero_grad()
            neg_marginal_likelihood = -model.log_marginal_likelihood(
                sigma_noise=log_sigma.exp()
            )
            neg_marginal_likelihood.backward()
            optimiser.step()
        return log_sigma.detach().cpu().numpy()
