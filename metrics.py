import numpy as np
import torch
import logging

import torch.nn.functional as F

from torch.distributions.normal import Normal


log = logging.getLogger(__name__)


def nll_map(model, sigma, dataloader):
    """Evaluate the NLL of the MAP model on the given dataset."""
    log.info(f"Evaluating NLL MAP with sigma={sigma}")
    nll = 0.0
    for X, y in dataloader:
        f_mu = model(X)
        nll += nll_univariate_normal(y, f_mu.squeeze(), sigma)
    return nll / len(dataloader.dataset)


def nll_bayesian(model, dataloader):
    """Evaluate the NLL of the Bayesian model on the given dataset."""
    log.info(f"Evaluating NLL Bayesian with sigma={model.sigma_noise}")
    nll = 0.0
    for X, y in dataloader:
        f_mu, f_var = model(x=X)
        pred_var = f_var + model.sigma_noise**2
        pred_std = pred_var.sqrt()
        nll += nll_univariate_normal(y, f_mu.squeeze(), pred_std.squeeze())
    return nll / len(dataloader.dataset)


def nll_univariate_normal(targets, loc, scale):
    dist = Normal(loc, scale)
    log_probs = -dist.log_prob(targets)
    return log_probs.sum()


def class_ll(y, log_probs=None, probs=None, eps=1e-40):
    assert log_probs is None or probs is None
    if log_probs is not None:
        pass
    elif probs is not None:
        log_probs = torch.log(probs.clamp(min=eps))
    else:
        raise Exception('Either log_probs or probs must be provided')
    nll = F.nll_loss(log_probs, y, reduction='mean')
    return -nll.item()


def class_error(target, log_prob):
    log.info("Target: {target.shape}")
    log.info("Log prob: {log_prob.shape}")
    pred = log_prob.max(dim=1, keepdim=False)
    err = pred.ne(target.data).sum().item() / target.shape[0]
    return err

