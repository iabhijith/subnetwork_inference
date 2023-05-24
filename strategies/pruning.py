import torch
from laplace import Laplace
from torch.nn.utils import parameters_to_vector


class SubnetMask:
    """Baseclass for all subnetwork masks in this library (for subnetwork Laplace).

    Parameters
    ----------
    model : torch.nn.Module
    """

    def __init__(self, model):
        self.model = model
        self.parameter_vector = parameters_to_vector(self.model.parameters()).detach()
        self._n_params = len(self.parameter_vector)
        self._device = next(self.model.parameters()).device
        self._indices = None
        self._n_params_subnet = None

    def _check_select(self):
        if self._indices is None:
            raise AttributeError("Subnetwork mask not selected. Run select() first.")

    @property
    def indices(self):
        self._check_select()
        return self._indices

    @property
    def n_params_subnet(self):
        if self._n_params_subnet is None:
            self._check_select()
            self._n_params_subnet = len(self._indices)
        return self._n_params_subnet

    def convert_subnet_mask_to_indices(self, subnet_mask):
        """Converts a subnetwork mask into subnetwork indices.

        Parameters
        ----------
        subnet_mask : torch.Tensor
            a binary vector of size (n_params) where 1s locate the subnetwork parameters
            within the vectorized model parameters
            (i.e. `torch.nn.utils.parameters_to_vector(model.parameters())`)

        Returns
        -------
        subnet_mask_indices : torch.LongTensor
            a vector of indices of the vectorized model parameters
            (i.e. `torch.nn.utils.parameters_to_vector(model.parameters())`)
            that define the subnetwork
        """
        if not isinstance(subnet_mask, torch.Tensor):
            raise ValueError("Subnetwork mask needs to be torch.Tensor!")
        elif (
            subnet_mask.dtype
            not in [
                torch.int64,
                torch.int32,
                torch.int16,
                torch.int8,
                torch.uint8,
                torch.bool,
            ]
            or len(subnet_mask.shape) != 1
        ):
            raise ValueError(
                "Subnetwork mask needs to be 1-dimensional integral or boolean tensor!"
            )
        elif (
            len(subnet_mask) != self._n_params
            or len(subnet_mask[subnet_mask == 0]) + len(subnet_mask[subnet_mask == 1])
            != self._n_params
        ):
            raise ValueError(
                "Subnetwork mask needs to be a binary vector of"
                "size (n_params) where 1s locate the subnetwork"
                "parameters within the vectorized model parameters"
                "(i.e. `torch.nn.utils.parameters_to_vector(model.parameters())`)!"
            )

        subnet_mask_indices = subnet_mask.nonzero(as_tuple=True)[0]
        return subnet_mask_indices

    def select(self, train_loader=None):
        """Select the subnetwork mask.

        Parameters
        ----------
        train_loader : torch.data.utils.DataLoader, default=None
            each iterate is a training batch (X, y);
            `train_loader.dataset` needs to be set to access \\(N\\), size of the data set

        Returns
        -------
        subnet_mask_indices : torch.LongTensor
            a vector of indices of the vectorized model parameters
            (i.e. `torch.nn.utils.parameters_to_vector(model.parameters())`)
            that define the subnetwork
        """
        if self._indices is not None:
            raise ValueError("Subnetwork mask already selected.")

        subnet_mask = self.get_subnet_mask(train_loader)
        self._indices = self.convert_subnet_mask_to_indices(subnet_mask)
        return self._indices

    def get_subnet_mask(self, train_loader):
        """Get the subnetwork mask.

        Parameters
        ----------
        train_loader : torch.data.utils.DataLoader
            each iterate is a training batch (X, y);
            `train_loader.dataset` needs to be set to access \\(N\\), size of the data set

        Returns
        -------
        subnet_mask: torch.Tensor
            a binary vector of size (n_params) where 1s locate the subnetwork parameters
            within the vectorized model parameters
            (i.e. `torch.nn.utils.parameters_to_vector(model.parameters())`)
        """
        raise NotImplementedError


class ScoreBasedSubnetMask(SubnetMask):
    """Baseclass for subnetwork masks defined by selecting
    the top-scoring parameters according to some criterion.

    Parameters
    ----------
    model : torch.nn.Module
    n_params_subnet : int
        number of parameters in the subnetwork (i.e. number of top-scoring parameters to select)
    """

    def __init__(self, model, n_params_subnet):
        super().__init__(model)

        if n_params_subnet is None:
            raise ValueError(
                "Need to pass number of subnetwork parameters when using subnetwork Laplace."
            )
        if n_params_subnet > self._n_params:
            raise ValueError(
                f"Subnetwork ({n_params_subnet}) cannot be larger than model ({self._n_params})."
            )
        self._n_params_subnet = n_params_subnet
        self._param_scores = None

    def compute_param_scores(self, train_loader):
        raise NotImplementedError

    def _check_param_scores(self):
        if self._param_scores.shape != self.parameter_vector.shape:
            raise ValueError(
                "Parameter scores need to be of same shape as parameter vector."
            )

    def get_subnet_mask(self, train_loader):
        """Get the subnetwork mask by (descendingly) ranking parameters based on their scores."""

        if self._param_scores is None:
            self._param_scores = self.compute_param_scores(train_loader)
        self._check_param_scores()

        idx = torch.argsort(self._param_scores, descending=True)[
            : self._n_params_subnet
        ]
        idx = idx.sort()[0]
        subnet_mask = torch.zeros_like(self.parameter_vector).bool()
        subnet_mask[idx] = 1
        return subnet_mask


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
