"""Module for the ordered logit layer."""

# standard library imports
from typing import Literal, Optional

# related third party imports
import numpy as np
import torch
from torch import nn


class OrderedLogitLayer(torch.nn.Module):
    def __init__(
        self,
        size_in: int,
        num_classes: int,
        init_cutpoints: Literal["equal_spaced", "random_spaced"] = "equal_spaced",
    ) -> None:
        """Initialize the OrderedLogitLayer.

        Parameters
        ----------
        size_in : int
            Number of input features for the inputs to the forward method, which
            are expected to have shape=(num_examples, num_features).
        num_classes : int
            Number of classes in the dataset.
        init_cutpoints : Literal["equal_spaced", "random_spaced"], optional
            How to initialze cutpoints, equal spaced if "equal_spaced" and randomly
            spaced if "random_spaced". By default "equal_spaced".

        Raises
        ------
        ValueError
            If `init_cutpoints` is not a valid option.
        """
        super().__init__()
        self.size_in = size_in
        self.num_classes = num_classes
        self.linear = torch.nn.Linear(self.size_in, 1)

        # Store parameters but don't set them yet
        self.num_deltas = self.num_classes - 2
        self.threshold_step = 1.0  # TODO: decide on init value
        # self.threshold_step = 1.0/num_deltas  # TODO: decide on init value
        # self.threshold_step = np.sqrt(1/768)/num_deltas  # TODO: decide on init value
        self.init_cutpoints = init_cutpoints
        self.bias_reset = 0.0

        # Initialize deltas and bias
        self.reset_parameters()

    def _initialize_deltas(self):
        """Initialize deltas based on the specified method."""
        if self.init_cutpoints == "equal_spaced":
            deltas = torch.log(torch.full((self.num_deltas,), self.threshold_step))
            self.deltas = torch.nn.Parameter(deltas)
            # calculate bias
            self.bias_reset = self.threshold_step * self.num_deltas / 2
        elif self.init_cutpoints == "equal_density":
            # Calculate thresholds that divide the logistic distribution into equal parts
            probs = torch.linspace(
                1.0 / self.num_classes,
                (self.num_classes - 1.0) / self.num_classes,
                self.num_classes - 1,
            )

            # Inverse of logistic CDF: F^(-1)(p) = ln(p/(1-p))
            thresholds = torch.log(probs / (1 - probs))

            # Calculate deltas (differences between adjacent thresholds)
            deltas = thresholds[1:] - thresholds[:-1]

            # Ensure numerical stability with log parametrization
            self.deltas = torch.nn.Parameter(torch.log(deltas))

            # calculate bias
            self.bias_reset = (torch.sum(deltas) / 2).item()

            # For debugging
            print(f"Thresholds: {thresholds}")  # TODO: remove
        elif self.init_cutpoints == "random_spaced":
            deltas = torch.log(torch.rand(self.num_deltas))
            self.deltas = torch.nn.Parameter(deltas)
        else:
            raise ValueError(
                f"{self.init_cutpoints} is not a valid init_cutpoints type"
            )

    def reset_parameters(self):
        """Custom method to reset parameters after model initialization."""
        # Initialize deltas
        self._initialize_deltas()

        # Initialize linear layer parameters
        if self.init_cutpoints == "equal_spaced":
            self.linear.bias.data.fill_(self.bias_reset)
        elif self.init_cutpoints == "equal_density":
            self.linear.bias.data.fill_(self.bias_reset)
        elif self.init_cutpoints == "random_spaced":
            pass
        print(f"After reset: {self.linear.bias = }")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor, shape=(num_examples, num_features)
            Input features.

        Returns
        -------
        torch.Tensor
            Probabilities for each class, shape=(num_examples, num_classes)
        """
        betax = self.linear(x)
        # calculate cutpoints
        cutpoints = torch.cumsum(
            torch.cat(
                (torch.tensor([0.0], device=x.device), torch.exp(self.deltas)), dim=0
            ),
            dim=0,
        )
        # calculate probabilities
        cdf = torch.sigmoid(cutpoints - betax)
        cdf_diff = cdf[:, 1:] - cdf[:, :-1]
        probs = torch.cat((cdf[:, [0]], cdf_diff, (1 - cdf[:, [-1]])), dim=1)
        # assert if all rows sum to 1 (probabilities)
        assert torch.all(
            torch.isclose(torch.sum(probs, dim=1), torch.tensor([1.0], device=x.device))
        )
        return probs


def _reduction(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    """
    Reduce loss

    Parameters
    ----------
    loss : torch.Tensor, [batch_size, num_classes]
        Batch losses.
    reduction : str
        Method for reducing the loss. Options include 'elementwise_mean',
        'none', and 'sum'.

    Returns
    -------
    loss : torch.Tensor
        Reduced loss.

    """
    if reduction == "elementwise_mean":
        return loss.mean()
    elif reduction == "none":
        return loss
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError(f"{reduction} is not a valid reduction")


def cumulative_link_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    reduction: str = "elementwise_mean",
    class_weights: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """
    Calculates the negative log likelihood using the logistic cumulative link
    function.

    See "On the consistency of ordinal regression methods", Pedregosa et. al.
    for more details. While this paper is not the first to introduce this, it
    is the only one that I could find that was easily readable outside of
    paywalls.

    Parameters
    ----------
    y_pred : torch.Tensor, [batch_size, num_classes]
        Predicted target class probabilities. float dtype.
    y_true : torch.Tensor, [batch_size, 1]
        True target classes. long dtype.
    reduction : str
        Method for reducing the loss. Options include 'elementwise_mean',
        'none', and 'sum'.
    class_weights : np.ndarray, [num_classes] optional (default=None)
        An array of weights for each class. If included, then for each sample,
        look up the true class and multiply that sample's loss by the weight in
        this array.

    Returns
    -------
    loss: torch.Tensor
    """
    eps = 1e-15
    y_true_unsqueeze = torch.unsqueeze(
        y_true, dim=1
    )  # NOTE: need [batch_size, 1] shape
    likelihoods = torch.clamp(torch.gather(y_pred, 1, y_true_unsqueeze), eps, 1 - eps)
    neg_log_likelihood = -torch.log(likelihoods)

    if class_weights is not None:
        # Make sure it's on the same device as neg_log_likelihood
        class_weights = torch.as_tensor(
            class_weights,
            dtype=neg_log_likelihood.dtype,
            device=neg_log_likelihood.device,
        )
        neg_log_likelihood *= class_weights[y_true_unsqueeze]

    loss = _reduction(neg_log_likelihood, reduction)
    return loss


class CumulativeLinkLoss(nn.Module):
    """
    Module form of cumulative_link_loss() loss function

    Source: https://github.com/EthanRosenthal/spacecutter/blob/master/spacecutter/models.py

    Parameters
    ----------
    reduction : str
        Method for reducing the loss. Options include 'elementwise_mean',
        'none', and 'sum'.
    class_weights : np.ndarray, [num_classes] optional (default=None)
        An array of weights for each class. If included, then for each sample,
        look up the true class and multiply that sample's loss by the weight in
        this array.

    """

    def __init__(
        self,
        reduction: str = "elementwise_mean",
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.class_weights = class_weights
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return cumulative_link_loss(
            y_pred, y_true, reduction=self.reduction, class_weights=self.class_weights
        )
