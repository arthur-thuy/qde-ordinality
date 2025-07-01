"""Metrics."""

# standard library imports
from typing import Any, Literal, Union

# related third party imports
import numpy as np
import torch
from coral_pytorch.dataset import corn_label_from_logits, proba_to_label
from lifelines.utils import concordance_index
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    root_mean_squared_error,
)
from sklearn.utils.class_weight import compute_sample_weight
from torchmetrics import Accuracy as torchAccuracy

# local application/library specific imports
from tools.constants import (
    ARC,
    ARC_BALANCED,
    RACE_PP,
    RACE_PP_4K,
    RACE_PP_8K,
    RACE_PP_12K,
    AM,
    CUPA,
    OrdinalType,
    OutputType,
)
from discretizer.discretizer import discretize_into_bins


def postprocess_preds(
    eval_pred: tuple[ArrayLike, ArrayLike],
    output_type: OutputType,
    num_classes: int,
    bin_edges: ArrayLike,
) -> tuple[ArrayLike, ArrayLike]:
    """Postprocess outputs, depending on the output type.

    Parameters
    ----------
    eval_pred : ArrayLike
        Predictions and labels
    output : OutputType
        Type of output
    num_classes : int
        Number of classes
    bin_edges : ArrayLike
        Bin edges for discretization

    Returns
    -------
    tuple[ArrayLike, ArrayLike]
    """
    preds, true_labels = eval_pred

    # convert to predicted label
    if output_type == OutputType.REGR:
        # NOTE: exclude outer edges of bins for inference
        bin_edges = bin_edges[1:-1]
        pred_labels = discretize_into_bins(
            arr=preds, bin_edges=bin_edges, include_right=False
        ).squeeze()
    elif output_type == OutputType.CLASS:
        pred_labels = np.argmax(preds, axis=-1)
    elif output_type == OrdinalType.CORAL:
        probas = torch.sigmoid(torch.from_numpy(preds))
        pred_labels = proba_to_label(probas).float().numpy()
    elif output_type == OrdinalType.CORN:
        pred_labels = corn_label_from_logits(torch.from_numpy(preds)).float().numpy()
    elif output_type == OrdinalType.OR_NN:
        probas = torch.sigmoid(torch.from_numpy(preds))
        pred_labels = proba_to_label(probas).float().numpy()
    elif output_type == OrdinalType.LOGIT:
        probas = torch.from_numpy(preds)
        pred_labels = torch.argmax(probas, dim=-1).float().numpy()
    else:
        raise NotImplementedError

    # convert to predicted probabilities
    # NOTE: pred_probs is (N, num_classes)
    preds = torch.from_numpy(preds)
    if output_type == OutputType.REGR:
        pred_probs = (
            torch.nn.functional.one_hot(
                torch.from_numpy(pred_labels), num_classes=num_classes
            )
            .float()
            .numpy()
        )
    elif output_type == OutputType.CLASS:
        pred_probs = torch.softmax(preds, dim=-1).float().numpy()
    elif output_type == OrdinalType.CORAL:
        pred_probs = compute_marginal_probs(torch.sigmoid(preds)).float().numpy()
    elif output_type == OrdinalType.CORN:
        pred_probs = torch.sigmoid(preds)
        pred_probs = torch.cumprod(pred_probs, dim=1)
        pred_probs = compute_marginal_probs(pred_probs).float().numpy()
    elif output_type == OrdinalType.OR_NN:
        pred_probs = compute_marginal_probs(torch.sigmoid(preds)).float().numpy()
    elif output_type == OrdinalType.LOGIT:
        pred_probs = preds.float().numpy()
    else:
        raise NotImplementedError

    assert (
        pred_probs.shape[1] == num_classes
    ), f"Predicted probs must be of shape (N, num_classes), is shape {pred_probs.shape}"

    return pred_labels, pred_probs, true_labels


def compute_metrics(
    eval_pred: ArrayLike,
    output_type: OutputType,
    num_classes: int,
    bin_edges: ArrayLike,
) -> dict[str, Any]:
    """Compute metrics based on the output type.

    Parameters
    ----------
    eval_pred : ArrayLike
        Predictions and labels
    output : OutputType
        Type of output
    num_classes : int
        Number of classes
    bin_edges : ArrayLike
        Bin edges for discretization

    Returns
    -------
    dict[str, Any]
        Metrics
    """
    # postprocess predictions
    pred_labels, pred_probs, true_labels = postprocess_preds(
        eval_pred=eval_pred,
        output_type=output_type,
        num_classes=num_classes,
        bin_edges=bin_edges,
    )

    # metrics based on probs
    metrics = {}

    # compute metrics
    metrics = {
        "rps": rps(
            torch.tensor(true_labels, dtype=torch.int64),
            torch.from_numpy(pred_probs),
        ),
        "bal_rps": rps(
            torch.tensor(true_labels, dtype=torch.int64),
            torch.from_numpy(pred_probs),
            balanced=True,
        ),
        "rps_discrete": rps(
            torch.tensor(true_labels, dtype=torch.int64),
            torch.nn.functional.one_hot(
                torch.tensor(pred_labels, dtype=torch.int64), num_classes=num_classes
            ).to(
                torch.float32
            ),  # NOTE: need float for upcoming MSE calculation
        ),
        "bal_rps_discrete": rps(
            torch.tensor(true_labels, dtype=torch.int64),
            torch.nn.functional.one_hot(
                torch.tensor(pred_labels, dtype=torch.int64), num_classes=num_classes
            ).to(
                torch.float32
            ),  # NOTE: need float for upcoming MSE calculation
            balanced=True,
        ),
        "c_idx": concordance_index(
            torch.from_numpy(true_labels),
            torch.from_numpy(pred_labels),
        ),
        "rmse": root_mean_squared_error(true_labels, pred_labels),
        "bal_rmse": balanced_rmse(true_labels, pred_labels),
        "accuracy": accuracy_score(true_labels, pred_labels),
        "bal_accuracy": balanced_accuracy_score(true_labels, pred_labels),
    }
    if output_type != OutputType.REGR:
        acc_class = torchAccuracy(
            task="multiclass", num_classes=num_classes, average=None
        )
        acc_class_result = acc_class(
            torch.from_numpy(pred_labels), torch.from_numpy(true_labels)
        ).tolist()
        metrics["acc_class"] = acc_class_result
    return metrics


def get_labels_binedges(dataset_name: str, num_classes: int) -> list[str]:
    """Get difficulty labels and bin edges from dataset name.

    Parameters
    ----------
    dataset_name : str
        Dataset name
    num_classes : int
        Number of classes

    Returns
    -------
    list[str]
        Difficulty labels

    Raises
    ------
    NotImplementedError
        If dataset is not recognized
    """
    if dataset_name in {RACE_PP, RACE_PP_4K, RACE_PP_8K, RACE_PP_12K}:
        return (["middle", "high", "university"], np.array([-0.5, 0.5, 1.5, 2.5]))
    elif dataset_name in {ARC, ARC_BALANCED}:
        return (
            [
                "level_3",
                "level_4",
                "level_5",
                "level_6",
                "level_7",
                "level_8",
                "level_9",
            ],
            np.array([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]),
        )
    elif dataset_name in [AM, CUPA]:
        return ([f"bin_{i}" for i in range(num_classes)], None)
    else:
        raise NotImplementedError


def balanced_rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Balanced RMSE (macroaveraged over classes)

    Parameters
    ----------
    y_true : ArrayLike
        True labels
    y_pred : ArrayLike
        Predicted labels

    Returns
    -------
    float
        Balanced RMSE

    Raises
    ------
    ValueError
        If y_true or y_pred contain non-integer values
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # check if all values in y_true and y_pred are integers
    if not np.all(y_true == np.round(y_true)):
        raise ValueError("y_true must contain only integers")
    if not np.all(y_pred == np.round(y_pred)):
        raise ValueError("y_pred must contain only integers")

    # group the indices of y_true by their integer values
    y_grouped_idx_list = []
    for i in np.unique(y_true):
        y_grouped_idx_list.append(np.where(y_true == i)[0])
    y_true_grouped_list = [y_true[idx] for idx in y_grouped_idx_list]
    y_pred_grouped_list = [y_pred[idx] for idx in y_grouped_idx_list]
    # calculate rmse for each group, and average them
    rmse_list = [
        root_mean_squared_error(y_true_grouped, y_pred_grouped)
        for y_true_grouped, y_pred_grouped in zip(
            y_true_grouped_list, y_pred_grouped_list
        )
    ]
    balanced_rmse = np.mean(rmse_list)
    return balanced_rmse


def compute_marginal_probs(preds: torch.Tensor) -> torch.Tensor:
    """Compute marginal probabilities P(y_i = r_k) from probabilities P(y_i > r_k).

    Parameters
    ----------
    preds : torch.Tensor
        Batch of probabilities P(y_i > r_k), shape (N,C)

    Returns
    -------
    torch.Tensor
        Batch of probabilities P(y_i = r_k), shape (N,C)
    """
    # check if probs are valid
    assert torch.all(
        (0 <= preds) & (preds <= 1)
    ), "Probabilities must be in the range [0,1]."
    # # check if monotonically non-increasing  # NOTE: allow this for OR-NN
    # assert torch.all(
    #     torch.diff(preds) <= 0
    # ), "Probabilities must be monotonically non-increasing."

    # P(y_i > -inf) = 1
    ones = torch.ones((preds.shape[0], 1))
    preds = torch.cat((ones, preds), dim=-1)
    # P(y_i > inf) = 0
    zeroes = torch.zeros((preds.shape[0], 1))
    preds = torch.cat((preds, zeroes), dim=-1)
    preds = -torch.diff(preds)
    torch.testing.assert_close(
        torch.sum(preds, dim=-1),
        torch.full((preds.shape[0],), 1.0),
        msg="Probabilities must sum to 1.",
    )
    return preds


def rps(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    balanced: bool = False,
    multidim_average: Literal["global", "samplewise"] = "global",
) -> Union[float, NDArray]:
    """Ranked Probability Score (RPS) for ordinal regression.

    Parameters
    ----------
    y_true : ArrayLike
        True labels of shape () or (N)
    y_pred : ArrayLike
        Predicted probabilities of shape (C) or (N,C).
    balanced : bool, default=False
        Whether to use balanced RPS.
        If True, each sample is weighted according to the inverse prevalence of its true class.
    multidim_average : str, default="global"
        Whether to return one RPS score for all samples or one RPS score per sample.
        "global" returns one RPS score for all samples.
        "samplewise" returns one RPS score per sample.

    Returns
    -------
    Union[float, np.NDArray]
        RPS score.
    """
    y_true_ohe = torch.nn.functional.one_hot(y_true, num_classes=y_pred.shape[-1])
    assert y_true_ohe.shape == y_pred.shape, "Shapes of y_true and y_pred must match."

    # do not use last probability because always 1
    cum_y_true = torch.cumsum(y_true_ohe, axis=-1)[:, :-1].float()
    cum_y_pred = torch.cumsum(y_pred, axis=-1)[:, :-1].float()
    raw_score = torch.nn.MSELoss(reduction="none")(cum_y_pred, cum_y_true)
    # compute one RPS per sample
    score_sample = torch.sum(raw_score, axis=-1)

    # compute balanced RPS
    if balanced:
        sample_weight = torch.as_tensor(
            compute_sample_weight(class_weight="balanced", y=y_true),
            dtype=score_sample.dtype,
        )
        score_sample = score_sample * sample_weight
    # get output
    if multidim_average == "global":
        score = torch.mean(score_sample).item()
        return score
    elif multidim_average == "samplewise":
        return score_sample.numpy()
    else:
        raise ValueError(
            f"multidim_average must be 'global' or 'samplewise', but got {multidim_average}."
        )
