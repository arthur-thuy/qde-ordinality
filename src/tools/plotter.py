"""File with plotting functionalities."""

# standard library imports
import os
from typing import Optional, Tuple

# related third party imports
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import structlog
import pandas as pd
from matplotlib.patches import Rectangle
from numpy.typing import ArrayLike

# local application/library specific imports
from tools.analyzer import get_config_id_print
from tools.utils import ensure_dir, read_pickle
from tools.analyzer import get_output_paths

# set up logger
logger = structlog.get_logger(__name__)


def plot_history(
    all_logs: list[dict[str, float]], metric: str, skip_warmup: int = 0
) -> None:
    """Plot metric and loss in function of number of epochs.

    Parameters
    ----------
    all_logs : list[dict[str, float]]
        List of dictionaries containing the training logs for each epoch.
    metric : str
        Metric to plot (in addition to loss).
    skip_warmup : int
        number of warmup epochs to not display
    """

    def _imitate_parsed_metric_name(metric: str) -> str:
        """Imitate parsed metric name from `transformers.modelcard.parse_log_history`.

        Parameters
        ----------
        metric : str
            Metric name

        Returns
        -------
        str
            Parsed metric name
        """
        splits = metric.split("_")
        name = " ".join([part.capitalize() for part in splits[1:]])
        return name

    epochs_arr = [log_epoch["Epoch"] for log_epoch in all_logs]
    train_loss_arr = [log_epoch["Training Loss"] for log_epoch in all_logs]
    train_loss_arr = [  # NOTE: replace "No log" with None for plotting
        None if train_loss == "No log" else train_loss for train_loss in train_loss_arr
    ]
    val_loss_arr = [log_epoch["Validation Loss"] for log_epoch in all_logs]

    parsed_metric = _imitate_parsed_metric_name(metric)
    metric_arr = [log_epoch[parsed_metric] for log_epoch in all_logs]

    plt.figure(figsize=(12, 5))

    # Metric
    plt.subplot(1, 2, 1)
    plt.plot(epochs_arr[skip_warmup:], metric_arr[skip_warmup:], color="tab:orange")
    # plt.ylim(0, 1)
    plt.title(parsed_metric)
    plt.ylabel(parsed_metric)
    plt.xlabel("Epochs")
    plt.legend(["valid"], loc="upper right")
    plt.grid(linestyle="dashed")

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_arr[skip_warmup:], train_loss_arr[skip_warmup:])
    plt.plot(epochs_arr[skip_warmup:], val_loss_arr[skip_warmup:])
    plt.title("Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend(["train", "valid"], loc="upper right")
    plt.grid(linestyle="dashed")

    plt.show()


def plot_violinplot(
    pred_label: tuple,
    label_map: dict,
    exp_name: str,
    config_id: str = None,
    config2legend: Optional[dict] = None,
    legend_exact: bool = False,
    save: bool = False,
    savefig_kwargs: Optional[dict] = None,
) -> None:
    """Plot violinplot.

    Parameters
    ----------
    pred_label : tuple
        Tuple of predictions and labels
    label_map : dict
        Mapping integer labels to string labels
    exp_name : str
        Experiment name
    config_id : str, optional
        Config ID, by default None
    config2legend : Optional[dict], optional
        Dictionary mapping config_id to legend name, by default None
    legend_exact : bool, optional
        Whether to find exact match in config2legend, by default False
    save : bool, optional
        Whether to save plot, by default False
    savefig_kwargs : Optional[dict], optional
        Kwargs to save the figure, by default None
    """
    # get bin_edges
    output_path = get_output_paths(exp_name, [config_id])[0]
    bin_edges = read_pickle(output_path)["bin_edges"]
    # pretty config id
    config_id_print = get_config_id_print(
        config_id=config_id, config2legend=config2legend, exact=legend_exact
    )

    # create df, ready for grouping
    predictions, labels = pred_label
    df = pd.DataFrame({"y_pred": predictions, "y_true_label": labels.astype(int)})
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    df["y_true_center"] = bin_centers[df["y_true_label"]]

    # plot
    _, ax = plt.subplots(figsize=(8, 8))
    sns.violinplot(
        x=df["y_true_center"],
        y=df["y_pred"],
        color="#c41331",
        alpha=0.25,
        native_scale=True,
    )
    # horizontal lines to divide range
    for bin_edge in bin_edges[1:-1]:
        ax.axhline(y=bin_edge, c="k", alpha=0.25)
    ax.set(
        xlabel="Difficulty",
        ylabel="Predicted difficulty",
        title=(None if save else config_id_print),
    )
    # xticks
    tick_labels = [
        f"{bin_centers[i]:.2f}\n({label_map[diff_level]})"
        for i, diff_level in enumerate(label_map.keys())
    ]
    ax.set_xticks(ticks=bin_centers, labels=tick_labels)
    # regression lines
    m, b = np.polyfit(labels, predictions, 1)
    if m and b:
        x0, x1 = bin_centers[0], bin_centers[-1]
        ax.plot([x0, x1], [x0 * m + b, x1 * m + b], c="#c41331", label="linear fit")
        ax.plot([x0, x1], [x0, x1], "--", c="darkred", label="ideal")
    ax.legend()
    if save:
        plt.tight_layout()
        ensure_dir(os.path.dirname(savefig_kwargs["fname"]))
        plt.savefig(**savefig_kwargs)
    plt.show()


def activate_latex(sans_serif: bool = False):
    """Activate latex for matplotlib."""
    if sans_serif:
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "Helvetica",
                "text.latex.preamble": r"\usepackage[cm]{sfmath}",
            }
        )
    else:
        plt.rcParams.update(
            {"text.usetex": True, "font.family": "Computer Modern Roman"}
        )


def deactivate_latex():
    """Deactivate latex for matplotlib."""
    plt.rcParams.update(
        {"text.usetex": False, "font.family": "DejaVu Sans", "text.latex.preamble": ""}
    )


def plot_confusion_matrix(
    conf_matrix: np.ndarray,
    int2label: dict,
    config_id: str = None,
    config2legend: Optional[dict] = None,
    legend_exact: bool = False,
    diagonal: bool = True,
    save: bool = False,
    savefig_kwargs: Optional[dict] = None,
) -> None:
    """Plot confusion matrix.

    Parameters
    ----------
    conf_matrix : np.ndarray
        Confusion matrix
    int2label : dict
        Mapping from integer labels to string labels
    config_id : str, optional
        Config ID, by default None
    config2legend : Optional[dict], optional
        Dictionary mapping config_id to legend name, by default None
    legend_exact : bool, optional
        Whether to find exact match in config2legend, by default False
    diagonal : bool, optional
        Whether to highlight diagonal cells, by default True
    save : bool, optional
        Whether to save plot, by default False
    savefig_kwargs : Optional[dict], optional
        Kwargs to save the figure, by default None
    """
    config_id_print = get_config_id_print(
        config_id=config_id, config2legend=config2legend, exact=legend_exact
    )

    # plot
    xticklabels = list(int2label.values())
    yticklabels = list(int2label.values())
    _, ax = plt.subplots(figsize=(len(xticklabels), len(yticklabels)))
    sns.heatmap(
        conf_matrix,
        annot=True,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        cmap="Blues",
        ax=ax,
        vmin=0.0,
        vmax=1.0,
        cbar=False,
        fmt=".2f",
    )
    # Add borders to diagonal cells
    if diagonal:
        for i in range(len(conf_matrix)):
            for i in range(len(conf_matrix)):
                ax.add_patch(
                    Rectangle((i, i), 1, 1, fill=False, edgecolor="black", lw=2)
                )
    ax.set(
        xlabel="Predicted label",
        ylabel="True label",
        title=(None if save else config_id_print),
    )
    ax.invert_yaxis()  # NOTE: labels read bottom-to-top
    ax.xaxis.label.set_size(15)
    ax.yaxis.label.set_size(15)
    ax.set_xticklabels(xticklabels, fontsize=13)
    ax.set_yticklabels(yticklabels, fontsize=13)
    if save:
        plt.tight_layout()
        ensure_dir(os.path.dirname(savefig_kwargs["fname"]))
        plt.savefig(**savefig_kwargs)
    plt.show()


def plot_rank_inconsistencies(
    arr: ArrayLike, num_classes: int, savename: Optional[str] = None
) -> None:
    """Plot relative frequency of rank inconsistencies per observation.

    Parameters
    ----------
    arr : ArrayLike
        Array of rank inconsistencies, of shape (N).
    num_classes : int
        Number of classes.
    savename : Optional[str], optional
        Name to save plot to, by default None
    """
    _, ax = plt.subplots(1, 1, figsize=(5, 3))
    # NOTE: if 7 classes, have 6 binary classifiers, so max 5 inconsistencies
    # NOTE: here `num_classes-1` because last element is exclusive
    ax.hist(
        arr,
        weights=np.zeros_like(arr) + 1.0 / arr.size,
        range=(0, num_classes - 1),
        bins=num_classes - 1,
        align="left",
        alpha=0.5,
    )
    ax.set_xlabel("Number of rank inconsistencies")
    ax.set_ylabel("Frequency")
    ax.grid(True, linestyle="--")
    ax.xaxis.get_major_locator().set_params(integer=True)
    # get ticks in sans-serif if sans-serif is used
    ax.xaxis.get_major_formatter()._usetex = False
    ax.yaxis.get_major_formatter()._usetex = False
    if savename is not None:
        plt.tight_layout()
        ensure_dir(os.path.dirname(savename))
        plt.savefig(savename)
    plt.show()


def plot_bin_edges(
    bin_edges: ArrayLike,
    start_end: Tuple[float, float],
    savename: Optional[str] = None,
) -> plt.Axes:
    """

    Parameters
    ----------
    bin_edges : ArrayLike
        Bin edges for discretization
    start_end : Tuple[float, float]
        Range of the x-axis
    savename : Optional[str], optional
        Name to save plot to, by default None

    Returns
    -------
    plt.Axes
        Axes object.
    """
    _, ax = plt.subplots(1, 1, figsize=(6, 1))
    ax.hlines(0, start_end[0], start_end[1], color="grey", linestyle="-", alpha=0.5)
    ax.vlines(bin_edges, -0.03, 0.03, color="blue", linestyle="-", alpha=0.5)
    # plot bin centers
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    ax.plot(bin_centers, np.zeros_like(bin_centers), "o", color="red")
    ax.set_xticks(np.arange(start_end[0], start_end[1] + 1, 1.0))
    ax.get_yaxis().set_visible(False)
    ax.grid(True, axis="x")
    if savename is not None:
        plt.tight_layout()
        ensure_dir(os.path.dirname(savename))
        plt.savefig(savename)
    plt.show()


def plot_cutpoints_history(params_history: dict[list]) -> plt.Axes:
    """Plot the cutpoints history for the ordered logit model.

    Parameters
    ----------
    params_history : dict[list]
        Parameter history

    Returns
    -------
    plt.Axes
        Axes object.
    """
    cutpoints_np = np.array(params_history["cutpoints"]).transpose().tolist()
    _, ax = plt.subplots(figsize=(8, 5))
    for i, cutpoint in enumerate(cutpoints_np):
        ax.plot(range(len(cutpoint)), cutpoint, label=f"cutpoint {i}")
    ax.set_xlabel("Evaluation step")
    ax.set_ylabel("Cutpoint value")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1])
    ax.grid(linestyle="--")
    plt.show()


def plot_bias_history(params_history: dict[list]) -> plt.Axes:
    """Plot the bias history for the ordered logit model.

    Parameters
    ----------
    params_history : dict[list]
        Parameter history

    Returns
    -------
    plt.Axes
        Axes object.
    """
    bias = [step["bias"][0] for step in params_history["betas"]]
    bias_np = np.array(bias).transpose().tolist()
    _, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        range(len(bias_np)),
        bias_np,
        label="bias",
    )
    ax.set_xlabel("Evaluation step")
    ax.set_ylabel("Bias value")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1])
    ax.grid(linestyle="--")
    plt.show()
