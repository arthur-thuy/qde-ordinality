"""File with analyzer functionalities."""

# standard library imports
import glob
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Tuple

# related third party imports
import numpy as np
import scipy
import structlog
import torch
from datasets import ClassLabel, Dataset
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix, root_mean_squared_error
from tabulate import tabulate
from transformers.modelcard import parse_log_history

# local application/library specific imports
from tools.constants import OutputType
from tools.metrics import (
    get_labels_binedges,
    postprocess_preds,
    rps,
)
from tools.utils import write_pickle, ensure_dir, read_pickle

# set up logger
logger = structlog.get_logger(__name__)


def get_output_paths(experiment: str, config_ids: list[str]) -> list[str]:
    """Get paths to output files.

    Parameters
    ----------
    experiment : str
        Experiment name
    config_ids : list[str]
        List of config IDs
    Returns
    -------
    list[str]
        List of output paths
    """
    # paths
    output_dir = os.path.join("output", experiment)
    output_paths = [
        os.path.join(output_dir, f"{config_id}.pickle") for config_id in config_ids
    ]

    return output_paths


def mean_stderror(ary: NDArray, axis: Optional[int] = None) -> Tuple[NDArray, NDArray]:
    """Calculate mean and standard error from array.

    Parameters
    ----------
    ary : NDArray
        Output array containing metrics
    axis : Union[Any, None], optional
        Axis to average over, by default None
    Returns
    -------
    Tuple[NDArray, NDArray]
        Tuple of mean and standard error
    """
    mean = np.mean(ary, axis=axis)
    stderror = scipy.stats.sem(ary, ddof=1, axis=axis)
    return mean, stderror


def _invert_runs_metrics(
    results_dict: dict[str, dict[str, Any]]
) -> dict[str, list[Any]]:
    """Invert dict 'run : metric' to 'metric : run'.

    Parameters
    ----------
    results_dict : dict[str, dict[str, Any]]
        Dict with 'run : metric'
    Returns
    -------
    dict[str, list[Any]]
        Inverted dict with 'metric : run'
    """
    keys = list(results_dict["run_1"].keys())
    inverted_dict: dict[str, list] = {key: [] for key in keys}
    for run in results_dict.keys():
        for key in keys:
            inverted_dict[key].append(results_dict[run][key])
    return inverted_dict


def _default_to_regular(d):
    """Convert defaultdict to regular dict."""
    if isinstance(d, defaultdict):
        d = {k: _default_to_regular(v) for k, v in d.items()}
    return d


def get_metrics(
    experiment: str, config_ids: list[str]
) -> tuple[dict[str, dict[str, dict[Any, Any]]], dict[str, dict[str, dict[str, Any]]]]:
    """Read output data and get dataset metrics (computed in analysis script).

    Parameters
    ----------
    experiment : str
        Experiment name
    config_ids : list[str]
        List of config IDs
    Returns
    -------
    tuple
        Tuple of unaggregated and aggregated metrics dicts
    """
    # paths
    output_paths = get_output_paths(experiment, config_ids)

    # metric
    results_raw = {}

    # get dict with metrics for all configs
    for config_id, output_path in zip(config_ids, output_paths):
        logger.info("Loading checkpoint", output_path=output_path)
        metrics_dict = read_pickle(output_path)["metrics"]
        results_raw[config_id] = metrics_dict

    results_unagg, results_agg = aggregate_metrics(results_raw)

    return results_unagg, results_agg


def aggregate_metrics(
    results_raw: dict[str, dict[str, dict[str, list[float]]]]
) -> tuple[dict, dict]:
    """Aggregate metrics.

    Concat logs of different epochs and aggregate train metrics over the different runs,
    compute mean and stderror.

    Parameters
    ----------
    results_raw : dict[str, dict[str, dict[str, list[float]]]]
        Results saved by experiment

    Returns
    -------
    tuple[dict, dict]
        Tuple of unaggregated and aggregated metrics dicts
    """
    results_unagg = results_raw
    results_agg: defaultdict = defaultdict(lambda: defaultdict(dict))
    for config_id, config_value in results_raw.items():
        results_unagg[config_id] = config_value
        hist_dict_inv = _invert_runs_metrics(config_value)
        for metric_key, metric_value in hist_dict_inv.items():
            mean, stderr = mean_stderror(np.array(metric_value), axis=0)
            results_agg[config_id][metric_key]["mean"] = mean
            results_agg[config_id][metric_key]["stderr"] = stderr
    results_agg = _default_to_regular(results_agg)
    return results_unagg, results_agg


def get_config_id_print(
    config_id: str, config2legend: Optional[dict] = None, exact: bool = False
) -> str:
    """Get the legend name for a given config_id.

    Parameters
    ----------
    config_id : str
        Configuration ID.
    config2legend : dict
        Dictionary mapping config_id to legend name.
    exact : bool, optional
        Whether to find exact match in config2legend, by default False

    Returns
    -------
    str
        Legend name for the given config id
    """
    if config2legend is None:
        return config_id

    if exact:
        return config2legend.get(config_id, config_id)
    else:
        config_id = config_id
        for key, value in config2legend.items():
            if config_id.startswith(key):
                config_id = value
        return config_id


def print_table_from_dict(
    eval_dict: dict,
    exp_name: Optional[str] = None,
    exclude_metrics: Optional[list[str]] = None,
    config2legend: Optional[dict] = None,
    legend_exact: bool = False,
    decimals: int = 4,
    save: bool = False,
    save_kwargs: Optional[dict] = None,
) -> None:
    """Print table of results, for all configs, averaged over runs.

    Parameters
    ----------
    eval_dict : dict
        Dictionary with evaluation results
    exp_name : Optional[str], optional
        Experiment name, by default None
    exclude_metrics : Optional[list[str]], optional
        List of metrics to exclude, by default None
    config2legend : Optional[dict], optional
        Mapping from config to legend name, by default None
    legend_exact : bool, optional
        Whether to find exact match in config2legend, by default False
    decimals : int, optional
        Number of decimals, by default 4
    save : bool, optional
        Whether to save the table, by default False
    save_kwargs : Optional[dict], optional
        Dictionary with save arguments, by default None

    Raises
    ------
    ValueError
        If experiment name is not provided when saving
    ValueError
        If type is unknown
    """
    if save and exp_name is None:
        raise ValueError("Please provide the experiment name.")
    if exclude_metrics is None:
        exclude_metrics = []

    # get list of metric names
    metric_names = []
    config_ids = list(eval_dict.keys())
    for metric_key in eval_dict[config_ids[0]].keys():
        if metric_key in exclude_metrics:
            continue
        metric_names.append(metric_key)

    table = [["Config"] + metric_names]
    # iterate over configs and append rows to table
    for config_id in eval_dict.keys():
        config_id_print = get_config_id_print(
            config_id=config_id, config2legend=config2legend, exact=legend_exact
        )
        row = [config_id_print]
        for metric_key, metric_value in eval_dict[config_id].items():
            if metric_key not in metric_names:
                continue
            if isinstance(metric_value, dict):
                entry = f"{metric_value['mean']:.{decimals}f} ± {metric_value['stderr']:.{decimals}f}"
            elif isinstance(metric_value, float):
                entry = f"{metric_value:.{decimals}f}"
            else:
                raise ValueError("Unknown type.")
            row.append(entry)
        table.append(row)
    tabu_table = tabulate(table, headers="firstrow", tablefmt="psql")

    if save:
        ensure_dir(os.path.dirname(save_kwargs["fname"]))
        with open(save_kwargs["fname"], "w", encoding="UTF-8") as text_file:
            text_file.write(tabu_table)
    print(tabu_table)


def get_train_logs(exp_name: str, config_id: str, run_id: int) -> tuple:
    """Get training logs to inspect learning convergence.

    Parameters
    ----------
    exp_name : str
        Experiment name
    config_id : str
        Config ID
    run_id : Optional[int], optional
        Run ID, by default None

    Returns
    -------
    tuple
        Tuple of train log, lines, and evaluation results
    """
    output_path = get_output_paths(exp_name, [config_id])[0]
    log_history = read_pickle(output_path)["convergence"][f"run_{run_id}"]["log_history"]

    train_log, lines, eval_results = parse_log_history(
        log_history
    )  # NOTE: func from transformers.modelcard
    return train_log, lines, eval_results


def merge_run_results(output_dir: str) -> list:
    """Merge all results from different runs within a config.

    Parameters
    ----------
    output_dir : str
        Path to output directory

    Returns
    -------
    list
        List with run IDs
    """
    output_paths = glob.glob(os.path.join(output_dir, "*.pickle"))

    # check if overlap in run IDs
    run_id_list = [
        re.search(r"run_(\d+)", output_path).group(1) for output_path in output_paths
    ]
    run_id_list = [int(run_id) for run_id in run_id_list]
    run_id_list.sort()
    if len(run_id_list) != len(set(run_id_list)):
        raise ValueError("Overlap in run IDs!")

    all_metrics = {}
    all_convergences = {}
    all_preds = {}
    all_label_maps = {}
    all_bin_edges = {}
    for output_path in output_paths:
        run_n = re.search(r"run_(\d+)", output_path).group(1)
        result = read_pickle(output_path)

        all_metrics[f"run_{run_n}"] = result["metrics"]
        all_convergences[f"run_{run_n}"] = result["convergence"]
        all_preds[f"run_{run_n}"] = result["preds"]
        all_label_maps[f"run_{run_n}"] = result["label_map"]
        all_bin_edges = result["bin_edges"]

    path = Path(output_dir)
    logger.info(
        f"Merging runs {run_id_list} in: "
        f"{os.path.join(path.parent, f'{path.name}.pickle')}"
    )
    write_pickle(
        {
            "metrics": all_metrics,
            "convergence": all_convergences,
            "preds": all_preds,
            "label_map": all_label_maps,
            "bin_edges": all_bin_edges,
        },
        save_dir=path.parent,
        fname=path.name,
    )
    return run_id_list


def merge_all_results(experiment: str, config_ids: list[str]) -> dict:
    """Merge results for all configs.

    Parameters
    ----------
    experiment : str
        Experiment name
    config_ids : list[str]
        List of config IDs

    Returns
    -------
    dict
        Dictionary with config_id to run_id list
    """
    run_id_dict = {}
    # paths
    output_dir = os.path.join("output", experiment)
    output_paths = [os.path.join(output_dir, config_id) for config_id in config_ids]

    for output_path in output_paths:
        run_id_list = merge_run_results(output_path)
        run_id_dict[Path(output_path).name] = run_id_list
    return run_id_dict


def compute_label_map(dataset: Dataset, name: str, num_classes: int) -> dict:
    """Find label map dictionary from HF dataset.

    Parameters
    ----------
    dataset : Dataset
        HF dataset
    name : str
        Dataset name
    num_classes : int
        Number of classes

    Returns
    -------
    dict
        Label map dictionary (type {int: str})
    """
    if isinstance(dataset.features["label"], ClassLabel):
        label_str = dataset.features["label"].names
    else:
        label_str = get_labels_binedges(name, num_classes)[0]
    label_int = np.unique(np.array(dataset["label"]).astype(int))
    label_map = {k: v for k, v in zip(label_int, label_str)}
    return label_map


class Dict2Class(object):
    """Turns a dictionary into a class with attributes.

    Parameters
    ----------
    object : dict
        dict of config parameters
    """

    def __init__(self, my_dict: dict):
        """Turn a dictionary into a class with attributes.

        Parameters
        ----------
        my_dict : dict
            Dict to convert to class
        """
        for key in my_dict:
            setattr(self, key, my_dict[key])


def get_label_map(exp_name: str, config_id: str) -> dict:
    """Get label map to analyze predictions.

    Parameters
    ----------
    exp_name : str
        Experiment name
    config_id : str
        Configuration ID

    Returns
    -------
    dict
        Label map
    """
    run_id = 1  # NOTE: label_map is same for all runs
    output_path = get_output_paths(exp_name, [config_id])[0]
    label_map = read_pickle(output_path)["label_map"][f"run_{run_id}"]
    return label_map


def get_single_pred_label(
    exp_name: str,
    config_id: str,
    run_id: int,
    config_dict: Optional[dict] = None,
) -> tuple:
    """Get predictions and labels.

    Parameters
    ----------
    exp_name : str
        Experiment name
    config_id : str
        Config ID
    run_id : int
        Run ID
    config_dict : Optional[dict], optional
        Config dictionary, by default None

    Returns
    -------
    tuple
        Tuple of predictions and true labels
    """
    output_path = get_output_paths(exp_name, [config_id])[0]

    output, true_label = read_pickle(output_path)["preds"][f"run_{run_id}"]
    logger.info("Loading preds", run_id=run_id, config_id=config_id)

    return output, true_label


def get_all_pred_label(
    exp_name: str,
    config_id: str,
    run_id: Optional[int] = None,
) -> list[tuple]:
    """Get predictions and labels for run_id.

    Parameters
    ----------
    exp_name : str
        Experiment name
    config_id : str
        Config ID
    run_id : Optional[int], optional
        Run ID, by default None

    Returns
    -------
    list[tuple]
        List of tuples containing predictions and true labels
    """
    output_path = get_output_paths(exp_name, [config_id])[0]

    # get list of tuples
    tuple_list = []
    if run_id is None:
        for run_value in read_pickle(output_path)["preds"].values():
            tuple_list.append(run_value)
    else:
        output, true_label = read_pickle(output_path)["preds"][f"run_{run_id}"]
        tuple_list.append((output, true_label))

    bin_edges = read_pickle(output_path)["bin_edges"]

    return tuple_list, bin_edges


def compute_single_confusion_matrix(
    y_true_label: np.ndarray,
    y_pred_label: np.ndarray,
    int2label: dict,
    normalize: str = "all",
) -> np.ndarray:
    """Compute confusion matrix.

    Parameters
    ----------
    y_true_label : np.ndarray
        True labels
    y_pred_label : np.ndarray
        Predicted labels
    int2label : dict
        Mapping from integer labels to string labels
    normalize : str, optional
        Confusion matrix normalization, by default "all"

    Returns
    -------
    np.ndarray
        Confusion matrix
    """
    # convert to string labels
    y_true_label_str = np.vectorize(int2label.get)(y_true_label)
    y_pred_label_str = np.vectorize(int2label.get)(y_pred_label)

    # get confusion matrix
    conf_matrix = confusion_matrix(
        y_true_label_str,
        y_pred_label_str,
        labels=list(int2label.values()),
        normalize=normalize,
    )
    return conf_matrix


def compute_avg_confusion_matrix(
    exp_name: str,
    config_id: str,
    int2label: dict,
    run_id: Optional[int] = None,
    config_dict: Optional[dict] = None,
    normalize: str = "all",
) -> np.ndarray:
    """Compute average confusion matrix.

    Parameters
    ----------
    exp_name : str
        Experiment name
    config_id : str
        Config ID
    int2label : dict
        Mapping from integer labels to string labels
    run_id : Optional[int], optional
        Run ID, by default None
    config_dict : Optional[dict], optional
        Config dictionary, by default None
    normalize : str, optional
        Confusion matrix normalization, by default "all"

    Returns
    -------
    np.ndarray
        Average confusion matrix
    """
    logger.info(
        "Computing average confusion matrix", config_id=config_id, run_id=run_id
    )

    num_classes = config_dict[config_id]["MODEL"]["NUM_LABELS"]
    assert num_classes == len(int2label)

    tuple_list, bin_edges = get_all_pred_label(
        exp_name=exp_name,
        config_id=config_id,
        run_id=run_id,
    )
    assert num_classes == len(bin_edges) - 1

    conf_matrix_list = []
    for output, true_label in tuple_list:
        # NOTE: need to postprocess the predictions
        post_output, _, _ = postprocess_preds(
            eval_pred=(output, true_label),
            output_type=config_dict[config_id]["LOADER"]["OUTPUT_TYPE"],
            num_classes=config_dict[config_id]["MODEL"]["NUM_LABELS"],
            bin_edges=bin_edges,
        )
        conf_matrix = compute_single_confusion_matrix(
            y_true_label=true_label,
            y_pred_label=post_output,
            int2label=int2label,
            normalize=normalize,
        )
        conf_matrix_list.append(conf_matrix)
    avg_conf_matrix = np.mean(np.array(conf_matrix_list), axis=0)
    return avg_conf_matrix


def compute_avg_rmse_per_level(
    exp_name: str,
    config_ids: list[str],
    run_id: Optional[int] = None,
    config_dict: Optional[dict] = None,
) -> dict:
    """Compute average RMSE per level for all config_ids.

    Parameters
    ----------
    exp_name : str
        Experiment name
    config_ids : list[str]
        List of config_ids
    run_id : Optional[int], optional
        Run ID, by default None
    config_dict : Optional[dict], optional
        Dictionary of config_id to config, by default None

    Returns
    -------
    dict
        Averaged RMSE per level for all config_ids
    """
    rmse_agg = {config_id: {} for config_id in config_ids}

    def dict_key(key) -> str:
        return f"RMSE\n{key}"

    for config_id in config_ids:
        int2label = get_label_map(exp_name, config_id)
        tuple_list, bin_edges = get_all_pred_label(
            exp_name=exp_name,
            config_id=config_id,
            run_id=run_id,
        )
        rmse_single_unagg = {dict_key(key): [] for key in int2label.values()}
        rmse_single_agg = rmse_single_unagg.copy()

        for output, true_label in tuple_list:
            # compute RMSE per level
            for label in int2label.keys():
                label_idx = np.where(true_label == label)[0]
                # NOTE: need to postprocess the predictions
                post_output_subset, post_true_subset = postprocess_preds(
                    eval_pred=(output[label_idx], true_label[label_idx]),
                    output_type=config_dict[config_id]["LOADER"]["OUTPUT_TYPE"],
                    num_classes=config_dict[config_id]["MODEL"]["NUM_LABELS"],
                    bin_edges=bin_edges,
                )
                rmse = root_mean_squared_error(post_true_subset, post_output_subset)
                rmse_single_unagg[dict_key(int2label[label])].append(rmse)

        # aggregate
        for key, value in rmse_single_agg.items():
            mean, stderr = mean_stderror(np.array(value), axis=0)
            rmse_single_agg[key] = {"mean": mean, "stderr": stderr}
        rmse_agg[config_id] = rmse_single_agg

    return rmse_agg


def compute_avg_rps_per_level(
    exp_name: str,
    config_ids: list[str],
    run_id: Optional[int] = None,
    config_dict: Optional[dict] = None,
) -> dict:
    """Compute average RPS per level for all config_ids.

    Parameters
    ----------
    exp_name : str
        Experiment name
    config_ids : list[str]
        List of config_ids
    run_id : Optional[int], optional
        Run ID, by default None
    config_dict : Optional[dict], optional
        Dictionary of config_id to config, by default None

    Returns
    -------
    dict
        Averaged RPS per level for all config_ids
    """
    metric_agg = {config_id: {} for config_id in config_ids}

    def dict_key(key) -> str:
        return f"RPS\n{key}"

    for config_id in config_ids:
        int2label = get_label_map(exp_name, config_id)
        tuple_list, bin_edges = get_all_pred_label(
            exp_name=exp_name,
            config_id=config_id,
            run_id=run_id,
        )
        metric_single_unagg = {dict_key(key): [] for key in int2label.values()}
        metric_single_agg = metric_single_unagg.copy()

        for output, true_label in tuple_list:
            # compute RPS per level
            for label in int2label.keys():
                label_idx = np.where(true_label == label)[0]
                # NOTE: need to postprocess the predictions
                _, pred_labels_subset, true_labels_subset = postprocess_preds(
                    eval_pred=(output[label_idx], true_label[label_idx]),
                    output_type=config_dict[config_id]["LOADER"]["OUTPUT_TYPE"],
                    num_classes=config_dict[config_id]["MODEL"]["NUM_LABELS"],
                    bin_edges=bin_edges,
                )
                metric = rps(
                    torch.tensor(true_labels_subset, dtype=torch.int64),
                    torch.from_numpy(pred_labels_subset),
                )
                metric_single_unagg[dict_key(int2label[label])].append(metric)

        # aggregate
        for key, value in metric_single_agg.items():
            mean, stderr = mean_stderror(np.array(value), axis=0)
            metric_single_agg[key] = {"mean": mean, "stderr": stderr}
        metric_agg[config_id] = metric_single_agg

    return metric_agg


def get_results_dict(
    exp_name: str,
    config_ids: list[str],
    run_id: Optional[int] = None,
) -> dict:
    """Get results dictionary for all configs, averaged over runs.

    Parameters
    ----------
    exp_name : str
        Experiment name
    config_ids : list[str]
        List of config ids
    run_id : Optional[int], optional
        Run id, by default None

    Returns
    -------
    dict
        Dictionary with evaluation results
    """
    results_unagg, results_agg = get_metrics(exp_name, config_ids)
    results_tmp = results_agg if run_id is None else results_unagg[f"run_{run_id}"]
    return results_tmp


def reorder_config_ids(config_ids: list, config2legend_dict: dict) -> list:
    """Reorder config_ids according to CONFIG2LEGEND_DICT keys.

    Parameters
    ----------
    config_ids : list
        Config IDs.
    config2legend_dict : dict
        Config2legend dictionary.

    Returns
    -------
    list
        Config IDs reordered.
    """
    config_ids_reorder = []
    for config_id_short in config2legend_dict.keys():
        for config_id in config_ids:
            if config_id.startswith(config_id_short):
                config_ids_reorder.append(config_id)
    return config_ids_reorder


def count_rank_inconsistencies(preds: NDArray) -> NDArray:
    """Count rank inconsistencies in ordinal predictions.

    Parameters
    ----------
    preds : NDArray
        Batch of test outputs of extended binary classifiers, shape (N,C-1).

    Returns
    -------
    NDArray
        Number of rank inconsistencies per test observation, shape (N,).
    """
    diffs = np.diff(preds)
    incons_arr = np.sum(diffs > 0, axis=-1)
    return incons_arr


def compute_rank_probs(pred_logits: torch.Tensor, output_type: OutputType) -> NDArray:
    """Compute rank probabilities from logits, e.g., P(y > r_{K-1}).

    Parameters
    ----------
    pred_logits : torch.Tensor
        Predicted logits (direct output from model), of shape (N, K-1)
    output_type : OutputType
        Output type of the model

    Returns
    -------
    NDArray
        Rank probabilities, of shape (N, K-1)

    Raises
    ------
    ValueError
        If output type is not supported
    """
    if output_type in [OutputType.ORD_OR_NN, OutputType.ORD_CORAL]:
        pred_probs = torch.sigmoid(torch.from_numpy(pred_logits))
    elif output_type == OutputType.ORD_CORN:
        pred_probs = torch.sigmoid(torch.from_numpy(pred_logits))
        pred_probs = torch.cumprod(pred_probs, dim=1)
    else:
        raise ValueError(f"Output type {output_type} not supported")
    return pred_probs.numpy()


def compute_rank_inconsistencies(
    exp_name: str,
    config_id: str,
    config_dict: dict,
    run_id: Optional[int] = None,
) -> tuple[float, NDArray]:
    """Compute rank inconsistencies.

    Parameters
    ----------
    exp_name : str
        Experiment name
    config_id : str
        Config ID
    config_dict : dict
        Config dictionary
    run_id : Optional[int], optional
        Run ID, by default None

    Returns
    -------
    tuple[float, NDArray]
        Sum of rank inconsistencies and rank inconsistencies per observation
    """
    logger.info("Computing rank inconsistencies", config_id=config_id, run_id=run_id)

    tuple_list, _ = get_all_pred_label(
        exp_name=exp_name,
        config_id=config_id,
        run_id=run_id,
    )

    counts_list = []
    for output, _ in tuple_list:  # NOTE: one loop for every run
        # NOTE: obtain rank probabilities from logits (P(y > r_{K-1}))
        pred_probs = compute_rank_probs(
            pred_logits=output,
            output_type=config_dict[config_id]["LOADER"]["OUTPUT_TYPE"],
        )
        counts = count_rank_inconsistencies(pred_probs)
        counts_list.append(counts)
    count_per_obs = np.mean(np.array(counts_list), axis=0)
    sum_count = np.sum(count_per_obs)
    return sum_count, count_per_obs


def get_logit_params_history(exp_name: str, config_id: str, run_id: int) -> dict:
    """Get parameters history for ordered logit model.

    Parameters
    ----------
    exp_name : str
        Experiment name
    config_id : str
        Config ID
    run_id : Optional[int], optional
        Run ID, by default None

    Returns
    -------
    dict
        Dictionary with parameters history
    """
    output_path = get_output_paths(exp_name, [config_id])[0]
    params_history = read_pickle(output_path)["convergence"][f"run_{run_id}"]["params_history"]
    return params_history
