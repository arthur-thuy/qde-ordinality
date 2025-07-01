"""Module to analyze Optuna results."""

# standard library imports
import glob
import os
from typing import Optional

# related third party imports
import click
import optuna
import structlog
from yacs.config import CfgNode
from transformers.modelcard import parse_log_history

# local application/library specific imports
from tools.analyzer import get_output_paths
from tools.utils import delete_previous_content, read_pickle

logger = structlog.get_logger(__name__)


def get_study_object(exp_name: str, config_id: str) -> optuna.study.Study:
    """Get Optuna study object.

    Parameters
    ----------
    exp_name : str
        Experiment name
    config_id : str
        Configuration ID

    Returns
    -------
    optuna.study.Study
        Optuna study object
    """
    # get path of overview checkpoint
    [output_path] = get_output_paths(os.path.join(exp_name, config_id), ["overview"])
    print(f"{output_path = }")
    study = read_pickle(output_path)["study"]
    return study


def get_history(
    exp_name: str,
    config_id: str,
    study: optuna.study.Study,
    trial_n: Optional[int] = None,
) -> dict:
    """Get history object of given trial.

    Parameters
    ----------
    exp_name : str
        Experiment name
    config_id : str
        Configuration ID
    study : optuna.study.Study
        Study object
    trial_n : Optional[int], optional
        Trial number to return history, by default None

    Returns
    -------
    dict
        Learning convergence history
    """
    if trial_n is None:
        # find number of best trial
        trial_n = study.best_trial.number
        logger.info("Finding best trial number", trial_n=trial_n)

    # find path of checkpoint
    output_dir = os.path.join("output", exp_name, config_id)
    search_path = os.path.join(output_dir, f"t{trial_n}_**")
    find_paths = glob.glob(search_path)
    assert len(find_paths) == 1, "Can only handle one .pickle file"
    [find_path] = find_paths
    # load checkpoint
    log_history = read_pickle(find_path)["convergence"]["log_history"]
    _, lines, _ = parse_log_history(
        log_history
    )  # NOTE: func from transformers.modelcard
    return lines


def continue_or_start_new_study(cfg: CfgNode) -> optuna.study.Study:
    """Continue or start new study.

    Parameters
    ----------
    cfg : CfgNode
        Configuration object

    Returns
    -------
    optuna.study.Study
        Study object
    """
    if os.path.isdir(cfg.OUTPUT_DIR):
        try:
            # NOTE: need path with "tune_{exp_name}" and tune ID
            exp_name = os.path.split(os.path.dirname(cfg.OUTPUT_DIR))[1]
            config_id = os.path.basename(cfg.OUTPUT_DIR)
            study = get_study_object(exp_name=exp_name, config_id=config_id)
            logger.info("Previous content found, loading study object.")
        except FileNotFoundError:
            logger.warning("Previous content found, but no study object.")
            delete_previous_content(cfg)
            study = None
    else:
        study = None

    if study:
        # ask to continue or start new study
        if click.confirm(
            f"Do you want to use previous `study` object in {cfg.OUTPUT_DIR}?",
            default=True,
        ):
            logger.info("Continuing with previous study object.")
        else:
            delete_previous_content(cfg)
            study = None

    return study


def inspect_trial(
    study: optuna.study.Study, trial_number: Optional[int] = None
) -> None:
    """Inspect specific trail.

    Parameters
    ----------
    study : optuna.study.Study
        Optuna study object
    trial_number : Optional[int], optional
        Number of trial to inspect. If None, best trial. By default None
    """
    if trial_number is None:
        trial = study.best_trial
        print(f"Best trial (trial {trial.number}):")
    else:
        trial = study.trials[trial_number]
        print(f"Trial {trial_number}:")
    print(f"  Value: {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
