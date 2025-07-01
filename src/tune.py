"""Module for hyperparam tuning."""

# standard library imports
import argparse
import os
import time
from functools import partial

# related third party imports
import optuna
import structlog
import torch
import torch.backends
from yacs.config import CfgNode

# local application/library specific imports
from data_loader.build import build_dataset
from tools.analyzer import compute_label_map
from tools.configurator import (
    check_cfg,
    load_configs,
    save_config,
)
from tools.constants import TRAIN, OPTUNA_STUDY_DIRECTION, METRIC_BEST_MODEL
from tools.optuna_analyzer import continue_or_start_new_study, inspect_trial
from tools.utils import (
    print_elapsed_time,
    write_pickle,
    set_device,
    set_seed,
    remove_hf_checkpoint,
)
from trainer.build import build_trainer
from tuner.build import build_tuner_cfg

# set up logger
logger = structlog.get_logger(__name__)

# allow TF32
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

parser = argparse.ArgumentParser(description="PyTorch hyperparam tuning")
parser.add_argument(
    "config",
    type=str,
    help="config file path",
)
parser.add_argument(
    "--dry-run", action="store_true", default=False, help="run a single epoch"
)

N_TRIALS = 20  # TODO: make more accessible


def objective(trial: optuna.Trial, cfg: CfgNode) -> float:
    """Do train run with suggested hyperparams.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial object
    cfg : CfgNode
        Config object

    Returns
    -------
    float
        Validation C-index
    """
    tuner_cfg = build_tuner_cfg(cfg=cfg, trial=trial)

    metric = train(tuner_cfg, trial_n=trial.number)
    return metric


def train(cfg: CfgNode, trial_n: int) -> float:
    """Train model and return validation nll.

    Parameters
    ----------
    cfg : CfgNode
        Config object

    Returns
    -------
    float
        Validation C-index
    """
    print("\n", "*" * 10, f"Trial: {cfg.ID}", "*" * 10)

    start_time = time.time()

    # device
    device = set_device(cfg.DEVICE.NO_CUDA, cfg.DEVICE.NO_MPS)

    # build datasets
    datasets, bin_edges = build_dataset(
        loader_cfg=cfg.LOADER, num_classes=cfg.MODEL.NUM_LABELS, seed=cfg.SEED
    )
    label_map = compute_label_map(
        dataset=datasets[TRAIN], name=cfg.LOADER.NAME, num_classes=cfg.MODEL.NUM_LABELS
    )

    # seed
    set_seed(cfg.SEED)

    # build and run trainer
    trainer_func = build_trainer(model_cfg=cfg.MODEL)
    metrics, convergence, preds = trainer_func(
        cfg=cfg,
        datasets=datasets,
        device=device,
        dry_run=args.dry_run,
        bin_edges=bin_edges,
    )

    write_pickle(
        {
            "metrics": metrics,
            "convergence": convergence,
            "preds": preds,
            "label_map": label_map,
            "bin_edges": bin_edges,
        },
        save_dir=cfg.OUTPUT_DIR,
        fname=f"t{trial_n}_{cfg.ID}",
    )
    print_elapsed_time(start_time, run_id=1)

    save_config(cfg, save_dir=cfg.OUTPUT_DIR, fname=cfg.ID)
    remove_hf_checkpoint(read_dir=cfg.OUTPUT_DIR)

    return metrics["val_bal_rps"]  # NOTE: metric


if __name__ == "__main__":
    """Run tuning."""
    args = parser.parse_args()

    # config
    configs = load_configs(args.config, freeze=False)

    for cfg in configs:
        print("\n", "=" * 10, f"Config: {cfg.TUNE_ID}", "=" * 10)

        # change OUTPUT_DIR
        file_w_prefix = f"tune_{os.path.basename(cfg.OUTPUT_DIR)}"
        cfg.OUTPUT_DIR = os.path.join(
            os.path.dirname(cfg.OUTPUT_DIR), file_w_prefix, cfg.TUNE_ID
        )

        # logical check
        check_cfg(cfg)

        # continue or start new study
        study = continue_or_start_new_study(cfg)
        if study is None:
            # start new study
            logger.info("Creating new study object.")
            study = optuna.create_study(
                direction=OPTUNA_STUDY_DIRECTION[
                    METRIC_BEST_MODEL[cfg.LOADER.OUTPUT_TYPE]
                ]
            )
        objective = partial(objective, cfg=cfg)
        study.optimize(objective, n_trials=(1 if args.dry_run else N_TRIALS))

        print("Number of finished trials: ", len(study.trials))
        inspect_trial(study=study, trial_number=None)

        write_pickle(
            {
                "trials": study.trials,
                "best_trial": study.best_trial,
                "study": study,
            },
            save_dir=cfg.OUTPUT_DIR,
            fname="overview",
        )
