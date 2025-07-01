"""Module for random prediction trainer."""

# standard library imports
from typing import Any, Callable, Optional

# related third party imports
import numpy as np
import structlog
from numpy.typing import ArrayLike
from yacs.config import CfgNode

# local application/library specific imports
from tools.constants import ROUND_FLOAT, BaselineType, OutputType, TEST
from tools.metrics import compute_metrics
from trainer.build import TRAINER_REGISTRY

# set up logger
logger = structlog.get_logger(__name__)

METRIC_KEY_PREFIX = "test"


@TRAINER_REGISTRY.register("baseline")
def build_baseline_trainer() -> Callable:
    """Build baseline trainer.

    Returns
    -------
    Callable
        Trainer function
    """
    return baseline_trainer


def baseline_trainer(
    cfg: CfgNode,
    datasets,
    device: Optional[Any] = None,  # NOTE: not used
    dry_run: bool = False,  # NOTE: not used
    bin_edges: Optional[ArrayLike] = None,
) -> dict:
    """Baseline prediction trainer.

    Can be either random or majority prediction.

    Parameters
    ----------
    cfg : CfgNode
        Config object
    datasets : _type_
        HF datasets
    device : Any
        Compute device (GPU or CPU)
    dry_run : bool
        Run only one AL epoch, by default False
    bin_edges : Optional[ArrayLike]
        Bin edges for discretization, by default None

    Returns
    -------
    dict
        Metrics
    """
    try:
        method = BaselineType(cfg.MODEL.NAME)
    except ValueError:
        raise ValueError(f"Invalid baseline: {cfg.MODEL.NAME}")
    ##### EVALUATION #####
    logger.info(
        "Evaluate - start",
        test_size=len(datasets["test"]),
    )
    # evaluate
    train_labels = datasets["train"]["label"]

    if cfg.LOADER.OUTPUT_TYPE == OutputType.REGR:
        if method == BaselineType.RANDOM:
            eval_preds = np.random.uniform(
                low=bin_edges[0],
                high=bin_edges[-1],
                size=len(datasets[TEST]["label"]),
            )
        elif method == BaselineType.MAJORITY:
            unique, counts = np.unique(train_labels, return_counts=True)
            unique = np.flip(unique)  # NOTE: for balanced ARC, make sure level is 8
            counts = np.flip(counts)
            majority = unique[np.argmax(counts)]
            # logger.debug(np.asarray((unique, counts)).T)
            # logger.debug(f"{majority = }")
            eval_preds = np.repeat(majority, len(datasets[TEST]["label"]))
    else:
        logger.error(
            "Baseline trainers are only implemented for regression tasks. "
            "Results are identical for other output types."
        )

    # compute metrics from preds
    test_pred_label = (eval_preds, np.array(datasets[TEST]["label"]))
    test_metrics = compute_metrics(
        test_pred_label,
        output_type=cfg.LOADER.OUTPUT_TYPE,
        num_classes=cfg.MODEL.NUM_LABELS,
        bin_edges=bin_edges,
    )
    test_metrics = {f"{METRIC_KEY_PREFIX}_{k}": v for k, v in test_metrics.items()}

    logger.info(
        "Evaluate - end",
        test_bal_rps=round(test_metrics["test_bal_rps"], ROUND_FLOAT),  # NOTE: metric
    )

    return test_metrics, None, test_pred_label
