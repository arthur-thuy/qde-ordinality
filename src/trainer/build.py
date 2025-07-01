"""Build file for trainer."""

# standard library imports
from typing import Callable

# related third party imports
import structlog
from yacs.config import CfgNode

# local application/library specific imports
from tools.registry import Registry

TRAINER_REGISTRY = Registry()

logger = structlog.get_logger(__name__)


def build_trainer(model_cfg: CfgNode) -> Callable:
    """Build the trainer.

    Parameters
    ----------
    model_cfg : CfgNode
        Model config object

    Returns
    -------
    Callable
        Trainer function
    """
    logger.info("Building trainer", type=model_cfg.TYPE)
    trainer = TRAINER_REGISTRY[model_cfg.TYPE]()
    return trainer
