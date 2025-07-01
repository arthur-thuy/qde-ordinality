"""Build file for tuning."""

# standard library imports
# /

# related third party imports
import optuna
import structlog
from yacs.config import CfgNode

# local application/library specific imports
from tools.constants import MODEL_NAME_SHORT
from tools.registry import Registry

TUNER_REGISTRY = Registry()

logger = structlog.get_logger(__name__)


def build_tuner_cfg(cfg: CfgNode, trial: optuna.Trial) -> CfgNode:
    """Build tuner cfg from trial suggestions.

    Parameters
    ----------
    cfg : CfgNode
        Config object
    trial : optuna.Trial
        Optuna trial object

    Returns
    -------
    CfgNode
        Tuner config object
    """
    search_name = (
        f"{cfg.LOADER.NAME}_{MODEL_NAME_SHORT[cfg.MODEL.NAME]}_{cfg.LOADER.OUTPUT_TYPE}"
    )
    logger.info("Building tuner", search_name=search_name)
    tuner_cfg = TUNER_REGISTRY[search_name](cfg, trial)
    return tuner_cfg
