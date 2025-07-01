"""Build file for data loader."""

# standard library imports
# /

# related third party imports
import structlog
from yacs.config import CfgNode

# local application/library specific imports
from data_loader.data_loader import QDET

logger = structlog.get_logger(__name__)


def build_dataset(loader_cfg: CfgNode, num_classes: int, seed: int) -> tuple:
    """Build the HuggingFace dataset.

    Parameters
    ----------
    loader_cfg : CfgNode
        Data loader config object
    num_classes : int
        Number of classes
    seed : int
        Random seed

    Returns
    -------
    _type_
        Train/Val/Test datasets + bin_edges
    """
    logger.info("Building dataset", name=loader_cfg.NAME)
    loader = QDET(
        name=loader_cfg.NAME,
        num_classes=num_classes,
        output_type=loader_cfg.OUTPUT_TYPE,
        small_dev=loader_cfg.SMALL_DEV,
        balanced=loader_cfg.BALANCED,
        discretizer=loader_cfg.DISCRETIZER,
        start_end=loader_cfg.START_END,
        temperature=loader_cfg.TEMPERATURE,
        seed=seed,
    )

    return loader.load_all()
