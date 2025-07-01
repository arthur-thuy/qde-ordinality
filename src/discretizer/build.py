"""Build file for discretization."""

# standard library imports
from typing import Optional

# related third party imports
import structlog
from numpy.typing import ArrayLike

# local application/library specific imports
from tools.registry import Registry

DISCRETIZER_REGISTRY = Registry()

logger = structlog.get_logger(__name__)


def build_discretizer(
    arr: ArrayLike,
    method: str,
    bins: int,
    start_end: Optional[tuple[int, int]] = None,
    temperature: float = 1.0,
) -> ArrayLike:
    logger.info("Building discretizer", type=method)
    bin_edges = DISCRETIZER_REGISTRY[method](
        arr=arr, bins=bins, start_end=start_end, temperature=temperature
    )
    logger.info(
        f"Discretized into {len(bin_edges) - 1} bins, in range [{bin_edges[0]}, {bin_edges[-1]}]"
    )
    return bin_edges
