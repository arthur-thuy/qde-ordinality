"""Module for discretizing continuous difficulty."""

# standard library imports
from typing import Optional, Literal

# related third party imports
import numpy as np
from numpy.typing import ArrayLike, NDArray

# local application/library specific imports
from discretizer.build import DISCRETIZER_REGISTRY


@DISCRETIZER_REGISTRY.register("equal_width")
def build_equal_width(
    arr: ArrayLike,
    bins: int,
    start_end: Optional[tuple[int, int]] = None,
    temperature: float = 1.0,  # noqa
) -> NDArray:
    bin_edges = get_bins_equal_width(arr, bins=bins, start_end=start_end)
    return bin_edges


@DISCRETIZER_REGISTRY.register("variable_width_right")
def build_variable_width_right(
    arr: ArrayLike,
    bins: int,
    start_end: Optional[tuple[int, int]] = None,
    temperature: float = 1.0,
) -> NDArray:
    bin_edges = get_bins_variable_width_right(
        arr,
        bins=bins,
        start_end=start_end,
        temperature=temperature,
    )
    return bin_edges


@DISCRETIZER_REGISTRY.register("variable_width_symmetric")
def build_variable_width_symmetric(
    arr: ArrayLike,
    bins: int,
    start_end: Optional[tuple[int, int]] = None,
    temperature: float = 1.0,
) -> NDArray:
    bin_edges = get_bins_variable_width_symmetric(
        arr,
        bins=bins,
        start_end=start_end,
        temperature=temperature,
    )
    return bin_edges


def get_bins_equal_width(
    arr: ArrayLike, bins: int, start_end: Optional[tuple[int, int]] = None
) -> NDArray:
    """Discretize the labels into equal width bins.

    Parameters
    ----------
    arr : ArrayLike
        Original labels to be discretized.
    bins : int
        Number of bins to discretize the labels into.
    start_end : tuple, optional
        Range, by default None. If None, the range is calculated from the min and max of the array.

    Returns
    -------
    NDArray
        Bin edges.
    """
    if start_end is None:
        start_end = (np.min(arr), np.max(arr))
    bin_edges = np.histogram_bin_edges(arr, bins=bins, range=start_end)
    return bin_edges


def get_bins_variable_width_right(
    arr: ArrayLike,
    bins: int,
    start_end: Optional[tuple[int, int]] = None,
    temperature: float = 1.0,
) -> NDArray:
    """Discretize the labels into variable width bins, in increasing size.

    Parameters
    ----------
    arr : ArrayLike
        Original labels to be discretized.
    bins : int
        Number of bins to discretize the labels into.
    start_end : tuple, optional
        Range, by default None. If None, the range is calculated from the min and max of the array.
    temperature : float, optional
        Agressiveness of with variations, by default 1.0. For 1.0, the bins are of equal width.

    Returns
    -------
    NDArray
        Bin edges.
    """
    if start_end is None:
        start_end = (np.min(arr), np.max(arr))
    start, end = start_end
    bin_edges = [start]

    # Calculate the total length of the range
    total_length = end - start

    # Calculate the width of each bin using an exponential growth factor
    for i in range(1, bins):
        width = (
            total_length
            * (temperature**i)
            / sum(temperature**j for j in range(1, bins + 1))
        )
        bin_edges.append(bin_edges[-1] + width)

    # Ensure the last bin edge is exactly the end of the range
    bin_edges.append(end)

    bin_edges = np.array(bin_edges)
    print(np.diff(bin_edges))
    assert len(bin_edges) - 1 == bins, f"Expected {bins} bins, got {len(bin_edges)-1}"
    assert bin_edges[0] == start, f"Expected start to be {start}, got {bin_edges[0]}"
    assert bin_edges[-1] == end, f"Expected end to be {end}, got {bin_edges[-1]}"
    return bin_edges


def _get_bin_indices_inner(
    bins: int,
) -> list:
    """Get bin indices for a symmetric binning scheme where inner bins are largest.

    The largest bins are in the center of the range.

    Parameters
    ----------
    bins : int
        Number of bins.

    Returns
    -------
    list
        List of bin indices, with largest values in the center.

    Raises
    ------
    ValueError
        If the number of bins is not a positive integer.
    """
    if bins <= 0:
        raise ValueError("Number of bins must be a positive integer.")

    half_bins = bins // 2
    widths = [i + 1 for i in range(half_bins)]

    if bins % 2 == 0:
        # Even number of bins
        widths = widths + widths[::-1]
    else:
        # Odd number of bins
        widths = widths + [half_bins + 1] + widths[::-1]

    return widths


def _get_bin_indices_outer(bins: int) -> list:
    """Get bin indices for a symmetric binning scheme where outer bins are largest.

    Parameters
    ----------
    bins : int
        Number of bins.

    Returns
    -------
    list
        List of bin indices, with largest values at the edges.
    """
    if bins <= 0:
        raise ValueError("Number of bins must be a positive integer.")

    half_bins = bins // 2
    # Create reversed widths starting from largest at the edges
    widths = [half_bins - i for i in range(half_bins)]

    if bins % 2 == 0:
        # Even number of bins
        widths = widths + widths[::-1]
    else:
        # Odd number of bins
        widths = widths + [0] + widths[::-1]  # 0 for smallest bin in middle

    return widths


def get_bins_variable_width_symmetric(
    arr: ArrayLike,
    bins: int,
    start_end: Optional[tuple[int, int]] = None,
    temperature: float = 1.0,
    largest: Literal["inner", "outer"] = "outer",
) -> NDArray:
    """Discretize the labels into variable width bins, symmetrical.

    Parameters
    ----------
    arr : ArrayLike
        Original labels to be discretized.
    bins : int
        Number of bins to discretize the labels into.
    start_end : tuple, optional
        Range, by default None. If None, the range is calculated from the min and max of the array.
    temperature : float, optional
        Agressiveness of with variations, by default 1.0. For 1.0, the bins are of equal width.

    Returns
    -------
    NDArray
        Bin edges.
    """
    if start_end is None:
        start_end = (np.min(arr), np.max(arr))
    start, end = start_end
    bin_edges = [start]

    # Calculate the total length of the range
    total_length = end - start

    # Calculate the width of each bin using an exponential growth factor
    if largest == "inner":
        bin_idx = _get_bin_indices_inner(bins=bins)
    else:
        bin_idx = _get_bin_indices_outer(bins=bins)
    for i in bin_idx[:-1]:
        width = total_length * (temperature**i) / sum(temperature**j for j in bin_idx)
        bin_edges.append(bin_edges[-1] + width)

    # Ensure the last bin edge is exactly the end of the range
    bin_edges.append(end)

    bin_edges = np.array(bin_edges)
    assert len(bin_edges) - 1 == bins, f"Expected {bins} bins, got {len(bin_edges)-1}"
    assert bin_edges[0] == start, f"Expected start to be {start}, got {bin_edges[0]}"
    assert bin_edges[-1] == end, f"Expected end to be {end}, got {bin_edges[-1]}"
    bin_distances = np.diff(bin_edges)
    assert np.isclose(
        bin_distances, bin_distances[::-1]
    ).all(), "Expected symmetric bin distances"

    return bin_edges


def discretize_into_bins(
    arr: ArrayLike, bin_edges: ArrayLike, include_right: bool = True
) -> NDArray:
    """Discretize an array into bins.

    Parameters
    ----------
    arr : ArrayLike
        Input array to be discretized.
    bin_edges : ArrayLike
        Bin edges to be used for discretization.
    include_right : bool, optional
        Whether to include the right edge, by default True.

    Returns
    -------
    NDArray
        Discretized array.
    """
    if include_right:
        # NOTE: increase last bin edge by epsilon to include the last value
        bin_edges = np.nextafter(bin_edges, bin_edges + (bin_edges == bin_edges[-1]))
    return np.digitize(arr, bin_edges)
