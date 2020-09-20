from typing import Tuple
import numpy as np
from . import rslc as _rslc


def cluster(
    distance_matrix: np.ndarray, num_clusters: int, min_size: float = None
) -> Tuple[np.ndarray, np.ndarray]:
    """The Robust Single-Linkage Clustering Algorithm

    Args:
        distance_matrix (numpy.ndarray): A square distance matrix between items (complete, not triangular)
        num_clusters (int): The number of clusters to find.
        min_size (int, optional): Minimum cluster-size (used to detect outliers). Can also be a percentage (float between 0 and 1) of the number of items (rows / columns in the distance matrix). Defaults to `num_clusters**-2`.

    Returns:
        numpy.ndarray: An integer vector with a cluster index for each item.
        numpy.ndarray: A boolen vector marking outliers.
    """
    # Default value for min_size
    if min_size is None:
        min_size = num_clusters ** -2
    # Some sanity checks
    assert len(distance_matrix.shape) == 2
    assert distance_matrix.shape[0] == distance_matrix.shape[1]
    assert 0 < num_clusters < distance_matrix.shape[0]
    assert 0 < min_size < distance_matrix.shape[0]
    # Handle minsizes in the for of percentages
    if isinstance(min_size, float) and 0.0 < min_size < 1.0:
        min_size = int(distance_matrix.shape[0] * min_size)
    # Select correct type and run the algorithm
    if distance_matrix.dtype == np.float64:
        return _rslc.rslc_f64(distance_matrix, num_clusters, min_size)
    if distance_matrix.dtype == np.float32:
        return _rslc.rslc_f32(distance_matrix, num_clusters, min_size)
    if distance_matrix.dtype == np.int32:
        return _rslc.rslc_i32(distance_matrix, num_clusters, min_size)
    if distance_matrix.dtype == np.int64:
        return _rslc.rslc_i64(distance_matrix, num_clusters, min_size)
    if distance_matrix.dtype == np.uint32:
        return _rslc.rslc_u32(distance_matrix, num_clusters, min_size)
    if distance_matrix.dtype == np.uint64:
        return _rslc.rslc_u64(distance_matrix, num_clusters, min_size)
    if distance_matrix.dtype == np.int16:
        return _rslc.rslc_i16(distance_matrix, num_clusters, min_size)
    if distance_matrix.dtype == np.uint16:
        return _rslc.rslc_u16(distance_matrix, num_clusters, min_size)
    return _rslc.rslc_f64(distance_matrix.astype(np.float64), num_clusters, min_size)
