
import numpy as np
from .rslc import *

def cluster(distance_matrix: np.ndarray, num_clusters: int, min_size: int = None) -> (np.ndarray, np.ndarray):
    # Default value for min_size
    if min_size is None:
        min_size = 0.5**(num_clusters + 1)
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
        return rslc_f64(distance_matrix, num_clusters, min_size)
    if distance_matrix.dtype == np.float32:
        return rslc_f32(distance_matrix, num_clusters, min_size)
    if distance_matrix.dtype == np.int32:
        return rslc_i32(distance_matrix, num_clusters, min_size)
    if distance_matrix.dtype == np.int64:
        return rslc_i64(distance_matrix, num_clusters, min_size)
    if distance_matrix.dtype == np.uint32:
        return rslc_u32(distance_matrix, num_clusters, min_size)
    if distance_matrix.dtype == np.uint64:
        return rslc_u64(distance_matrix, num_clusters, min_size)
    if distance_matrix.dtype == np.int16:
        return rslc_i16(distance_matrix, num_clusters, min_size)
    if distance_matrix.dtype == np.uint16:
        return rslc_u16(distance_matrix, num_clusters, min_size)
    return rslc_f64(distance_matrix.astype(np.float64), num_clusters, min_size)
