# Test that the rust extension works from python

import numpy as np
from rslc import cluster

def test_rslc():
    # Distance  matrix from a "smiley-face"
    distances = np.array([
        [0,1,1, 4,4,4, 2,2,3,4,5,6, 5,6],
        [1,0,1, 4,4,4, 2,2,3,4,5,6, 5,6],
        [1,1,0, 4,4,4, 2,2,3,4,5,6, 5,6],

        [4,4,4, 0,1,1, 6,5,4,3,2,2, 6,5],
        [4,4,4, 1,0,1, 6,5,4,3,2,2, 6,5],
        [4,4,4, 1,1,0, 6,5,4,3,2,2, 6,5],

        [2,2,2 ,6,6,6, 0,1,2,3,4,5, 4,6],
        [2,2,2 ,5,5,5, 1,0,1,2,3,4, 3,5],
        [3,3,3 ,4,4,4, 2,1,0,1,2,3, 3,4],
        [4,4,4 ,3,3,3, 3,2,1,0,1,2, 4,3],
        [5,5,5 ,2,2,2, 4,3,2,1,0,1, 5,3],
        [6,6,6 ,2,2,2, 5,4,3,2,1,0, 6,4],

        [5,5,5 ,6,6,6, 4,3,3,4,5,6, 0,4],
        [6,6,6 ,5,5,5, 6,5,4,3,3,4, 4,0]])

    # Correct Output
    clusters = np.array([0,0,0, 1,1,1, 2,2,2,2,2,2, 2,2])
    outliers = np.array([False,False,False, False,False,False, False,False,False,False,False,False, True,True])

    # Try with an array of float64
    (c, o) = cluster(distances.astype(np.float64), 3, 2)
    np.testing.assert_array_almost_equal(c, clusters)
    np.testing.assert_array_almost_equal(o, outliers)

    # Try with an array of int16
    (c, o) = cluster(distances.astype(np.int16), 3, 2)
    np.testing.assert_array_almost_equal(c, clusters)
    np.testing.assert_array_almost_equal(o, outliers)

    # Try with an array of uint32
    (c, o) = cluster(distances.astype(np.uint32), 3, 2)
    np.testing.assert_array_almost_equal(c, clusters)
    np.testing.assert_array_almost_equal(o, outliers)
