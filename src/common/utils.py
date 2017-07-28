"""Some utilities for learning."""

import numpy as np
from scipy import linalg

def random_orthogonal_matrix(n):
    """Returns a random, orthogonal matrix of n by n."""
    a = np.random.randn(n, n)
    U, _, _ = linalg.svd(a)
    assert U.shape == (n, n)
    return U
