"""Auxiliary functions and other utilities."""
import numpy as np


def add_uniform_noise(data, p=1):
    """Add uniform noise to a given dataset.

    Function to add uniform random noise to a given dataset.
    Uniform noise in range [-5*10^(-p), 5*10^(-p)] is added to each
    data entry. For integer-valued sequences, p = 1.

    Parameters
    ----------
    data : (N,) array_like
        _Numpy_ array of data to be processed.
    p : int
        Integer parameter controlling noise amplitude.
        Has to be greater than or equal to ``1``.

    Returns
    -------
    (N, ) array_like
        _Numpy_ array with noise-added entries.
    """
    if p < 1:
        raise ValueError("'p' has to be greater or equal to 1.")
    noise = np.random.uniform(-5.*10**(-p), 5*10**(-p), size = len(data))
    randomized_data = data + noise
    # ensure there are no negative entries after noise is added
    randomized_data = \
        randomized_data[np.where(randomized_data > 0)]
    return randomized_data

def logbin_distribution(data, nbins = 30):
    """Log-binning of a distribution.

    Parameters
    ----------
    data : (N, ) array_like
        _Numpy_ array with data to calculate log-binned PDF on.
    nbins: int
        number of logarithmic bins to use.

    Returns
    -------
    (N, ), (N, ) array_like
        _Numpy_ arrays containing midpoints of bins and corresponding PDF values.
    """
    # define the support of the distribution
    lower_bound = min(data)
    upper_bound = max(data)
    # define bin edges
    log = np.log10
    lower_bound = log(lower_bound) if lower_bound > 0 else -1
    upper_bound = log(upper_bound)
    bins = np.logspace(lower_bound, upper_bound, nbins)

    # compute the histogram using numpy
    y, _ = np.histogram(data, bins=bins, density=True)
    # for each bin, compute its midpoint
    x = bins[1:] - np.diff(bins) / 2.0
    # if bin is empty, drop it from the resulting list
    drop_indices = [i for i,k in enumerate(y) if k == 0.0]
    x = [k for i,k in enumerate(x) if i not in drop_indices]
    y = [k for i,k in enumerate(y) if i not in drop_indices]
    return x, y

def get_ccdf(degseq):
    """Get CCDF from a degree sequence.

    Parameters
    ----------
    degseq : (N, ) array_like
        _Numpy_ array of nodes' degrees.

    Returns
    -------
    (N, ) array_like
        unique degree values met in the sequence.
    (N, array_like) :
        corresponding CCDF values.
    """
    uniques, counts = np.unique(degseq, return_counts=True)
    cumprob = np.cumsum(counts).astype(np.double) / (degseq.size)
    return uniques[::-1], (1. - cumprob)[::-1]

def get_eps_stop(data, amse_border=1, sort=True):
    """Get `eps_stop` from AMSE upper bound.

    Upper bound for order statistic to consider for double-bootstrap
    AMSE minimizer. Entries that are smaller or equal to the border value
    are ignored during AMSE minimization.

    Parameters
    ----------
    data : (N, ) array_like
        data : (N, ) array_like
        _Numpy_ array for which double-bootstrap is performed.
        Data has to be sorted in decreasing order.
        By default (``sort=True``) it is sorted, but this leads to copying
        the data. To avoid this pass sorted data and set ``sort=False``.
    sort : bool
        Should data be copied and sorted.

    Returns
    -------
    eps_stop : float
        Parameter controlling range of AMSE minimization.
        Defined as the fraction of order statistics to consider
        during the AMSE minimization step
    """
    if sort:
        data = np.sort(data)[::-1]
    return 1 - len(data[np.where(data <= amse_border)]) / len(data)
