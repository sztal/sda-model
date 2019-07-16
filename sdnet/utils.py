"""Simulation related functions and utilities."""
from itertools import repeat, chain
from random import uniform
from joblib import Parallel, delayed
import numpy as np
from numba import njit


@njit
def Lp_norm(u, p=2):
    x = (np.abs(u)**p).sum()**(1/p)
    return np.float(x)

@njit
def Lp_dist(u, v, p=2, normalized=False):
    if normalized and u.size > 1 and v.size > 1:
        u_norm = u / Lp_norm(u, p=p)
        v_norm = v / Lp_norm(v, p=p)
        return Lp_norm(u_norm - v_norm, p=p)
    return Lp_norm(u - v, p=p)

@njit
def euclidean_dist(u, v, normalized=False):
    return Lp_dist(u, v, p=2, normalized=normalized)

@njit
def manhattan_dist(u, v, normalized=False):
    return Lp_dist(u, v, p=1, normalized=normalized)

def _make_dist_matrix(X, measure, symmetric=True):
    """Generate a distance matrix.

    Parameters
    ----------
    X : array_like (N, k)
        Dataset with nodes' features.
        One row is one node.
    measure : callable
        Measure function that takes two main arguments which are
        feature vectors for two nodes.
    symmetric : bool
        Is the measure function symmetric in the two main arguments.

    Returns
    -------
    (N, N) array_like
        Edge formation probability matrix.
    """
    N = X.shape[0]
    D = np.zeros((N, N))
    if symmetric:
        for i in range(N):
            for j in range(i):
                D[i, j] = D[j, i] = measure(X[i:i+1], X[j:j+1])
    else:
        for i in range(N):
            for j in range(N):
                D[i, j] = measure(X[i:i+1], X[j:j+1])
    return D

make_dist_matrix = njit(_make_dist_matrix)

def run_simulations(func, params, n=1, n_jobs=4, out_func=None):
    """Run in parallel.

    Parameters
    ----------
    func : callable
        Function that takes parameters as inputs.
    params : iterable
        Iterable of parameters' values.
        They are passed to the `func` as ``*args``.
    n : int
        How many repetition for every combination of parameters.
    n_jobs : int
        Number of parallel jobs to run.
    out_func : callable or None
        Optional function for processing output.
    """
    pars = chain.from_iterable(repeat(params, n))
    results = Parallel(n_jobs=n_jobs)(delayed(func)(*p) for p in pars)
    if out_func is not None:
        results = out_func(results)
    return results

def get_distance_least_upper_bound(P, n_edges):
    """Get least upper bound for distances in a dataset.

    Parameters
    ----------
    P : (N, N) array_like
        Distance matrix.
    n_edges : int
        Number of edges.
    """
    dists = np.hstack((
        P[np.triu_indices_from(P, k=1)],
        P[np.tril_indices_from(P, k=-1)]
    ))
    dists.sort()
    least_upper_dist = dists[n_edges - 1]
    return least_upper_dist

@njit
def get_walk2(A, i):
    """Get length 2 walks for a given node in an adjacency matrix.

    Parameters
    ----------
    A : (N, N) array_like
        An adjacency matrix.
    i : int
        Index of a node.
    """
    return A[A[i, :].nonzero()].sum(axis=0)

@njit
def walk2_matrix(A):
    """2-walks matrix from an adjacency matrix.

    Parameters
    ----------
    A : (N, N) array_like
        An adjacency matrix.
    """
    N = A.shape[0]
    A2 = np.full((N, N), 0)
    for i in range(N):
        A2[i, :] = get_walk2(A, i)
    return A2

@njit
def _transitivity_lu(A, A2):
    N = A.shape[0]
    T = np.zeros((N,))
    for i in range(N):
        n = A[i, :].sum()
        n_paths = n*(n-1)/2*3
        n_triangle = (A[i]*A2[i]).sum()
        if n_triangle == 0:
            T[i] = np.nan if n <= 1 else 0
        else:
            T[i] = n_triangle*1.5 / n_paths
    return T

def transitivity_local_undirected(A, average=True):
    """Compute local undirected transitivity from an adjacency matrix.

    Notes
    -----
    This is not very efficient for large graphs. Better to use _igraph_.

    Parameters
    ----------
    A : (N, N)
        An adjacency matrix.
    average : bool
        Should average value be returned.
    """
    A2 = walk2_matrix(A)
    T = _transitivity_lu(A, A2)
    if average:
        T = T[~np.isnan(T)].mean()
    return T

@njit
def random_integer(p):
    c = np.cumsum(p)
    x = uniform(0, c[-1])
    return np.where(c >= x)[0][0]
