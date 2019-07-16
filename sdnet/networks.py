"""Simple network models and related utilities."""
from random import uniform as _uniform, choice as _choice
import numpy as np
from numpy.random import random, uniform
from numba import njit


@njit
def _rn_undirected_nb(X, p):
    for i in range(X.shape[0]):
        for j in range(i):
            if random() <= p:
                X[i, j] = X[j, i] = 1
    return X

def random_network(N, p=None, k=None, directed=False):
    """Generate a random network.

    Parameters
    ----------
    N : int
        Number of nodes.
    p : float
        Edge formation probability.
        Should be set to ``None`` if `k` is used.
    k : float
        Average node degree.
        Should be set to ``None`` if `p` is used.
    directed : bool
        Should network be directed.

    Notes
    -----
    `p` or `k` (but not both) must be not ``None``.

    Returns
    -------
    (N, N) array_like
        Adjacency matrix of a graph.
    """
    if p is None and k is None:
        raise TypeError("Either 'p' or 'k' must be used")
    if p is not None and k is not None:
        raise TypeError("'p' and 'k' can not be used at the same time")
    if k is not None:
        if k > N-1:
            raise ValueError(f"average degree of {k:.4} can not be attained with {N} nodes")
        p = k / (N-1)
    if directed:
        X = np.where(uniform(0, 1, (N, N)) <= p, 1, 0)
        np.fill_diagonal(X, 0)
    else:
        X = np.zeros((N, N), dtype=int)
        X = _rn_undirected_nb(X, p)
    return X

@njit
def _am_undirected_nb(P, A):
    for i in range(A.shape[0]):
        for j in range(i):
            if random() <= P[i, j]:
                A[i, j] = A[j, i] = 1
    return A

def make_adjacency_matrix(P, directed=False):
    """Generate adjacency matrix from edge formation probabilities.

    Parameters
    ----------
    P : (N, N) array_like
        Edge formation probability matrix.
    directed : bool
        Should network be directed.
    """
    # pylint: disable=no-member
    if directed:
        A = np.where(uniform(0, 1, P.shape) <= P, 1, 0)
        A = A.astype(int)
        np.fill_diagonal(A, 0)
    else:
        A = np.zeros_like(P, dtype=int)
        A = _am_undirected_nb(P, A)
    return A

def get_edgelist(A, directed=False):
    """Get ordered edgelist from an adjacency matrix.

    Parameters
    ----------
    A : (N, N) array_like
        An adjacency matrix.
    directed : bool
        Is the graph directed.
    """
    E = np.argwhere(A)
    E = E[E.sum(axis=1).argsort()]
    sum_idx = E.sum(axis=1)
    max_idx = E.max(axis=1)
    max1c_idx = (E[:, 0] > E[:, 1])
    E = E[np.lexsort((max1c_idx, sum_idx, max_idx))]
    if directed:
        dual = np.full((E.shape[0], 1), -1)
    else:
        dual = np.arange(E.shape[0])
        dual[::2] += 1
        dual[1::2] -= 1
        dual = dual.reshape(E.shape[0], 1)
    E = np.hstack((E, dual))
    return E

def rewire_edges(A, p=0.01, directed=False, copy=False):
    """Randomly rewire edges in an adjacency matrix with given probability.

    Parameters
    ----------
    A : (N, N) array_like
        An adjacency matrix.
    p : float
        Rewiring probability.
    directed : bool
        Is the graph directed.
    copy : bool
        Should copy of the adjacency array be returned.
    """
    if copy:
        A = A.copy()
    E = get_edgelist(A, directed=directed)
    loop = range(0, E.shape[0]) if directed else range(0, E.shape[0], 2)
    for u in loop:
        rand = _uniform(0, 1)
        if rand <= p:
            i, j = E[u, :2]
            if not directed and rand <= p/2:
                new_i = j
            else:
                new_i = i
            idx = np.nonzero(np.where(A[new_i, :] == 0, 1, 0))[0]
            idx = idx[idx != new_i]
            if idx.size == 0:
                continue
            new_j = _choice(idx)
            A[i, j] = 0
            A[new_i, new_j] = 1
            if not directed:
                A[j, i] = 0
                A[new_j, new_i] = 1
    return A
