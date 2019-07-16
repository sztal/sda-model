"""Data analysis and processing routines."""
import warnings
import numpy as np
import igraph as ig
from tailest.estimation import hill_estimator, moments_estimator
from tailest.estimation import kernel_type_estimator
from tailest.utils import get_eps_stop, add_uniform_noise

def graph_from_sparse(X, directed=False, simplify=True):
    src, tgt = X.nonzero()
    el = list(zip(src.tolist(), tgt.tolist()))
    G = ig.Graph(el)
    if not directed:
        G = G.as_undirected()
    if simplify:
        G = G.simplify()
    return G

def get_lcc(G):
    return G.decompose()[np.argmax(G.components().sizes())]


def estimate_tail_exponent(X, n=5, classify=True):
    X = np.sort(X)[::-1]
    eps_stop = get_eps_stop(X, sort=False)

    Y = np.zeros((n, 3), dtype=float)

    def estimate(X):
        X = X.copy()
        X = add_uniform_noise(X, p=1)
        X = np.sort(X)[::-1]
        xis = []
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            hill = hill_estimator(X, eps_stop=eps_stop)
            xis.append(hill[3])
            moments = moments_estimator(X, eps_stop=eps_stop)
            xis.append(moments[3])
            kernel = kernel_type_estimator(X, eps_stop=eps_stop)
            xis.append(kernel[3])
        xis = np.array(xis)
        return xis

    for i in range(n):
        Y[i] = estimate(X)
    if classify:
        return classify_tail_exponent(Y)
    return Y

def classify_tail_exponent(Y):
    Y = Y.mean(axis=0)
    if (Y < 0).any():
        return 'NPL'
    if (Y <= 1/4).any():
        return 'HPL'
    if (Y > 1/2).all():
        return 'DSM'
    return 'PL'
