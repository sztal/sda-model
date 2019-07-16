# pylint: disable=unused-argument
import gc
from itertools import product
from functools import reduce
import numpy as np
import pandas as pd
import igraph as ig
from scipy import stats
from sklearn.datasets import make_blobs
from sklearn.externals.joblib import Parallel, delayed
from tqdm import tqdm
from sdnet.utils import euclidean_dist, make_dist_matrix
from sdnet.sda import SDA
import ds


# Data simulation routines ----------------------------------------------------

def simulate_uniform(N, ndim, a=0, b=1, **kwds):
    return np.random.uniform(a, b, (N, ndim)), None

def simulate_normal(N, ndim, mean=0, sd=1, **kwds):
    return np.random.normal(mean, sd, (N, ndim)), None

def simulate_lognormal(N, ndim, mean=0, sd=1, **kwds):
    return np.random.lognormal(mean, sd, (N, ndim)), None

def simulate_normal_clusters(N, ndim, centers=4, center_box=(-8, 8), **kwds):
    return make_blobs(N, ndim, centers=centers, center_box=center_box, **kwds)

def simulate_lognormal_clusters(N, ndim, centers=4, center_box=(-4, 4), **kwds):
    X, labels = make_blobs(N, ndim, centers=centers, center_box=center_box, **kwds)
    return np.exp(X), labels

def simulate_space(kind, N, ndim, centers, **kwds):
    kind = kind.lower()
    if kind == 'uniform':
        return simulate_uniform(N, ndim, **kwds)
    if kind == 'normal':
        return simulate_normal(N, ndim, **kwds)
    if kind == 'lognormal':
        return simulate_lognormal(N, ndim, **kwds)
    if kind == 'clusters_normal':
        return simulate_normal_clusters(N, ndim, centers, **kwds)
    if kind == 'clusters_lognormal':
        return simulate_lognormal_clusters(N, ndim, centers, **kwds)
    raise ValueError(f"incorrect kind of data '{kind}'")

def simulate_degseq(kind, n, k):
    kind = kind.lower()
    if kind == 'poisson':
        degseq = np.random.poisson(k, (n,))
    elif kind == 'negbinom':
        degseq = np.random.negative_binomial(1, 1 / (1+k), (n,))
    elif kind == 'powerlaw':
        m = int(k/2)
        degseq = np.array(ig.Graph.Barabasi(n, m=m, directed=False).degree())
        degseq = degseq + np.random.randint(-m, m, degseq.shape)
        degseq[degseq >= n] = n-1
    elif kind == 'lognormal':
        degseq = np.random.lognormal(np.log(k) - 1/2, 1, (n,)).astype(int)
    else:
        raise ValueError(f"'{kind}' is not a correct type of degree sequence")
    if degseq.sum() % 2 != 0:
        return simulate_degseq(kind, n, k)
    return degseq


# Simulation runners ----------------------------------------------------------

def run_sda(X, labels, params, rep, simparams, dist=euclidean_dist):
    # pylint: disable=too-many-locals
    params = product(*[ tuple((k, _) for _ in v) for k, v in params.items() ])
    classify_pl = simparams.get('classify_pl', False)
    D = make_dist_matrix(X, dist, symmetric=True).astype(np.float32)
    records = []
    for p in params:
        p = dict(p)
        sda = SDA.from_dist_matrix(D, **p)
        for idx in range(1, rep+1):
            A = sda.adjacency_matrix(sparse=True) \
                .astype(bool) \
                .tolil()
            degseq = A.toarray().sum(axis=1).astype(np.uint16)
            G = ds.graph_from_sparse(A)
            dct = {
                'sid': idx,
                **p,
                'b': sda.b,
                'k_avg': degseq.mean(),
                'A': A if idx == 1 else None,
                'labels': labels.astype(np.int8) if labels is not None else None,
                'degseq': degseq,
                'deg_skew': stats.skew(degseq),
                'deg_kurt': stats.kurtosis(degseq),
                'pl_type': ds.estimate_tail_exponent(degseq, n=2, classify=True) \
                    if classify_pl else None,
                'isolates': (degseq == 0).sum(),
                'clustering': G.transitivity_undirected(),
                'assortativity': G.assortativity_degree(),
                'average_path_length': G.average_path_length()
            }
            dct['degseq'] = degseq_to_text(degseq)
            records.append(dct)

    df = pd.DataFrame.from_dict(records)
    df = df.loc[:, [
        'sid', 'k', 'alpha', 'b', 'p_rewire',
        'A', 'k_avg', 'clustering', 'assortativity', 'average_path_length',
        'isolates', 'labels', 'directed',
        'degseq', 'deg_skew', 'deg_kurt', 'pl_type'
    ]]
    return df

def run_sdac(X, labels, params, rep, simparams, dist=euclidean_dist):
    # pylint: disable=too-many-locals
    params = list(product(*[ tuple((k, _) for _ in v) for k, v in params.items() ]))
    conf_model_params = list(product(simparams['degseq_type'], simparams['sort']))
    D = make_dist_matrix(X, dist, symmetric=True).astype(np.float32)
    records = []
    for p in params:
        p = dict(p)
        sda = SDA.from_dist_matrix(D, **p)
        for degseq_type, sort in conf_model_params:
            _degseq = simulate_degseq(
                kind=degseq_type,
                n=sda.N,
                k=sda.k
            )
            sda.set_degseq(_degseq, sort=sort)
            for idx in range(1, rep+1):
                try:
                    A = sda.conf_model(simplify=True, sparse=True) \
                        .astype(bool) \
                        .tolil()
                # pylint: disable=broad-except
                except Exception:
                    continue
                degseq = A.toarray().sum(axis=1).astype(np.uint16)
                G = ds.graph_from_sparse(A)
                dct = {
                    'sid': idx,
                    **p,
                    'b': sda.b,
                    'k_avg': degseq.mean(),
                    'degseq_type': degseq_type,
                    'degseq_sort': sort,
                    'A': A if idx == 1 else None,
                    'labels': labels.astype(np.int8) if labels is not None else None,
                    'degseq': degseq,
                    'deg_skew': stats.skew(degseq),
                    'deg_kurt':  stats.kurtosis(degseq),
                    'isolates': (degseq == 0).sum(),
                    'clustering': G.transitivity_undirected(),
                    'assortativity': G.assortativity_degree(),
                    'average_path_length': G.average_path_length()
                }
                dct['degseq'] = degseq_to_text(degseq)
                records.append(dct)

    df = pd.DataFrame.from_dict(records)
    df = df.loc[:, [
        'sid', 'k', 'alpha', 'b', 'p_rewire',
        'A', 'k_avg', 'clustering', 'assortativity', 'average_path_length',
        'degseq_type', 'degseq_sort',
        'isolates', 'labels', 'directed',
        'degseq', 'deg_skew', 'deg_kurt',
    ]]
    return df

def simulate(space, dparams, drep, sdaparams, sdarep, simparams,
             n_jobs=4, simfunc=run_sda, **kwds):
    dpars = list(product(range(1, drep+1), product(*dparams)))

    def _func(idx, dpar, sdaparams, sdarep):
        gc.collect()
        X, labels = simulate_space(space, *dpar)
        df = simfunc(X, labels, sdaparams, sdarep, simparams, **kwds)
        df.insert(0, 'did', idx)
        df.insert(2, 'centers', dpar[2])
        df.insert(2, 'm', dpar[1])
        df.insert(2, 'N', dpar[0])
        df.insert(2, 'space', space)
        # Remove unneseccary network objects
        df.loc[df['did'] != 1, 'A'] = None
        return df

    results = Parallel(n_jobs=n_jobs)(
        delayed(_func)(idx, dpar, sdaparams, sdarep) for idx, dpar in tqdm(dpars)
    )
    df = None
    for _df in results:
        if _df is None:
            continue
        if df is None:
            df = _df
        else:
            df = pd.concat((df, _df), ignore_index=True)
    return df


# Postprocessing functions ----------------------------------------------------

def am_to_text(A):
    if A is None:
        return None
    return '|'.join(map('-'.join, A.toarray().astype(str)))

def degseq_to_text(degseq):
    if degseq is None:
        return None
    return '|'.join(degseq.astype(str))
