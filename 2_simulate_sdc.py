"""Run simulations for SDC model.

Parameters
----------
N_JOBS
    Number of cores used for parallelization.
RANDOM_SEED
    Seed for the random numbers generator.
SPACE
    Types of social space.
    Available values: 'uniform', 'lognormal', 'clusters_normal'.
N
    Sizes of networks,
NDIM
    Number of dimensions of simulated social spaces.
DATA_REP
    Number of independent realizations of social spaces.

SDA_PARAMS
    k
        Expected average degree.
    alpha
        Homophily level.
    directed
        Directed/undirected networks.
    p_rewire
        Probability of random rewiring.

SDA_REP
    Number of independent realizations of adjacency matrices.

SIM_PARAMS
    degseq_type
        Degree sequence type.
        One of: 'poisson', 'negbinom', 'powerlaw'.
    degseq_sort
        Should degree sequence be sorted by expected node degrees.
"""
import os
import gc
import numpy as np
import pandas as pd
from sklearn.externals.joblib import Memory
import _


# Globals
ROOT = os.path.dirname(os.path.realpath(__file__))
HERE = ROOT
DATAPATH = os.path.join(HERE, 'raw-data')

# Persistence
MEMORY = Memory(location='.cache', verbose=1)
N_JOBS = 4

# Data generation params
RANDOM_SEED = 101
SPACE = ('uniform', 'lognormal', 'clusters_normal')
N = (1000, 2000, 4000, 8000)
NDIM = (1, 2, 4, 8, 16)
CENTERS = (4,)
DATA_REP = 2

# SDA params
SDA_PARAMS = {
    'k': (30,),
    'alpha': (2, 4, 8, np.inf),
    'directed': (False,),
    'p_rewire': (.01,)
}
SDA_REP = 3
SIM_PARAMS = {
    'degseq_type': ('poisson', 'negbinom', 'powerlaw'),
    'sort': (True, False)
}


@MEMORY.cache(ignore=['n_jobs'])
def simulate_cm(space, dparams, drep, sdaparams, sdarep, simparams, n_jobs):
    return _.simulate(space, dparams, drep, sdaparams, sdarep,
                      simparams, n_jobs, simfunc=_.run_sdac)


# Run simulations
if RANDOM_SEED is not None:
    np.random.seed(RANDOM_SEED)

sim = lambda s: simulate_cm(
    space=s,
    dparams=(N, NDIM, CENTERS),
    drep=DATA_REP,
    sdaparams=SDA_PARAMS,
    sdarep=SDA_REP,
    simparams=SIM_PARAMS,
    n_jobs=N_JOBS
)

df = None       # main data frame
gdf = None      # graph data frame
for s in SPACE:
    sim(s)
gc.collect()
for s in SPACE:
    print(f"\rloading and processing '{s}' space' ...", end="")
    _df = sim(s)
    _df.drop(columns=['A', 'labels'], inplace=True)
    if df is None:
        df = _df
    else:
        df = pd.concat((df, _df), ignore_index=True)


# Save data -------------------------------------------------------------------
# Standard data get saved as feather file, so it can be easily
# shared with R for data analysis and visualization.
# Adjacency matrices data is saved as a separate pickle file.
# It will be used for graph visualizations.
os.makedirs(DATAPATH, exist_ok=True)

# Save main data as a feather file
df.to_feather(os.path.join(DATAPATH, 'sda-data-cm.feather'))
# Save graph data as a pickle file
# joblib.dump(gdf, os.path.join(DATAPATH, 'sda-graphs-cm.pkl'))
