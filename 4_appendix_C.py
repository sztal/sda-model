from itertools import repeat, chain
import numpy as np
import pandas as pd
from tqdm import tqdm
import _
import ds


N = (1000, 2000, 4000, 8000)
K = 30
rep = 50

df = pd.DataFrame({
    'N': [ x for x in sorted(chain.from_iterable(repeat(N, rep))) ]
})

def run_pl(n):
    degseq = _.simulate_degseq('powerlaw', n=n, k=K)
    return ds.estimate_tail_exponent(degseq, n=2, classify=True)

np.random.seed(111)
df['pl_type'] = [ run_pl(n) for n in tqdm(df['N']) ]
gdf = df.groupby(['N'])['pl_type'].value_counts()
print(gdf)
