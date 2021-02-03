"""
Feb 2021

Make the following CSV files of source lists in the neighborhood,
core, and halo.

    /Users/luke/Dropbox/proj/earhart/results/tables/NGC_2516_nbhd_fullfaint.csv
    /Users/luke/Dropbox/proj/earhart/results/tables/NGC_2516_core_fullfaint.csv
    /Users/luke/Dropbox/proj/earhart/results/tables/NGC_2516_halo_fullfaint.csv
    /Users/luke/Dropbox/proj/earhart/results/tables/NGC_2516_trgt_fullfaint.csv
"""
import os, corner, pickle
import numpy as np, matplotlib.pyplot as plt, pandas as pd

from earhart.helpers import (
    get_gaia_basedata
)
from earhart.paths import RESULTSDIR

OUTDIR = os.path.join(RESULTSDIR, 'tables')

runid = "NGC_2516"
basedata = 'fullfaint'

nbhd_df, cg18_df, kc19_df, trgt_df = get_gaia_basedata(basedata)

dfdict = {
    'nbhd': nbhd_df,
    'core': cg18_df,
    'halo': kc19_df,
    'trgt': trgt_df
}

for k,df in dfdict.items():
    outpath = os.path.join(OUTDIR, f'{runid}_{k}_{basedata}.csv')
    df.to_csv(outpath, index=False)
    print(f'Saved {outpath}')
