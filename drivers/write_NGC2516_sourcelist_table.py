"""
Mar 2021

Make CSV file of candidate members of the cluster. Include the
neighborhood members too.

    /Users/luke/Dropbox/proj/earhart/results/tables/NGC_2516_full_fullfaint.csv
        (--> Has everything)
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

nbhd_df, core_df, halo_df, full_df, trgt_df = get_gaia_basedata(basedata)

dfdict = {
    'full': full_df,
    'nbhd': nbhd_df,
    'core': core_df,
    'halo': halo_df,
    'trgt': trgt_df
}

for k,df in dfdict.items():
    outpath = os.path.join(OUTDIR, f'{runid}_{k}_{basedata}.csv')
    df.to_csv(outpath, index=False)
    print(f'Saved {outpath}')
