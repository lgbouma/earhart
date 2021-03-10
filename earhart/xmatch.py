"""
Crossmatching utilities.

Contents:
    xmatch_dataframes_using_radec
"""
import pandas as pd, numpy as np
from numpy import array as nparr
from astropy.io import fits
from astropy.table import Table
from astropy import units as u, constants as const
from astropy.coordinates import SkyCoord

def xmatch_dataframes_using_radec(df0, df1, rakey0, deckey0, rakey1, deckey1,
                                  RADIUS=0.1, raise_size_error=1):
    """
    If one is longer than the other, make df1 the short one.

    Args:
        df0 and df1: DataFrames with RA, dec keys specified.
        RADIUS: float, arcseconds.

    Returns: merged dataframe.
    """

    N_0 = len(df0)
    N_1 = len(df1)
    if N_1 > 5*N_0:
        raise ValueError(
            'xmatch_dataframes_using_radec is much faster if df1 is shorter '
            'than df0.'
        )
    if N_0 > 1e4 or N_1 > 1e4:
        if raise_size_error:
            raise ValueError(
                """
                xmatch_dataframes_using_radec uses a bad one-by-one algorithm to do
                the crossmatch. this is extremely inefficient. for >1e4 sources,
                it'll take >10 minutes. consider doing something smarter.
                """
            )

    c0 = SkyCoord(ra=nparr(df0[rakey0])*u.deg, dec=nparr(df0[deckey0])*u.deg)
    c1 = SkyCoord(ra=nparr(df1[rakey1])*u.deg, dec=nparr(df1[deckey1])*u.deg)

    cutoff_radius = RADIUS*u.arcsec
    has_matchs, match_idxs, match_rows, match_seps = [], [], [], []
    for ix, _c in enumerate(c1):
        if ix % int(N_1/10) == 0:
            print(f'{ix}/{len(c1)}...')
        seps = _c.separation(c0)
        if min(seps.to(u.arcsec)) < cutoff_radius:
            has_matchs.append(True)
            match_idx = np.argmin(seps)
            match_idxs.append(match_idx)
            match_rows.append(df0.iloc[match_idx])
            match_seps.append(seps[match_idx].to(u.arcsec).value)
        else:
            has_matchs.append(False)
            match_seps.append(np.nan)

    has_matchs = nparr(has_matchs)
    match_seps = nparr(match_seps)

    left_df = df1[has_matchs]

    right_df = pd.DataFrame(match_rows)

    mdf = pd.concat((left_df.reset_index(drop=True),
                     right_df.reset_index(drop=True)), axis=1)
    mdf['sep_arcsec'] = mdf['sep_arcsec'] = match_seps[~pd.isnull(match_seps)]

    print(f'Got {len(mdf)} matches with separation < {RADIUS} arcsec.')

    return mdf
