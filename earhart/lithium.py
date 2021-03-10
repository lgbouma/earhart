"""
Contents:
    get_Randich18_NGC2516
    get_GalahDR3_lithium
    get_GalahDR3_li_EWs
"""
import os
import numpy as np, pandas as pd
from numpy import array as nparr
from astropy.io import fits
from astropy.table import Table
from astropy import units as u, constants as const
from astropy.coordinates import SkyCoord

from earhart.paths import DATADIR, RESULTSDIR

def _get_lithium_EW_df(gaiaeso, galahdr3):

    datapath = os.path.join(DATADIR, 'lithium',
                            'randich_fullfaintkinematic_xmatch_20210310.csv')
    if not os.path.exists(datapath):
        from earhart.lithium import _make_Randich18_xmatch
        _make_Randich18_xmatch(datapath, vs_rotators=0)
    gaiaeso_df = pd.read_csv(datapath)
    gaiaeso_li_col = 'EWLi'

    from earhart.lithium import get_GalahDR3_li_EWs
    galah_df = get_GalahDR3_li_EWs()
    # the gaussian fit, numerically integrated
    galah_li_col = 'Fitted_Li_EW_mA'

    s_gaiaeso_df = gaiaeso_df[[gaiaeso_li_col, 'source_id']]
    s_gaiaeso_df = s_gaiaeso_df.rename({gaiaeso_li_col: 'Li_EW_mA'}, axis='columns')
    s_galah_df = galah_df[[galah_li_col, 'source_id']]
    s_galah_df = s_galah_df.rename({galah_li_col: 'Li_EW_mA'}, axis='columns')

    if gaiaeso and galahdr3:
        df = pd.concat((s_gaiaeso_df, s_galah_df))
    if gaiaeso and not galahdr3:
        df = s_gaiaeso_df
    if not gaiaeso and galahdr3:
        df = s_galah_df

    return df

def get_GalahDR3_li_EWs(verbose=1):
    """
    Made by drivers.measure_galah_dr3_Li_EWs.py

    Target spectra are from the crossmatch of "fullfaint" (DR2) with the GALAH
    target list.
    """

    if verbose:
        print("WRN! These GALAH DR3 EWs are janky for <50mA b/c of S/N.")
    galah_li_path = os.path.join(
        DATADIR, 'lithium', 'galahdr3_fullfaintkinematic_xmatch_20210310.csv'
    )
    li_df = pd.read_csv(galah_li_path)

    return li_df


def get_GalahDR3_lithium(verbose=1, defaultflags=0):
    """
    Get astropy table of stellar parameter flag == 0 stars from GALAH DR3, with
    lithium detections or upper limits.
    """

    # downloaded via wget, per
    # https://github.com/svenbuder/GALAH_DR3/blob/master/tutorials/tutorial1_dr3_main_catalog_overview.ipynb
    dr3path = os.path.join(DATADIR, 'lithium', 'GALAH_DR3_main_allstar_v1.fits')

    dr3_tab = Table.read(dr3path)

    if defaultflags:
        # stellar parameter flag
        qual = np.array(dr3_tab['flag_sp'])
        binary_repr_vec = np.vectorize(np.binary_repr)
        qual_binary = binary_repr_vec(qual, width=11)

        # 11 total flag bits:
        # 2^0 = 1: Gaia DR2 RUWE > 1.4
        # 2^1 = 2: unreliable broadening
        # 2^2 = 4: low S/N
        # 2^3 = 8: reduction issues (wvlen soln, t-SNE reduction issues,
        #    weird fluxes, spikes, etc.
        # 4   = 16: t-SNE projected emission features
        # 5   = 32: t-SNE projected binaries
        # 6   = 64: on binary sequence / PMS sequence
        # 7   = 128: S/N dependent high SME chi2 (bad fit)
        # 8   = 256: problems with Fe: line flux not between 0.03 and 1.00,
        #      [Fe/H] unreliable, or blending suspects and SME didnt finish
        # 9   = 512: SME did not finish. Either a) No convergence -> nonfinite stellar
        #      parameters. Or b) Gaussian RV fit failed.
        # 2^10 = 1024: MARCS grid limit reached or outside reasonable
        #       parameter range.

        # Need reliable broadening, S/N, etc.
        # 2^1 = 2: unreliable broadening
        # 2^2 = 4: low S/N
        # 2^3 = 8: reduction issues (wvlen soln, t-SNE reduction issues,
        #    weird fluxes, spikes, etc.
        # 8   = 256: problems with Fe: line flux not between 0.03 and 1.00,
        #      [Fe/H] unreliable, or blending suspects and SME didnt finish
        # 9   = 512: SME did not finish. Either a) No convergence -> nonfinite stellar
        #      parameters. Or b) Gaussian RV fit failed.
        # 2^10 = 1024: MARCS grid limit reached or outside reasonable
        #       parameter range.
        badbits = [1,2,3]

        sel = np.isfinite(dr3_tab['source_id'])
        for bb in badbits:
            # zero -> one-based count here to convert bitwise flags to
            # python flags
            sel &= ~(np.array([q[bb-1] for q in qual_binary]).astype(bool))

    else:
        sel = np.isfinite(dr3_tab['source_id'])

    if verbose:
        print(f'All stars in GALAH DR3: {len(dr3_tab)}')
        if defaultflags:
            print(f'All stars in GALAH DR3 with bitflags {repr(badbits)} not set: {len(dr3_tab[sel])}')

    return dr3_tab[sel]


def get_Randich18_NGC2516():

    hl = fits.open(
        os.path.join(DATADIR, 'lithium',
                     'Randich_2018_NGC2516_all796_entries_vizier.fits')
    )

    t_df = Table(hl[1].data).to_pandas()

    return t_df


def _make_Randich18_xmatch(datapath, vs_rotators=1, RADIUS=0.5):
    """
    For every Randich+18 Gaia-ESO star with a spectrum, look for a rotator
    match (either the "gold" or "autorot" samples) within RADIUS arcseconds.
    If you find it, pull its data. If there are multiple, take the closest.
    """

    rdf = get_Randich18_NGC2516()

    if vs_rotators:
        raise DeprecationWarning
        rotdir = os.path.join(DATADIR, 'rotation')
        rot_df = pd.read_csv(
            os.path.join(rotdir, 'ngc2516_rotation_periods.csv')
        )
        comp_df = rot_df[rot_df.Tags == 'gold']
        print('Comparing vs the "gold" NGC2516 rotators sample (core + halo)...')
    else:

        from earhart.helpers import _get_fullfaint_dataframes
        nbhd_df, core_df, halo_df, full_df, target_df = _get_fullfaint_dataframes()
        comp_df = full_df
        print(f'Comparing vs the {len(comp_df)} "fullfaint" kinematic NGC2516 rotators sample (core + halo)...')

    c_comp = SkyCoord(ra=nparr(comp_df.ra)*u.deg, dec=nparr(comp_df.dec)*u.deg)
    c_r18 = SkyCoord(ra=nparr(rdf._RA)*u.deg, dec=nparr(rdf._DE)*u.deg)

    cutoff_radius = RADIUS*u.arcsec
    has_matchs, match_idxs, match_rows = [], [], []
    for ix, _c in enumerate(c_r18):
        if ix % 100 == 0:
            print(f'{ix}/{len(c_r18)}')
        seps = _c.separation(c_comp)
        if min(seps.to(u.arcsec)) < cutoff_radius:
            has_matchs.append(True)
            match_idx = np.argmin(seps)
            match_idxs.append(match_idx)
            match_rows.append(comp_df.iloc[match_idx])
        else:
            has_matchs.append(False)

    has_matchs = nparr(has_matchs)

    left_df = rdf[has_matchs]

    right_df = pd.DataFrame(match_rows)

    mdf = pd.concat((left_df.reset_index(drop=True),
                     right_df.reset_index(drop=True)), axis=1)

    if vs_rotators:
        print(f'Got {len(mdf)} gold rot matches from {len(rdf)} Randich+18 shots.')
    else:
        print(f'Got {len(mdf)} fullfaint kinematic matches from {len(rdf)} Randich+18 shots.')

    # "Comparing the Gaia color and GES Teff, 15 of these (all with
    # Bp-Rp0 $>$ 2.0) are spurious matches, which we remove."
    if not vs_rotators:

        from earhart.priors import AVG_EBpmRp
        assert abs(AVG_EBpmRp - 0.1343) < 1e-4 # used by KC19

        badmatch = (
            ((mdf['phot_bp_mean_mag'] - mdf['phot_rp_mean_mag'] - AVG_EBpmRp)>2.0)
            &
            (mdf['Teff'] > 4300)
        )
        mdf = mdf[~badmatch]
        print(f'Got {len(mdf)} fullfaint kinematic matches from {len(rdf)} Randich+18 shots after cleaning "BADMATCHES".')
        print(f'Got {len(mdf[mdf.subcluster=="core"])} in core.')
        print(f'Got {len(mdf[mdf.subcluster=="halo"])} in halo.')

    mdf.to_csv(datapath, index=False)
