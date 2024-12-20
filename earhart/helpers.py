"""
Really, mostly data getters.

get_toi1937_lightcurve
get_groundphot
get_autorotation_dataframe
get_gaia_basedata
    _get_nbhd_dataframes
    _get_fullfaint_dataframes
    _get_fullfaint_edr3_dataframes
    _get_denis_fullfaint_edr3_dataframes
    _get_extinction_dataframes
    _get_median_ngc2516_core_params
get_denis_xmatch
append_phot_binary_column
PleaidesQuadProtModel
"""
import os, collections, pickle
import numpy as np, pandas as pd
from glob import glob
from copy import deepcopy

from numpy import array as nparr

from astropy.io import fits
from astropy import units as u
from astropy.table import Table
from astroquery.vizier import Vizier
from astroquery.xmatch import XMatch

import cdips.utils.lcutils as lcu
import cdips.lcproc.detrend as dtr
import cdips.lcproc.mask_orbit_edges as moe

from cdips.utils.catalogs import (
    get_cdips_catalog, get_tic_star_information
)
from cdips.utils.gaiaqueries import (
    query_neighborhood, given_source_ids_get_gaia_data,
    given_dr2_sourceids_get_edr3_xmatch
)

from earhart.paths import PHOTDIR, RESULTSDIR, DATADIR

def get_toi1937_lightcurve():
    """
    Create the stitched CDIPS FFI light curve for TOI 1937. (Starting from the
    raw light curves, and the PCA eigenvectors previously made for this
    sector). Note: the main execution of this PCA detrending happens on
    phtess2.

    A few notes:
        * 3 eigenvectors were used, plus the background light BGV timeseries.
        * a +/-12 hour orbit edge mask was used (to avoid what looked like
        scattered light)
        * the output can be checked at
        /results/quicklook_lcs/5489726768531119616_allvar_report.pdf
    """

    picklepath = os.path.join(
        PHOTDIR, 'toi1937_merged_stitched_s7s9_lc_20201130.pkl'
    )

    if not os.path.exists(picklepath):

        # Use the CDIPS IRM2 light curves as starting base.
        # 5489726768531119616_s09_llc.fits
        lcpaths = glob(os.path.join(PHOTDIR, '*_s??_llc.fits'))
        assert len(lcpaths) == 2

        infodicts = [
            {'SECTOR': 7, 'CAMERA': 3, 'CCD': 4, 'PROJID': 1527},
            {'SECTOR': 9, 'CAMERA': 3, 'CCD': 3, 'PROJID': 1558},
        ]

        ##########################################
        # next ~45 lines pinched from cdips.drivers.do_allvariable_report_making
        ##########################################
        #
        # detrend systematics. each light curve yields tuples of:
        #   primaryhdr, data, ap, dtrvecs, eigenvecs, smooth_eigenvecs
        #
        dtr_infos = []
        for lcpath, infodict in zip(lcpaths, infodicts):
            dtr_info = dtr.detrend_systematics(
                lcpath, infodict=infodict, max_n_comp=3
            )
            dtr_infos.append(dtr_info)

        #
        # stitch all available light curves
        #
        ap = dtr_infos[0][2]
        timelist = [d[1]['TMID_BJD'] for d in dtr_infos]
        maglist = [d[1][f'PCA{ap}'] for d in dtr_infos]
        magerrlist = [d[1][f'IRE{ap}'] for d in dtr_infos]

        extravecdict = {}
        extravecdict[f'IRM{ap}'] = [d[1][f'IRM{ap}'] for d in dtr_infos]
        for i in range(0,7):
            extravecdict[f'CBV{i}'] = [d[3][i, :] for d in dtr_infos]

        time, flux, fluxerr, vec_dict = lcu.stitch_light_curves(
            timelist, maglist, magerrlist, extravecdict
        )

        #
        # mask orbit edges
        #
        s_time, s_flux, inds = moe.mask_orbit_start_and_end(
            time, flux, raise_expectation_error=False, orbitgap=0.7,
            orbitpadding=12/24,
            return_inds=True
        )
        s_fluxerr = fluxerr[inds]

        #
        # save output
        #

        ap = dtr_infos[0][2]
        lcdict = {
            'source_id': np.int64(5489726768531119616),
            'E_BpmRp': 0.1343,
            'ap': ap,
            'TMID_BJD': time,
            f'IRM{ap}': vec_dict[f'IRM{ap}'],
            f'PCA{ap}': flux,
            f'IRE{ap}': fluxerr,
            'STIME': s_time.astype(np.float64),
            f'SPCA{ap}': s_flux.astype(np.float64),
            f'SPCAE{ap}': s_fluxerr.astype(np.float64),
            'dtr_infos': dtr_infos,
            'vec_dict': vec_dict,
            'tess_texp': np.nanmedian(np.diff(s_time))
        }

        with open(picklepath , 'wb') as f:
            pickle.dump(lcdict, f)

        #
        # verify output
        #
        from cdips.plotting.allvar_report import make_allvar_report
        plotdir = os.path.join(RESULTSDIR, 'quicklook_lcs')
        outd = make_allvar_report(lcdict, plotdir)

    with open(picklepath, 'rb') as f:
        print(f'Found {picklepath}: loading it!')
        lcdict = pickle.load(f)

    return (
        lcdict['STIME'].astype(np.float64) - 2457000,
        lcdict['SPCA2'].astype(np.float64),
        lcdict['SPCAE2'].astype(np.float64),
        lcdict['tess_texp']
    )


def get_groundphot(datestr=None):

    lcglob = os.path.join(PHOTDIR, 'collected',
                          f'*{datestr}*.txt')
    lcpath = glob(lcglob)
    assert len(lcpath) == 1
    lcpath = lcpath[0]

    if 'epdlc' in lcpath:
        # LCOGT reduced by Joel Hartman format.

        colnames = [
            "frameid", "time_bjd_UTC_minus_2400000", "raw_mag_ap1",
            "raw_mag_err_ap1", "quality_ap1", "raw_mag_ap2", "raw_mag_err_ap2",
            "quality_ap2", "raw_mag_ap3", "raw_mag_err_ap3", "quality_ap3",
            "fit_mag_ap1", "fit_mag_ap2", "fit_mag_ap3", "epd_mag_ap1",
            "epd_mag_ap2", "epd_mag_ap3", "x_px", "y_px", "bkgd",
            "bkgd_deviation", "S", "D", "K", "hour_angle", "zenith_distance",
            "time_JD_UTC"
        ]

        df = pd.read_csv(lcpath, delim_whitespace=True, names=colnames,
                         comment='#')

        # TT = TAI + 32.184 = UTC + (number of leap seconds) + 32.184
        # TDB ~= TT
        # for these data the leap second list indicates 37 is the correct
        # number: https://www.ietf.org/timezones/data/leap-seconds.list

        t_offset = (37 + 32.184)*u.second

        x_obs_bjd_utc = np.array(df["time_bjd_UTC_minus_2400000"]) + 2400000
        # return times in BJD_TDB
        x_obs = x_obs_bjd_utc + float(t_offset.to(u.day).value)

        y_obs, y_err = (
            lcu._given_mag_get_flux(df['fit_mag_ap1'], df["raw_mag_err_ap1"])
        )

        t_exp = np.nanmedian(np.diff(x_obs))

    elif 'El_Sauce' in lcpath:
        # Phil Evan's El Sauce reduction format.
        raise NotImplementedError

    return x_obs, y_obs, y_err, t_exp


def get_gaia_basedata(basedata):

    if basedata == 'extinctioncorrected':
        raise NotImplementedError('need to implement extinction')
        nbhd_df, core_df, halo_df, full_df, target_df = _get_extinction_dataframes()
    elif basedata == 'fullfaint':
        nbhd_df, core_df, halo_df, full_df, target_df = _get_fullfaint_dataframes()
    elif basedata == 'fullfaint_edr3':
        nbhd_df, core_df, halo_df, full_df, target_df = _get_fullfaint_edr3_dataframes()
    elif basedata == 'bright':
        nbhd_df, core_df, halo_df, full_df, target_df = _get_nbhd_dataframes()
    else:
        raise NotImplementedError

    full_df = append_phot_binary_column(full_df)

    return nbhd_df, core_df, halo_df, full_df, target_df


def _get_nbhd_dataframes():
    """
    WARNING!: this "bright" subset is a crossmatch between the full NGC 2516
    target list (CG18+KC19+M21), and the CDIPS target catalog (G_Rp<16; v0.4).
    However, since the CDIPS targets didn't incorporate M21, it's not as direct
    of a match as desired. This is fine for understanding the auto-detection of
    rotation periods. But for overall cluster rotation period completeness,
    it's not.

    The "neighborhood" was selected via

        bounds = { 'parallax_lower': 1.5, 'parallax_upper': 4.0, 'ra_lower': 108,
        'ra_upper': 132, 'dec_lower': -76, 'dec_upper': -45 }

        nbhd_df = query_neighborhood(bounds, groupname, n_max=6000,
                                     overwrite=False, manual_gmag_limit=17)

    This procedure yields:
        Got 7052 neighbors with Rp<16
        Got 893 in core from CDIPS target catalog
        Got 1345 in corona from CDIPS target catalog
    """

    df = get_cdips_catalog(ver=0.4)

    nbhd_df, core_df, halo_df, full_df, target_df = _get_fullfaint_dataframes()

    #
    # do the "bright" selection by a crossmatch between the full target list
    # and the CDIPS catalog. so implicitly, it's a CDIPS target star catalog
    # match.  this misses some Meingast stars, b/c they weren't in the CDIPS
    # v0.4 target list. but this 
    #
    cdips_df = df['source_id']
    mdf = full_df.merge(cdips_df, on='source_id', how='inner')

    nbhd_df = nbhd_df[nbhd_df.phot_rp_mean_mag < 16]

    core_df = mdf[mdf.subcluster == 'core']
    halo_df = mdf[mdf.subcluster == 'halo']

    print(42*'.')
    print('"Bright" sample:')
    print(f'...Got {len(nbhd_df)} neighbors with Rp<16')
    print(f'...Got {len(core_df)} in core from CDIPS target catalog')
    print(f'...Got {len(halo_df)} in corona from CDIPS target catalog')
    print(42*'.')

    return nbhd_df, core_df, halo_df, full_df, target_df


def _get_fullfaint_dataframes():
    """
    Return: nbhd_df, core_df, halo_df, full_df, target_df
    (for NGC 2516, "full faint" sample -- i.e., as faint as possible.)

    The "core" is all available Cantat-Gaudin 2018 members, with no magnitude
    cutoff.

    The "halo" is the full Kounkel & Covey 2019 + Meingast 2021 member set,
    provided that the source is not in the core. (i.e., KC19 and M21 get no
    points for getting the "core" targets correct).

    The "neighborhood" was selected via

        bounds = { 'parallax_lower': 1.5, 'parallax_upper': 4.0, 'ra_lower': 108,
        'ra_upper': 132, 'dec_lower': -76, 'dec_upper': -45 }

        nbhd_df = query_neighborhood(bounds, groupname, n_max=14000,
                                     overwrite=False, manual_gmag_limit=19)

    This procedure yields:

        Got 1106 in fullfaint CG18
        Got 3003 in fullfaint KC19
        Got 1860 in fullfaint M21
        Got 1912 in fullfaint KC19 after removing core matches
        Got 1096 in fullfaint M21 after removing core matches
        Got 280 in fullfaint M21 after removing KC19 matches

        Got 13834 neighbors
        Got 1106 in core
        Got 2192 in corona
        Got 1091 KC19 / CG18 overlaps
        Got 764 M21 / CG18 overlaps
        Got 3298 unique sources in the cluster.
    """

    # get the full CG18 NGC 2516 memberships, downloaded from Vizier
    cg18path = os.path.join(DATADIR, 'gaia',
                            'CantatGaudin2018_vizier_only_NGC2516.fits')
    hdul = fits.open(cg18path)
    cg18_tab = Table(hdul[1].data)
    cg18_df = cg18_tab.to_pandas()
    cg18_df['source_id'] = cg18_df['Source']

    # get the full KC19 NGC 2516 memberships, from Marina's file
    # NGC 2516 == "Theia 613" in Kounkel's approach.
    kc19path = os.path.join(DATADIR, 'gaia', 'string_table1.csv')
    kc19_df = pd.read_csv(kc19path)
    kc19_df = kc19_df[kc19_df.group_id == 613]

    # get the full M21 NGC 2516 memberships
    m21path = os.path.join(DATADIR, 'gaia', 'Meingast_2021_NGC2516_all1860members.fits')
    m21_df = Table(fits.open(m21path)[1].data).to_pandas()
    m21_df = m21_df.rename(mapper={'GaiaDR2': 'source_id'}, axis=1)

    print(f'Got {len(cg18_df)} in fullfaint CG18')
    print(f'Got {len(kc19_df)} in fullfaint KC19')
    print(f'Got {len(m21_df)} in fullfaint M21')

    kc19_cg18_overlap_df = kc19_df[(kc19_df.source_id.isin(cg18_df.source_id))]
    kc19_df = kc19_df[~(kc19_df.source_id.isin(cg18_df.source_id))]
    print(f'Got {len(kc19_df)} in fullfaint KC19 after removing core matches')

    m21_cg18_overlap_df = m21_df[(m21_df.source_id.isin(cg18_df.source_id))]
    m21_df = m21_df[~(m21_df.source_id.isin(cg18_df.source_id))]
    print(f'Got {len(m21_df)} in fullfaint M21 after removing core matches')
    m21_df = m21_df[~(m21_df.source_id.isin(kc19_df.source_id))]
    print(f'Got {len(m21_df)} in fullfaint M21 after removing KC19 matches')

    ##########

    # NGC 2516 rough
    bounds = {
        'parallax_lower': 1.5, 'parallax_upper': 4.0, 'ra_lower': 108,
        'ra_upper': 132, 'dec_lower': -76, 'dec_upper': -45
    }
    groupname = 'customngc2516_fullfaint'

    nbhd_df = query_neighborhood(bounds, groupname, n_max=14000,
                                 overwrite=False, manual_gmag_limit=19)

    # query gaia DR2 to get the fullfaint photometry
    kc19_df_0 = given_source_ids_get_gaia_data(
        np.array(kc19_df.source_id),
        'ngc2516_kc19_earhart_fullfaint', n_max=10000, overwrite=False,
        enforce_all_sourceids_viable=True
    )
    cg18_df_0 = given_source_ids_get_gaia_data(
        np.array(cg18_df.Source),
        'ngc2516_cg18_earhart_fullfaint', n_max=10000, overwrite=False,
        enforce_all_sourceids_viable=True
    )
    m21_df_0 = given_source_ids_get_gaia_data(
        np.array(m21_df.source_id),
        'ngc2516_m21_earhart_fullfaint', n_max=10000, overwrite=False,
        enforce_all_sourceids_viable=True
    )

    assert len(cg18_df) == len(cg18_df_0)
    assert len(kc19_df) == len(kc19_df_0)
    assert len(m21_df) == len(m21_df_0)

    target_df = kc19_df_0[kc19_df_0.source_id == 5489726768531119616] # TIC 2683...

    sel_nbhd = (
        (~nbhd_df.source_id.isin(kc19_df.source_id))
        &
        (~nbhd_df.source_id.isin(cg18_df.source_id))
        &
        (~nbhd_df.source_id.isin(m21_df.source_id))
    )
    orig_nbhd_df = deepcopy(nbhd_df)
    nbhd_df = nbhd_df[sel_nbhd]

    print(f'Got {len(nbhd_df)} neighbors')
    print(f'Got {len(cg18_df)} in core')
    print(f'Got {len(kc19_df)+len(m21_df)} in corona')
    print(f'Got {len(kc19_cg18_overlap_df)} KC19 / CG18 overlaps')
    print(f'Got {len(m21_cg18_overlap_df)} M21 / CG18 overlaps')

    #
    # wrap up into the full source list
    #
    cg18_df_0['subcluster'] = 'core'
    kc19_df_0['subcluster'] = 'halo'
    m21_df_0['subcluster'] = 'halo'

    core_df = cg18_df_0
    halo_df = pd.concat((kc19_df_0, m21_df_0)).reset_index()

    full_df = pd.concat((core_df, halo_df)).reset_index()
    assert len(np.unique(full_df.source_id)) == len(full_df)
    print(f'Got {len(full_df)} unique sources in the cluster.')

    full_df['in_CG18'] = full_df.source_id.isin(cg18_df.source_id)
    kc19_df = pd.read_csv(kc19path)
    kc19_df = kc19_df[kc19_df.group_id == 613]
    full_df['in_KC19'] = full_df.source_id.isin(kc19_df.source_id)
    m21_df = Table(fits.open(m21path)[1].data).to_pandas()
    m21_df = m21_df.rename(mapper={'GaiaDR2': 'source_id'}, axis=1)
    full_df['in_M21'] = full_df.source_id.isin(m21_df.source_id)

    return nbhd_df, core_df, halo_df, full_df, target_df


def _get_fullfaint_edr3_dataframes():
    """
    Return: nbhd_df, core_df, halo_df, full_df, target_df
    (for NGC 2516, "full faint" sample -- i.e., as faint as possible, but
    ***after crossmatching the GAIA DR2 targets with GAIA EDR3***. This
    crossmatch is run using the dr2_neighbourhood table from the Gaia archive,
    and then taking the closest angular separation match for cases with
    multiple matches.)

    Further notes are in "_get_fullfaint_dataframes" docstring.

    This procedure yields:

		FOR DR2:
			Got 1106 in fullfaint CG18
			Got 3003 in fullfaint KC19
			Got 1860 in fullfaint M21
			Got 1912 in fullfaint KC19 after removing core matches
			Got 1096 in fullfaint M21 after removing core matches
			Got 280 in fullfaint M21 after removing KC19 matches
			Got 13834 neighbors
			Got 1106 in core
			Got 2192 in corona
			Got 1091 KC19 / CG18 overlaps
			Got 764 M21 / CG18 overlaps

        FOR EDR3:

			Got 1106 EDR3 matches in core.
			99th pct [arcsec] 1577.8 -> 0.3
			Got 1912 EDR3 matches in KC19.
			99th pct [arcsec] 1702.8 -> 0.5
			Got 280 EDR3 matches in M21.
			99th pct [arcsec] 1426.6 -> 0.3
			Got 13843 EDR3 matches in nbhd.
			99th pct [arcsec] 1833.9 -> 3.7

			(((
				CG18/core: got 1143 matches vs 1106 source id queries.
				KC19/halo: got 2005 matches vs 1912 source id queries
				Nbhd:      got 15123 matches vs 13843 source id queries.
			)))
    """

    # get the full CG18 NGC 2516 memberships, downloaded from Vizier
    cg18path = os.path.join(DATADIR, 'gaia',
                            'CantatGaudin2018_vizier_only_NGC2516.fits')
    hdul = fits.open(cg18path)
    cg18_tab = Table(hdul[1].data)
    cg18_df = cg18_tab.to_pandas()
    cg18_df['source_id'] = cg18_df['Source']

    # get the full KC19 NGC 2516 memberships, from Marina's file
    # NGC 2516 == "Theia 613" in Kounkel's approach.
    kc19path = os.path.join(DATADIR, 'gaia', 'string_table1.csv')
    kc19_df = pd.read_csv(kc19path)
    kc19_df = kc19_df[kc19_df.group_id == 613]

    # get the full M21 NGC 2516 memberships
    m21path = os.path.join(DATADIR, 'gaia', 'Meingast_2021_NGC2516_all1860members.fits')
    m21_df = Table(fits.open(m21path)[1].data).to_pandas()
    m21_df = m21_df.rename(mapper={'GaiaDR2': 'source_id'}, axis=1)

    print(42*'='+'\nFOR DR2:')
    print(f'Got {len(cg18_df)} in fullfaint CG18')
    print(f'Got {len(kc19_df)} in fullfaint KC19')
    print(f'Got {len(m21_df)} in fullfaint M21')

    kc19_cg18_overlap_df = kc19_df[(kc19_df.source_id.isin(cg18_df.source_id))]
    kc19_df = kc19_df[~(kc19_df.source_id.isin(cg18_df.source_id))]
    print(f'Got {len(kc19_df)} in fullfaint KC19 after removing core matches')

    m21_cg18_overlap_df = m21_df[(m21_df.source_id.isin(cg18_df.source_id))]
    m21_df = m21_df[~(m21_df.source_id.isin(cg18_df.source_id))]
    print(f'Got {len(m21_df)} in fullfaint M21 after removing core matches')
    m21_df = m21_df[~(m21_df.source_id.isin(kc19_df.source_id))]
    print(f'Got {len(m21_df)} in fullfaint M21 after removing KC19 matches')

    ##########

    # NGC 2516 rough
    bounds = {
        'parallax_lower': 1.5, 'parallax_upper': 4.0, 'ra_lower': 108,
        'ra_upper': 132, 'dec_lower': -76, 'dec_upper': -45
    }
    groupname = 'customngc2516_fullfaint'

    nbhd_df = query_neighborhood(bounds, groupname, n_max=14000,
                                 overwrite=False, manual_gmag_limit=19)

    sel_nbhd = (
        (~nbhd_df.source_id.isin(kc19_df.source_id))
        &
        (~nbhd_df.source_id.isin(cg18_df.source_id))
        &
        (~nbhd_df.source_id.isin(m21_df.source_id))
    )
    orig_nbhd_df = deepcopy(nbhd_df)
    nbhd_df = nbhd_df[sel_nbhd]

    print(f'Got {len(nbhd_df)} neighbors')
    print(f'Got {len(cg18_df)} in core')
    print(f'Got {len(kc19_df)+len(m21_df)} in corona')
    print(f'Got {len(kc19_cg18_overlap_df)} KC19 / CG18 overlaps')
    print(f'Got {len(m21_cg18_overlap_df)} M21 / CG18 overlaps')
    assert (
        len(cg18_df)+len(kc19_df)+len(m21_df) ==
        len(np.unique(np.array(pd.concat((cg18_df, kc19_df, m21_df))['source_id'])))
    )

    cg18_df_edr3 = (
        given_dr2_sourceids_get_edr3_xmatch(
            nparr(cg18_df.Source).astype(np.int64), 'fullfaint_ngc2516_cg18_df',
            overwrite=False)
    )
    kc19_df_edr3 = (
        given_dr2_sourceids_get_edr3_xmatch(
            nparr(kc19_df.source_id).astype(np.int64), 'fullfaint_ngc2516_kc19_df',
            overwrite=False)
    )
    m21_df_edr3 = (
        given_dr2_sourceids_get_edr3_xmatch(
            nparr(m21_df.source_id).astype(np.int64), 'fullfaint_ngc2516_m21_df',
            overwrite=False)
    )
    nbhd_df_edr3 = (
        given_dr2_sourceids_get_edr3_xmatch(
            nparr(nbhd_df.source_id).astype(np.int64), 'fullfaint_ngc2516_nbhd_df',
            overwrite=False)
    )

    print(42*'='+'\nFOR EDR3:')

    # Take the closest (proper motion and epoch-corrected) angular distance as
    # THE single match.
    get_edr3_xm = lambda _df: (
        _df.sort_values(by='angular_distance').
        drop_duplicates(subset='dr2_source_id', keep='first')
    )

    s_cg18_df_edr3 = get_edr3_xm(cg18_df_edr3)
    s_kc19_df_edr3 = get_edr3_xm(kc19_df_edr3)
    s_m21_df_edr3 = get_edr3_xm(m21_df_edr3)
    s_nbhd_df_edr3 = get_edr3_xm(nbhd_df_edr3)

    print(f'Got {len(s_cg18_df_edr3)} EDR3 matches in core.\n'+
          f'99th pct [arcsec] {np.nanpercentile(cg18_df_edr3.angular_distance, 99):.1f} -> {np.nanpercentile(s_cg18_df_edr3.angular_distance, 99):.1f}')

    print(f'Got {len(s_kc19_df_edr3)} EDR3 matches in KC19.\n'+
          f'99th pct [arcsec] {np.nanpercentile(kc19_df_edr3.angular_distance, 99):.1f} -> {np.nanpercentile(s_kc19_df_edr3.angular_distance, 99):.1f}')

    print(f'Got {len(s_m21_df_edr3)} EDR3 matches in M21.\n'+
          f'99th pct [arcsec] {np.nanpercentile(m21_df_edr3.angular_distance, 99):.1f} -> {np.nanpercentile(s_m21_df_edr3.angular_distance, 99):.1f}')

    print(f'Got {len(s_nbhd_df_edr3)} EDR3 matches in nbhd.\n'+
          f'99th pct [arcsec] {np.nanpercentile(nbhd_df_edr3.angular_distance, 99):.1f} -> {np.nanpercentile(s_nbhd_df_edr3.angular_distance, 99):.1f}')

    # Finally, query Gaia EDR3 to get the latest and greatest fullfaint
    # photometry
    kc19_df_0 = given_source_ids_get_gaia_data(
        np.array(s_kc19_df_edr3.dr3_source_id),
        'fullfaint_ngc2516_kc19_df_edr3', n_max=10000, overwrite=False,
        enforce_all_sourceids_viable=True, gaia_datarelease='gaiaedr3'
    )
    cg18_df_0 = given_source_ids_get_gaia_data(
        np.array(s_cg18_df_edr3.dr3_source_id),
        'fullfaint_ngc2516_cg18_df_edr3', n_max=10000, overwrite=False,
        enforce_all_sourceids_viable=True, gaia_datarelease='gaiaedr3'
    )
    m21_df_0 = given_source_ids_get_gaia_data(
        np.array(s_m21_df_edr3.dr3_source_id),
        'fullfaint_ngc2516_m21_df_edr3', n_max=10000, overwrite=False,
        enforce_all_sourceids_viable=True, gaia_datarelease='gaiaedr3'
    )
    nbhd_df_0 = given_source_ids_get_gaia_data(
        np.array(s_nbhd_df_edr3.dr3_source_id),
        'fullfaint_ngc2516_nbhd_df_edr3', n_max=15000, overwrite=False,
        enforce_all_sourceids_viable=True, gaia_datarelease='gaiaedr3'
    )

    assert len(cg18_df) == len(cg18_df_0)
    assert len(kc19_df) == len(kc19_df_0)
    assert len(m21_df) == len(m21_df_0)
    assert len(nbhd_df) == len(nbhd_df_0)

    # nb. these "source_ids" are now EDR3 source_ids.
    np.testing.assert_array_equal(np.array(kc19_df_0.source_id),
                                  np.array(kc19_df_0.source_id_2))
    np.testing.assert_array_equal(np.array(cg18_df_0.source_id),
                                  np.array(cg18_df_0.source_id_2))
    np.testing.assert_array_equal(np.array(m21_df_0.source_id),
                                  np.array(m21_df_0.source_id_2))
    np.testing.assert_array_equal(np.array(nbhd_df_0.source_id),
                                  np.array(nbhd_df_0.source_id_2))

    kc19_df_0['dr2_source_id'] = nparr(s_kc19_df_edr3['dr2_source_id']).astype(np.int64)
    cg18_df_0['dr2_source_id'] = nparr(s_cg18_df_edr3['dr2_source_id']).astype(np.int64)
    m21_df_0['dr2_source_id'] = nparr(s_m21_df_edr3['dr2_source_id']).astype(np.int64)
    nbhd_df_0['dr2_source_id'] = nparr(s_nbhd_df_edr3['dr2_source_id']).astype(np.int64)

    target_df = kc19_df_0[kc19_df_0.source_id == 5489726768531119616] # TIC 2683...

    #
    # wrap up into the full source list
    #
    cg18_df_0['subcluster'] = 'core'
    kc19_df_0['subcluster'] = 'halo'
    m21_df_0['subcluster'] = 'halo'

    core_df = cg18_df_0
    halo_df = pd.concat((kc19_df_0, m21_df_0)).reset_index()

    full_df = pd.concat((core_df, halo_df)).reset_index()
    assert len(np.unique(full_df.source_id)) == len(full_df)
    print(f'Got {len(full_df)} unique sources in the cluster.')

    full_df['in_CG18'] = full_df.source_id.isin(cg18_df.source_id)
    full_df['in_KC19'] = full_df.source_id.isin(kc19_df.source_id)
    full_df['in_M21'] = full_df.source_id.isin(m21_df.source_id)

    nbhd_df['dr2_radial_velocity'] = nbhd_df['radial_velocity']

    return nbhd_df, core_df, halo_df, full_df, target_df


def _get_denis_fullfaint_edr3_dataframes():

    targetpath = '../data/denis/target_gaia_denis_xm.csv'
    cg18path = '../data/denis/cg18_gaia_denis_xm.csv'
    kc19path = '../data/denis/kc19_gaia_denis_xm.csv'
    nbhdpath = '../data/denis/nbhd_gaia_denis_xm.csv'

    return (
        pd.read_csv(nbhdpath),
        pd.read_csv(cg18path),
        pd.read_csv(kc19path),
        pd.read_csv(targetpath)
    )


def _get_extinction_dataframes():
    # supplement _get_nbhd_dataframes raw Gaia results with extinctions from
    # TIC8 (Stassun et al 2019).

    extinction_pkl = os.path.join(DATADIR, 'extinction_nbhd.pkl')

    if not os.path.exists(extinction_pkl):

        nbhd_df, cg18_df, kc19_df, target_df = _get_nbhd_dataframes()

        cg18_tic8_df = given_source_ids_get_tic8_data(
            np.array(cg18_df.source_id)[1],
            'ngc2516_cg18_earhart_tic8', n_max=len(cg18_df), overwrite=False,
            enforce_all_sourceids_viable=True
        )
        import IPython; IPython.embed()

        desiredcols = ['ID', 'GAIA', 'ebv', 'e_ebv', 'eneg_EBV', 'epos_EBV',
                       'EBVflag']

        thesecols = []
        for ix, s in enumerate(nbhd_df[:10].source_id):
            if ix % 10 == 0:
                print(f'{datetime.utcnow().isoformat()}: {ix}/{len(nbhd_df)}')

            t = gaiadr2_to_tic(str(s))
            tdf = get_tic_star_information(t, desiredcols=desiredcols,
                                           raise_error_on_multiple_match=True)
            thesecols.append(tdf)

        import IPython; IPython.embed()
        assert 0
        #FIXME get TICv8 matches through vizier's TAP service!

    else:

        with open(pklpath, 'rb') as f:
            d = pickle.load()

    nbhd_df, cg18_df, kc19_df, target_df = (
        d['nbhd_df'], d['cg18_df'], d['kc19_df'], d['target_df']
    )

    return nbhd_df, cg18_df, kc19_df, target_df


def get_denis_xmatch(c, _id=None, mag=None, drop_duplicates=1):
    """
    Given J2000 coordinate(s), search for the DENIS crossmatch(es).
    Does this via the CDS XMatch service, which goes through the Vizier tables
    and does a spatial crossmatch.

    args:

        c: astropy SkyCoord. Can contain many.

        _id: identifier string for the star(s).

        mag: optional magnitude(s) to compute a difference against.

        drop_duplicates: whether to use angular distance to do a true
        "left-join".

    returns:

        denis_xm (pandas DataFrame): containing searched coordinates,
        identifier, magnitudes, the default DENIS columns, and crucially
        `angDist` and `mag_diff`.
    """

    t = Table()
    t['RAJ2000'] = c.ra
    t['DEJ2000'] = c.dec
    if mag is not None:
        t['mag'] = mag
    if _id is not None:
        t['_id'] = _id

    # # NOTE: simplest crossmatching approach fails, b/c no easy merging.
    # # however, astroquery.XMatch, which queries the CDS XMatch service 
    # # <http://cdsxmatch.u-strasbg.fr/xmatch>
    # Vizier.ROW_LIMIT = -1
    # v = Vizier(columns=["*", "+_r"], catalog="B/denis")
    # denis_xm = v.query_region(t, radius="3s")[0]

    denis_xm = XMatch.query(
        cat1=t,
        cat2='vizier:B/denis/denis',
        max_distance=3*u.arcsec,
        colRA1='RAJ2000',
        colDec1='DEJ2000',
        colRA2='RAJ2000',
        colDec2='DEJ2000'
    )

    if mag is not None:
        denis_xm['mag_diff'] = denis_xm['mag'] - denis_xm['Imag']

    # Take the closest (proper motion and epoch-corrected) angular distance as
    # THE single match.
    get_leftjoin_xm = lambda _t: (
        _t.to_pandas().sort_values(by='angDist').
        drop_duplicates(subset='_id', keep='first')
    )

    if drop_duplicates:
        return get_leftjoin_xm(denis_xm)
    else:
        return denis_xm.to_pandas()


def get_autorotation_dataframe(runid='NGC_2516', verbose=1, returnbase=0,
                               cleaning=None):
    """
    runid = 'NGC_2516', for example

    Cleaning options:
        'defaultcleaning' P<15d, LSP>0.08, Nequal==0, Nclose<=1.
        'periodogram_match' requires LS and SPDM periods to agree to within 10%
        'match234_alias': requires LS and SPDM periods to agree to within 10%
            (up to 1x,2x,3x,4x harmonic).
        'nocleaning': P<99d.
        'defaultcleaning_cutProtColor': add Prot-color plane cut to
            defaultcleaning.
    """

    assert isinstance(cleaning, str)

    from earhart.paths import DATADIR
    rotdir = os.path.join(DATADIR, 'rotation')

    df = pd.read_csv(
        os.path.join(rotdir, f'{runid}_rotation_periods.csv')
    )
    if runid == 'NGC_2516':
        df = append_phot_binary_column(df)

    if cleaning in ['defaultcleaning', 'periodogram_match',
                    'match234_alias','harderlsp', 'defaultcleaning_cutProtColor']:
        # automatic selection criteria for viable rotation periods
        NEQUAL_CUTOFF = 0 # could also do 1
        NCLOSE_CUTOFF = 1
        LSP_CUTOFF = 0.08 # 0.08 standard
        if cleaning == 'harderlsp':
            LSP_CUTOFF = 0.15
        sel = (
            (df.period < 15)
            &
            (df.lspval > LSP_CUTOFF)
            &
            (df.nequal <= NEQUAL_CUTOFF)
            &
            (df.nclose <= NCLOSE_CUTOFF)
        )
        if cleaning == 'defaultcleaning_cutProtColor':
            from earhart.priors import AVG_EBpmRp
            BpmRp0 = (
                df['phot_bp_mean_mag'] - df['phot_rp_mean_mag'] - AVG_EBpmRp
            )
            Prot_boundary = PleaidesQuadProtModel(BpmRp0)
            sel &= (
                df.period < Prot_boundary
            )

    elif cleaning in ['nocleaning']:
        NEQUAL_CUTOFF = 99999
        NCLOSE_CUTOFF = 99999
        LSP_CUTOFF = 0
        sel = (
            (df.period < 99)
        )
    else:
        raise ValueError(f'Got cleaning == {cleaning}, not recognized.')

    if cleaning in ['defaultcleaning', 'nocleaning', 'harderlsp',
                    'defaultcleaning_cutProtColor']:
        pass

    elif cleaning == 'periodogram_match':
        sel_periodogram_match = (
            (0.9 < (df.spdmperiod/df.period))
            &
            (1.1 > (df.spdmperiod/df.period))
        )
        sel &= sel_periodogram_match

    elif cleaning == 'match234_alias':
        sel_match = (
            (0.9 < (df.spdmperiod/df.period))
            &
            (1.1 > (df.spdmperiod/df.period))
        )
        sel_spdm2x = (
            (1.9 < (df.spdmperiod/df.period))
            &
            (2.1 > (df.spdmperiod/df.period))
        )
        sel_spdm3x = (
            (2.9 < (df.spdmperiod/df.period))
            &
            (3.1 > (df.spdmperiod/df.period))
        )
        sel_spdm4x = (
            (3.9 < (df.spdmperiod/df.period))
            &
            (4.1 > (df.spdmperiod/df.period))
        )
        sel &= (
            sel_match
            |
            sel_spdm2x
            |
            sel_spdm3x
            |
            sel_spdm4x
        )

    else:
        raise ValueError(f'Got cleaning == {cleaning}, not recognized.')

    ref_sel = (
        (df.nequal <= NEQUAL_CUTOFF)
        &
        (df.nclose <= NCLOSE_CUTOFF)
    )

    if verbose:
        print(f'Getting autorotation dataframe for {runid}...')
        print(f'Starting with {len(df[ref_sel])} entries that meet NEQUAL and NCLOSE criteria...')
        print(f'Got {len(df[sel])} entries with P<15d, LSP>{LSP_CUTOFF}, nequal<={NEQUAL_CUTOFF}, nclose<={NCLOSE_CUTOFF}')
        if cleaning == 'periodogram_match':
            print(f'...AND required LS and SPDM periods to agree.')
        elif cleaning == 'match234_alias':
            print(f'...AND required LS and SPDM periods to agree (up to 1x,2x,3x,4x harmonic).')
        print(10*'.')

        if 'compstar' in runid:
            rp_sel = (df.phot_rp_mean_mag < 13)
            from earhart.priors import AVG_EBpmRp
            BpmRp0 = (
                df['phot_bp_mean_mag'] - df['phot_rp_mean_mag'] - AVG_EBpmRp
            )
            bpmrp_sel = (BpmRp0 < 0.8) & (BpmRp0 > 0.4)

            print(f'Requiring 0.4<Bp-Rp0<0.8 as well for comparison gives...')

            print(f'Starting with {len(df[ref_sel & bpmrp_sel])} entries that meet NEQUAL and NCLOSE criteria...')
            print(f'Got {len(df[sel & bpmrp_sel])} entries with P<15d, LSP>{LSP_CUTOFF}, nequal<={NEQUAL_CUTOFF}, nclose<={NCLOSE_CUTOFF}')
            frac = len(df[sel & bpmrp_sel])/len(df[ref_sel & bpmrp_sel])
            print(f'Fraction: {frac:.4f}')
            print(10*'.')


    if not returnbase:
        return df[sel]
    else:
        return df[sel], df[ref_sel]


def _get_median_ngc2516_core_params(core_df, basedata, CUTOFF_PROB=0.7):
    """
    To get median parameters of the NGC 2516 cluster, select the high
    probability CG18 members, and take a median.
    """

    core_path = os.path.join(DATADIR, 'gaia', 'CantatGaudin2018_vizier_only_NGC2516.fits')
    t_df = Table(fits.open(core_path)[1].data).to_pandas()
    if basedata == 'fullfaint':
        assert len(t_df) == len(core_df)

    sel = (t_df.PMemb > CUTOFF_PROB)
    s_t_df = t_df[sel]
    assert np.all(s_t_df.Source.isin(core_df.source_id))
    sel = core_df.source_id.isin(s_t_df.Source)
    s_core_df = core_df[sel]
    rvkey = (
        'radial_velocity' if 'edr3' not in basedata else 'dr2_radial_velocity'
    )
    getcols = ['ra', 'dec', 'parallax', 'pmra', 'pmdec', rvkey]
    med_df = pd.DataFrame(s_core_df[getcols].median()).T
    std_df = pd.DataFrame(s_core_df[getcols].std()).T

    return med_df, std_df


def append_phot_binary_column(df, DIFFERENCE_CUTOFF=0.3):

    from scipy.interpolate import interp1d

    csvpath = os.path.join(DATADIR, 'gaia',
                           'ngc2516_AbsG_BpmRp_empirical_locus_webplotdigitzed.csv')
    ldf = pd.read_csv(csvpath)

    fn_BpmRp_to_AbsG = interp1d(ldf.BpmRp, ldf.AbsG, kind='quadratic',
                                bounds_error=False, fill_value=np.nan)

    get_yval = (
        lambda _df: np.array(
            _df['phot_g_mean_mag'] + 5*np.log10(_df['parallax']/1e3) + 5
        )
    )
    get_xval = (
        lambda _df: np.array(
            _df['phot_bp_mean_mag'] - _df['phot_rp_mean_mag']
        )
    )

    sel_photbin = (
        get_yval(df) <
        ( fn_BpmRp_to_AbsG(get_xval(df)) - DIFFERENCE_CUTOFF)
    )

    df['is_phot_binary'] = sel_photbin

    return df


def PleaidesQuadProtModel(BpmRp0):
    """
    Pleiades Prot vs color from Curtis+20...
    used as an exclusion criteria with some slack on the red and blue
    ends.
    """
    x = BpmRp0
    c0 = -8.467
    c1 = 19.64
    c2 = -5.438
    Protmod = (
        c0*x**0 + c1*x**1 + c2*x**2
        + 2
    )
    sel = (x > 1.4)
    Protmod[sel] = 12
    sel = (x > 0.3) & (x < 0.5)
    Protmod[sel] = 2

    sel = (x < 0.3) | (x>2.4)
    Protmod[sel] = np.nan

    return Protmod
