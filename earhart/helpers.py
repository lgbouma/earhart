"""
get_toi1937_lightcurve
get_groundphot
_get_nbhd_dataframes
_get_fullfaint_dataframes
_get_fullfaint_edr3_dataframes
_get_extinction_dataframes
"""
import os, collections, pickle
import numpy as np, pandas as pd
from glob import glob
from copy import deepcopy

from numpy import array as nparr

from astropy.io import fits
from astropy import units as u
from astropy.table import Table

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


def _get_nbhd_dataframes():
    """
    Return: nbhd_df, cg18_df, kc19_df, target_df
    (for NGC 2516)

    The "core" is the match of the CDIPS target catalog (G_Rp<16) with
    Cantat-Gaudin 2018.

    The "halo" is the match of the CDIPS target catalog (G_Rp<16) with
    Kounkel & Covey 2019, provided that the source is not in the core. (i.e.,
    KC19 get no points for getting the "core" targets correct).

    The "neighborhood" was selected via

        bounds = { 'parallax_lower': 1.5, 'parallax_upper': 4.0, 'ra_lower': 108,
        'ra_upper': 132, 'dec_lower': -76, 'dec_upper': -45 }

        nbhd_df = query_neighborhood(bounds, groupname, n_max=6000,
                                     overwrite=False, manual_gmag_limit=17)

    This procedure yields:
        Got 5908 neighbors
        Got 893 in core (CG18)
        Got 1345 in corona (KC19, minus any CG18)
        Got 888 KC19 / CG18 overlaps
    """

    df = get_cdips_catalog(ver=0.4)

    kc19_sel = (
        (df.cluster.str.contains('NGC_2516')) &
        (df.reference.str.contains('Kounkel_2019'))
    )
    cg18_sel = (
        (df.cluster.str.contains('NGC_2516')) &
        (df.reference.str.contains('CantatGaudin_2018'))
    )

    kc19_df = df[kc19_sel]
    cg18_df = df[cg18_sel]

    kc19_cg18_overlap_df = kc19_df[(kc19_df.source_id.isin(cg18_df.source_id))]

    kc19_df = kc19_df[~(kc19_df.source_id.isin(cg18_df.source_id))]


    ##########

    # NGC 2516 rough
    bounds = {
        'parallax_lower': 1.5, 'parallax_upper': 4.0, 'ra_lower': 108,
        'ra_upper': 132, 'dec_lower': -76, 'dec_upper': -45
    }
    groupname = 'customngc2516'

    nbhd_df = query_neighborhood(bounds, groupname, n_max=6000,
                                 overwrite=False, manual_gmag_limit=17)

    # NOTE: kind of silly: need to requery gaia DR2 because the CDIPS target
    # catalog, where the source_ids are from, doesn't have the radial
    # velocities.

    kc19_df_0 = given_source_ids_get_gaia_data(
        np.array(kc19_df.source_id),
        'ngc2516_kc19_earhart', n_max=10000, overwrite=False,
        enforce_all_sourceids_viable=True
    )
    cg18_df_0 = given_source_ids_get_gaia_data(
        np.array(cg18_df.source_id),
        'ngc2516_cg18_earhart', n_max=10000, overwrite=False,
        enforce_all_sourceids_viable=True
    )

    assert len(cg18_df) == len(cg18_df_0)
    assert len(kc19_df) == len(kc19_df_0)

    target_df = kc19_df_0[kc19_df_0.source_id == 5489726768531119616] # TIC 2683...

    sel_nbhd = (
        (~nbhd_df.source_id.isin(kc19_df.source_id))
        &
        (~nbhd_df.source_id.isin(cg18_df.source_id))
    )
    orig_nbhd_df = deepcopy(nbhd_df)
    nbhd_df = nbhd_df[sel_nbhd]

    print(f'Got {len(nbhd_df)} neighbors')
    print(f'Got {len(cg18_df)} in core')
    print(f'Got {len(kc19_df)} in corona')
    print(f'Got {len(kc19_cg18_overlap_df)} KC19 / CG18 overlaps')

    return nbhd_df, cg18_df_0, kc19_df_0, target_df


def _get_fullfaint_dataframes():
    """
    Return: nbhd_df, cg18_df, kc19_df, target_df
    (for NGC 2516, "full faint" sample -- i.e., as faint as possible.)

    The "core" is all available Cantat-Gaudin 2018 members, with no magnitude
    cutoff.

    The "halo" is the full Kounkel & Covey 2019 member set, provided that the
    source is not in the core. (i.e., KC19 get no points for getting the "core"
    targets correct).

    The "neighborhood" was selected via

        bounds = { 'parallax_lower': 1.5, 'parallax_upper': 4.0, 'ra_lower': 108,
        'ra_upper': 132, 'dec_lower': -76, 'dec_upper': -45 }

        nbhd_df = query_neighborhood(bounds, groupname, n_max=14000,
                                     overwrite=False, manual_gmag_limit=19)

    This procedure yields:

        Got 1106 in fullfaint CG18
        Got 3003 in fullfaint KC19
        Got 1912 in fullfaint KC19 after removing matches
        Got 13843 neighbors
        Got 1106 in core
        Got 1912 in corona
        Got 1091 KC19 / CG18 overlaps
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

    print(f'Got {len(cg18_df)} in fullfaint CG18')
    print(f'Got {len(kc19_df)} in fullfaint KC19')

    kc19_cg18_overlap_df = kc19_df[(kc19_df.source_id.isin(cg18_df.source_id))]
    kc19_df = kc19_df[~(kc19_df.source_id.isin(cg18_df.source_id))]

    print(f'Got {len(kc19_df)} in fullfaint KC19 after removing matches')

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

    assert len(cg18_df) == len(cg18_df_0)
    assert len(kc19_df) == len(kc19_df_0)

    target_df = kc19_df_0[kc19_df_0.source_id == 5489726768531119616] # TIC 2683...

    sel_nbhd = (
        (~nbhd_df.source_id.isin(kc19_df.source_id))
        &
        (~nbhd_df.source_id.isin(cg18_df.source_id))
    )
    orig_nbhd_df = deepcopy(nbhd_df)
    nbhd_df = nbhd_df[sel_nbhd]

    print(f'Got {len(nbhd_df)} neighbors')
    print(f'Got {len(cg18_df)} in core')
    print(f'Got {len(kc19_df)} in corona')
    print(f'Got {len(kc19_cg18_overlap_df)} KC19 / CG18 overlaps')

    return nbhd_df, cg18_df_0, kc19_df_0, target_df


def _get_fullfaint_edr3_dataframes():
    """
    Return: nbhd_df, cg18_df, kc19_df, target_df
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
            Got 1912 in fullfaint KC19 after removing matches
            Got 13843 neighbors
            Got 1106 in core
            Got 1912 in corona
            Got 1091 KC19 / CG18 overlaps
        FOR EDR3:

            CG18/core: got 1143 matches vs 1106 source id queries.
            KC19/halo: got 2005 matches vs 1912 source id queries
            Nbhd:      got 15123 matches vs 13843 source id queries.

            Got 1106 EDR3 matches in core.
            99th pct [arcsec] 1577.8 -> 0.3
            Got 1912 EDR3 matches in halo.
            99th pct [arcsec] 1702.8 -> 0.5
            Got 13843 EDR3 matches in nbhd.
            99th pct [arcsec] 1833.9 -> 3.7

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

    print(42*'='+'\nFOR DR2:')
    print(f'Got {len(cg18_df)} in fullfaint CG18')
    print(f'Got {len(kc19_df)} in fullfaint KC19')

    kc19_cg18_overlap_df = kc19_df[(kc19_df.source_id.isin(cg18_df.source_id))]
    kc19_df = kc19_df[~(kc19_df.source_id.isin(cg18_df.source_id))]

    print(f'Got {len(kc19_df)} in fullfaint KC19 after removing matches')

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
    )
    orig_nbhd_df = deepcopy(nbhd_df)
    nbhd_df = nbhd_df[sel_nbhd]

    print(f'Got {len(nbhd_df)} neighbors')
    print(f'Got {len(cg18_df)} in core')
    print(f'Got {len(kc19_df)} in corona')
    print(f'Got {len(kc19_cg18_overlap_df)} KC19 / CG18 overlaps')

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
    s_nbhd_df_edr3 = get_edr3_xm(nbhd_df_edr3)

    print(f'Got {len(s_cg18_df_edr3)} EDR3 matches in core.\n'+
          f'99th pct [arcsec] {np.nanpercentile(cg18_df_edr3.angular_distance, 99):.1f} -> {np.nanpercentile(s_cg18_df_edr3.angular_distance, 99):.1f}')

    print(f'Got {len(s_kc19_df_edr3)} EDR3 matches in halo.\n'+
          f'99th pct [arcsec] {np.nanpercentile(kc19_df_edr3.angular_distance, 99):.1f} -> {np.nanpercentile(s_kc19_df_edr3.angular_distance, 99):.1f}')

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
    nbhd_df_0 = given_source_ids_get_gaia_data(
        np.array(s_nbhd_df_edr3.dr3_source_id),
        'fullfaint_ngc2516_nbhd_df_edr3', n_max=15000, overwrite=False,
        enforce_all_sourceids_viable=True, gaia_datarelease='gaiaedr3'
    )

    assert len(cg18_df) == len(cg18_df_0)
    assert len(kc19_df) == len(kc19_df_0)
    assert len(nbhd_df) == len(nbhd_df_0)

    # nb. these source_ids are now EDR3 source_ids.
    np.testing.assert_array_equal(np.array(kc19_df_0.source_id),
                                  np.array(kc19_df_0.source_id_2))
    np.testing.assert_array_equal(np.array(cg18_df_0.source_id),
                                  np.array(cg18_df_0.source_id_2))
    np.testing.assert_array_equal(np.array(nbhd_df_0.source_id),
                                  np.array(nbhd_df_0.source_id_2))

    target_df = kc19_df_0[kc19_df_0.source_id == 5489726768531119616] # TIC 2683...

    return nbhd_df_0, cg18_df_0, kc19_df_0, target_df


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


def _given_gaia_df_get_icrs_arr(df):

    import astropy.coordinates as coord
    coord.galactocentric_frame_defaults.set('v4.0')

    return coord.SkyCoord(
        ra=nparr(df.ra)*u.deg,
        dec=nparr(df.dec)*u.deg,
        distance=nparr(1/(df.parallax*1e-3))*u.pc,
        pm_ra_cosdec=nparr(df.pmra)*u.mas/u.yr,
        pm_dec=nparr(df.pmdec)*u.mas/u.yr,
        radial_velocity=nparr(df.dr2_radial_velocity)*u.km/u.s
    )

def calc_dist(x0, y0, z0, x1, y1, z1):

    d = np.sqrt(
        (x0-x1)**2 + (y0-y1)**2 + (z0-z1)**2
    )

    return d
