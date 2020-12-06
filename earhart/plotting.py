"""
Plots:
    plot_full_kinematics
    plot_TIC268_nbhd_small
    plot_hr
    plot_rotation
    plot_skypositions_x_rotn
    plot_randich_lithium
    plot_galah_dr3_lithium
    plot_auto_rotation

Helpers:
    _get_nbhd_dataframes
"""
import os, corner, pickle
from glob import glob
from datetime import datetime
import numpy as np, matplotlib.pyplot as plt, pandas as pd, pymc3 as pm
from numpy import array as nparr

import matplotlib as mpl

from astropy import units as u, constants as const
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time

from copy import deepcopy

import matplotlib.patheffects as path_effects

from aesthetic.plot import savefig, format_ax
from aesthetic.plot import set_style

from astrobase.services.identifiers import gaiadr2_to_tic
from cdips.utils.catalogs import (
    get_cdips_catalog, get_tic_star_information
)
from cdips.utils.gaiaqueries import (
    query_neighborhood, given_source_ids_get_gaia_data
)
from cdips.utils.tapqueries import given_source_ids_get_tic8_data
from cdips.utils.plotutils import rainbow_text
from cdips.utils.mamajek import get_interp_BpmRp_from_Teff

from earhart.paths import DATADIR, RESULTSDIR

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


def plot_TIC268_nbhd_small(outdir=RESULTSDIR):

    nbhd_df, cg18_df, kc19_df, target_df = _get_nbhd_dataframes()

    set_style()

    plt.close('all')

    f, axs = plt.subplots(figsize=(4,3), ncols=2)

    xv, yv = 'ra', 'dec'
    axs[0].scatter(
        nbhd_df[xv], nbhd_df[yv], c='gray', alpha=0.5, zorder=2, s=7,
        rasterized=True, linewidths=0, label='Field', marker='.'
    )
    axs[0].scatter(
        kc19_df[xv], kc19_df[yv], c='lightskyblue', alpha=0.9, zorder=3, s=7,
        rasterized=True, linewidths=0.15, label='Halo', marker='.',
        edgecolors='k'
    )
    axs[0].scatter(
        cg18_df[xv], cg18_df[yv], c='k', alpha=0.9, zorder=4, s=7,
        rasterized=True, label='Core', marker='.'
    )
    axs[0].plot(
        target_df[xv], target_df[yv], alpha=1, mew=0.5,
        zorder=8, label='TOI 1937', markerfacecolor='yellow',
        markersize=14, marker='*', color='black', lw=0
    )

    axs[0].set_xlabel(r'$\alpha$ [deg]')
    axs[0].set_ylabel(r'$\delta$ [deg]')
    axs[0].set_xlim([108, 132])
    axs[0].set_ylim([-76, -45])

    ##########

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

    axs[1].scatter(
        get_xval(nbhd_df), get_yval(nbhd_df), c='gray', alpha=0.8, zorder=2,
        s=7, rasterized=True, linewidths=0, label='Field', marker='.'
    )
    axs[1].scatter(
        get_xval(kc19_df), get_yval(kc19_df), c='lightskyblue', alpha=1,
        zorder=3, s=7, rasterized=True, linewidths=0.15, label='Halo',
        marker='.', edgecolors='k'
    )
    axs[1].scatter(
        get_xval(cg18_df), get_yval(cg18_df), c='k', alpha=0.9,
        zorder=4, s=7, rasterized=True, linewidths=0, label='Core', marker='.'
    )
    axs[1].plot(
        get_xval(target_df), get_yval(target_df), alpha=1, mew=0.5,
        zorder=8, label='TOI 1937', markerfacecolor='yellow',
        markersize=14, marker='*', color='black', lw=0
    )

    axs[1].set_ylim(axs[1].get_ylim()[::-1])

    axs[1].set_xlabel('Bp - Rp [mag]')
    axs[1].set_ylabel('Absolute G [mag]', labelpad=-6)

    ##########

    words = ['Field', 'Halo', 'Core'][::-1]
    colors = ['gray', 'lightskyblue', 'k'][::-1]
    rainbow_text(0.98, 0.02, words, colors, size='medium', ax=axs[0])

    f.tight_layout(w_pad=2)

    outpath = os.path.join(outdir, 'small_ngc2516.png')
    savefig(f, outpath)


def plot_full_kinematics(outdir):

    nbhd_df, cg18_df, kc19_df, target_df = _get_nbhd_dataframes()

    plt.close('all')

    params = ['ra', 'dec', 'parallax', 'pmra', 'pmdec', 'radial_velocity']
    nparams = len(params)

    qlimd = {
        'ra': 0, 'dec': 0, 'parallax': 0, 'pmra': 1, 'pmdec': 1,
        'radial_velocity': 1
    } # whether to limit axis by 16/84th percetile
    nnlimd = {
        'ra': 1, 'dec': 1, 'parallax': 1, 'pmra': 0, 'pmdec': 0,
        'radial_velocity': 0
    } # whether to limit axis by 99th percentile

    ldict = {
        'ra': r'$\alpha$ [deg]',
        'dec': r'$\delta$ [deg]',
        'parallax': r'$\pi$ [mas]',
        'pmra': r"$\mu_{{\alpha'}}$ [mas/yr]",
        'pmdec':  r'$\mu_{{\delta}}$ [mas/yr]',
        'radial_velocity': 'RV [km/s]'
    }

    f, axs = plt.subplots(figsize=(6,6), nrows=nparams-1, ncols=nparams-1)

    for i in range(nparams):
        for j in range(nparams):
            print(i,j)
            if j == nparams-1 or i == nparams-1:
                continue
            if j>i:
                axs[i,j].set_axis_off()
                continue

            xv = params[j]
            yv = params[i+1]
            print(i,j,xv,yv)

            axs[i,j].scatter(
                nbhd_df[xv], nbhd_df[yv], c='gray', alpha=0.9, zorder=2, s=5,
                rasterized=True, linewidths=0, label='Field', marker='.'
            )
            axs[i,j].scatter(
                kc19_df[xv], kc19_df[yv], c='lightskyblue', alpha=1,
                zorder=3, s=12, rasterized=True, label='Halo',
                linewidths=0.1, marker='.', edgecolors='k'
            )
            axs[i,j].scatter(
                cg18_df[xv], cg18_df[yv], c='k', alpha=0.9, zorder=4, s=5,
                rasterized=True, label='Core', marker='.'
            )
            axs[i,j].plot(
                target_df[xv], target_df[yv], alpha=1, mew=0.5,
                zorder=8, label='TOI 1937', markerfacecolor='yellow',
                markersize=14, marker='*', color='black', lw=0
            )

            # set the axis limits as needed
            if qlimd[xv]:
                xlim = (np.nanpercentile(nbhd_df[xv], 16),
                        np.nanpercentile(nbhd_df[xv], 84))
                axs[i,j].set_xlim(xlim)
            if qlimd[yv]:
                ylim = (np.nanpercentile(nbhd_df[yv], 16),
                        np.nanpercentile(nbhd_df[yv], 84))
                axs[i,j].set_ylim(ylim)
            if nnlimd[xv]:
                xlim = (np.nanpercentile(nbhd_df[xv], 1),
                        np.nanpercentile(nbhd_df[xv], 99))
                axs[i,j].set_xlim(xlim)
            if nnlimd[yv]:
                ylim = (np.nanpercentile(nbhd_df[yv], 1),
                        np.nanpercentile(nbhd_df[yv], 99))
                axs[i,j].set_ylim(ylim)


            # fix labels
            if j == 0 :
                axs[i,j].set_ylabel(ldict[yv], fontsize='small')
                if not i == nparams - 2:
                    # hide xtick labels
                    labels = [item.get_text() for item in axs[i,j].get_xticklabels()]
                    empty_string_labels = ['']*len(labels)
                    axs[i,j].set_xticklabels(empty_string_labels)

            if i == nparams - 2:
                axs[i,j].set_xlabel(ldict[xv], fontsize='small')
                if not j == 0:
                    # hide ytick labels
                    labels = [item.get_text() for item in axs[i,j].get_yticklabels()]
                    empty_string_labels = ['']*len(labels)
                    axs[i,j].set_yticklabels(empty_string_labels)

            if (not (j == 0)) and (not (i == nparams - 2)):
                # hide ytick labels
                labels = [item.get_text() for item in axs[i,j].get_yticklabels()]
                empty_string_labels = ['']*len(labels)
                axs[i,j].set_yticklabels(empty_string_labels)
                # hide xtick labels
                labels = [item.get_text() for item in axs[i,j].get_xticklabels()]
                empty_string_labels = ['']*len(labels)
                axs[i,j].set_xticklabels(empty_string_labels)

    axs[2,2].legend(loc='best', handletextpad=0.1, fontsize='medium', framealpha=0.7)
    leg = axs[2,2].legend(bbox_to_anchor=(0.8,0.8), loc="upper right",
                          handletextpad=0.1, fontsize='medium',
                          bbox_transform=f.transFigure)

    # NOTE: hack size of legend markers
    leg.legendHandles[0]._sizes = [20]
    leg.legendHandles[1]._sizes = [25]
    leg.legendHandles[2]._sizes = [25]
    leg.legendHandles[3]._sizes = [20]

    for ax in axs.flatten():
        format_ax(ax)

    f.tight_layout(h_pad=0.05, w_pad=0.05)

    outpath = os.path.join(outdir, 'full_kinematics.png')
    savefig(f, outpath)


def plot_hr(outdir, isochrone=None, color0='phot_bp_mean_mag',
            include_extinction=False):

    set_style()

    if include_extinction:
        raise NotImplementedError('still need to implement extinction')
        nbhd_df, cg18_df, kc19_df, target_df = _get_extinction_dataframes()
    else:
        nbhd_df, cg18_df, kc19_df, target_df = _get_nbhd_dataframes()

    if isochrone in ['mist', 'parsec']:
        raise NotImplementedError
        if isochrone == 'mist':
            from timmy.read_mist_model import ISOCMD
            isocmdpath = os.path.join(DATADIR, 'cluster',
                                      'MIST_isochrones_age7pt60206_Av0pt217_FeH0',
                                      'MIST_iso_5f04eb2b54f51.iso.cmd')
            # relevant params: star_mass log_g log_L log_Teff Gaia_RP_DR2Rev
            # Gaia_BP_DR2Rev Gaia_G_DR2Rev
            isocmd = ISOCMD(isocmdpath)
            # 10, 20, 30, 40 Myr.
            assert len(isocmd.isocmds) == 4

        elif isochrone == 'parsec':
            isopath = os.path.join(DATADIR, 'cluster', 'PARSEC_isochrones',
                                   'output799447099984.dat')
            iso_df = pd.read_csv(isopath, delim_whitespace=True)


    ##########

    plt.close('all')

    f, ax = plt.subplots(figsize=(4,3))

    get_yval = (
        lambda _df: np.array(
            _df['phot_g_mean_mag'] + 5*np.log10(_df['parallax']/1e3) + 5
        )
    )
    get_xval = (
        lambda _df: np.array(
            _df[color0] - _df['phot_rp_mean_mag']
        )
    )

    ax.scatter(
        get_xval(nbhd_df), get_yval(nbhd_df), c='gray', alpha=0.8, zorder=2,
        s=5, rasterized=True, linewidths=0, label='Field', marker='.'
    )
    ax.scatter(
        get_xval(kc19_df), get_yval(kc19_df), c='lightskyblue', alpha=1,
        zorder=3, s=5, rasterized=True, linewidths=0.15, label='Halo',
        marker='.', edgecolors='k'
    )
    ax.scatter(
        get_xval(cg18_df), get_yval(cg18_df), c='k', alpha=0.9,
        zorder=4, s=5, rasterized=True, linewidths=0, label='Core', marker='.'
    )
    ax.plot(
        get_xval(target_df), get_yval(target_df), alpha=1, mew=0.5,
        zorder=8, label='TOI 1937', markerfacecolor='yellow',
        markersize=10, marker='*', color='black', lw=0
    )

    if isochrone:

        if isochrone == 'mist':

            if not do_cmd:
                print(f'{mediancorr:.2f}')

            ages = [10, 20, 30, 40]
            N_ages = len(ages)
            colors = plt.cm.cool(np.linspace(0,1,N_ages))[::-1]

            for i, (a, c) in enumerate(zip(ages, colors)):
                mstar = isocmd.isocmds[i]['star_mass']
                sel = (mstar < 7)

                if not do_cmd:
                    _yval = isocmd.isocmds[i]['Gaia_G_DR2Rev'][sel] + mediancorr
                else:
                    corr = 5.75
                    _yval = isocmd.isocmds[i]['Gaia_G_DR2Rev'][sel] + corr

                if color0 == 'phot_bp_mean_mag':
                    _c0 = 'Gaia_BP_DR2Rev'
                elif color0 == 'phot_g_mean_mag':
                    _c0 = 'Gaia_G_DR2Rev'
                else:
                    raise NotImplementedError

                ax.plot(
                    isocmd.isocmds[i][_c0][sel]-isocmd.isocmds[i]['Gaia_RP_DR2Rev'][sel],
                    _yval,
                    c=c, alpha=1., zorder=4, label=f'{a} Myr', lw=0.5
                )

                if i == 3 and do_cmd:
                    sel = (mstar < 1.15) & (mstar > 1.08)
                    print(mstar[sel])
                    teff = 10**isocmd.isocmds[i]['log_Teff']
                    print(teff[sel])
                    logg = isocmd.isocmds[i]['log_g']
                    print(logg[sel])
                    rstar = ((( (10**logg)*u.cm/(u.s*u.s)) /
                              (const.G*mstar*u.Msun))**(-1/2)).to(u.Rsun)
                    print(rstar[sel])
                    rho = (mstar*u.Msun/(4/3*np.pi*rstar**3)).cgs
                    print(rho[sel])

                    _yval = isocmd.isocmds[i]['Gaia_G_DR2Rev'][sel] + corr

                    ax.scatter(
                        isocmd.isocmds[i][_c0][sel]-isocmd.isocmds[i]['Gaia_RP_DR2Rev'][sel],
                        _yval,
                        c=c, alpha=1., zorder=10, s=0.5, marker=".", linewidths=0
                    )

        elif isochrone == 'parsec':

            if not do_cmd:
                print(f'{mediancorr:.2f}')

            ages = [10, 20, 30, 40]
            logages = [7, 7.30103, 7.47712, 7.60206]
            N_ages = len(ages)
            colors = plt.cm.cool(np.linspace(0,1,N_ages))[::-1]

            for i, (a, la, c) in enumerate(zip(ages, logages, colors)):

                sel = (np.abs(iso_df.logAge - la) < 0.01) & (iso_df.Mass < 7)

                if not do_cmd:
                    _yval = iso_df[sel]['Gmag'] + mediancorr
                else:
                    corr = 5.65
                    _yval = iso_df[sel]['Gmag'] + corr

                if color0 == 'phot_bp_mean_mag':
                    _c0 = 'G_BPmag'
                elif color0 == 'phot_g_mean_mag':
                    _c0 = 'Gmag'
                else:
                    raise NotImplementedError

                ax.plot(
                    iso_df[sel][_c0]-iso_df[sel]['G_RPmag'],
                    _yval,
                    c=c, alpha=1., zorder=4, label=f'{a} Myr', lw=0.5
                )

                if i == 3 and do_cmd:
                    sel = (
                        (np.abs(iso_df.logAge - la) < 0.01) &
                        (iso_df.Mass < 1.1) &
                        (iso_df.Mass > 0.95)
                    )
                    mstar = np.array(iso_df.Mass)

                    print(42*'#')
                    print(f'{_c0} - Rp')
                    print(mstar[sel])
                    teff = np.array(10**iso_df['logTe'])
                    print(teff[sel])
                    logg = np.array(iso_df['logg'])
                    print(logg[sel])
                    rstar = ((( (10**logg)*u.cm/(u.s*u.s)) /
                              (const.G*mstar*u.Msun))**(-1/2)).to(u.Rsun)
                    print(rstar[sel])
                    rho = (mstar*u.Msun/(4/3*np.pi*rstar**3)).cgs
                    print(rho[sel])

                    _yval = iso_df[sel]['Gmag'] + corr

                    ax.scatter(
                        iso_df[sel][_c0]-iso_df[sel]['G_RPmag'],
                        _yval,
                        c='red', alpha=1., zorder=10, s=2, marker=".", linewidths=0
                    )


    leg = ax.legend(loc='upper right', handletextpad=0.1, fontsize='x-small',
                    framealpha=0.9)
    # NOTE: hack size of legend markers
    leg.legendHandles[0]._sizes = [18]
    leg.legendHandles[1]._sizes = [25]
    leg.legendHandles[2]._sizes = [25]
    leg.legendHandles[3]._sizes = [25]


    ax.set_ylabel('Absolute G [mag]', fontsize='large')
    if color0 == 'phot_bp_mean_mag':
        ax.set_xlabel('Bp - Rp [mag]', fontsize='large')
    elif color0 == 'phot_g_mean_mag':
        ax.set_xlabel('G - Rp [mag]', fontsize='large')
    else:
        raise NotImplementedError

    ylim = ax.get_ylim()
    ax.set_ylim((max(ylim),min(ylim)))

    format_ax(ax)
    if not isochrone:
        s = ''
    else:
        s = '_'+isochrone
    c0s = '_Bp_m_Rp' if color0 == 'phot_bp_mean_mag' else '_G_m_Rp'
    outpath = os.path.join(outdir, f'hr{s}{c0s}.png')

    savefig(f, outpath, dpi=400)


def plot_rotation(outdir, BpmRp=0, include_ngc2516=0, ngc_core_halo=0):

    from earhart.priors import TEFF, P_ROT, AVG_EBpmRp

    set_style()

    rotdir = os.path.join(DATADIR, 'rotation')

    # make plot
    plt.close('all')

    f, ax = plt.subplots(figsize=(4,3))

    if not include_ngc2516:
        classes = ['pleiades', 'praesepe']
        colors = ['k', 'gray']
        zorders = [3, 2]
        markers = ['o', 'x']
        lws = [0, 0.]
        mews= [0.5, 0.5]
        ss = [3.0, 6]
        labels = ['Pleaides', 'Praesepe']
    else:
        classes = ['pleiades', 'praesepe', 'ngc2516']
        colors = ['k', 'gray', 'C0']
        zorders = [3, 2, 4]
        markers = ['o', 'x', 'o']
        lws = [0, 0., 0]
        mews= [0.5, 0.5, 0.5]
        ss = [3.0, 6, 3]
        labels = ['Pleaides', 'Praesepe', 'NGC2516']

    # plot vals
    for _cls, _col, z, m, l, lw, s, mew in zip(
        classes, colors, zorders, markers, labels, lws, ss, mews
    ):

        if 'ngc2516' not in _cls:
            df = pd.read_csv(os.path.join(rotdir, f'curtis19_{_cls}.csv'))
        else:
            df = pd.read_csv(
                os.path.join(rotdir, 'ngc2516_rotation_periods.csv')
            )
            df = df[df.Tags == 'gold']

        if BpmRp:
            if 'ngc2516' not in _cls:
                xval = get_interp_BpmRp_from_Teff(df['teff'])
                df['BpmRp_interp'] = xval
                df.to_csv(
                    os.path.join(rotdir, f'curtis19_{_cls}_BpmRpinterp.csv'),
                    index=False
                )
            else:
                xval = (
                    df['phot_bp_mean_mag'] - df['phot_rp_mean_mag'] -
                    AVG_EBpmRp
                )
        else:
            xval = df['teff']

        ykey = 'prot' if 'ngc2516' not in _cls else 'period'

        if 'ngc2516' not in _cls:
            ax.plot(
                xval, df[ykey], c=_col, alpha=1, zorder=z, markersize=s,
                rasterized=False, lw=lw, label=l, marker=m, mew=mew,
                mfc=_col
            )
        else:
            if ngc_core_halo:
                sel = (df.Tags == 'gold') & (df.subcluster == 'core')
                ax.plot(
                    xval[sel],
                    df[sel][ykey],
                    c='C0', alpha=1, zorder=z, markersize=s, rasterized=False,
                    lw=lw, label='NGC2516 core', marker=m, mew=mew, mfc='C0'
                )
                sel = (df.Tags == 'gold') & (df.subcluster == 'halo')
                ax.plot(
                    xval[sel],
                    df[sel][ykey],
                    c='C1', alpha=1, zorder=z, markersize=s, rasterized=False,
                    lw=lw, label='NGC2516 halo', marker=m, mew=mew, mfc='C1'
                )

            else:
                ax.plot(
                    xval, df[ykey], c=_col, alpha=1, zorder=z, markersize=s,
                    rasterized=False, lw=lw, label=l, marker=m, mew=mew,
                    mfc=_col
                )


    _x = TEFF
    if BpmRp:
        print(42*'-')
        print(f'Applying E(Bp-Rp) = {AVG_EBpmRp:.4f}')
        print(42*'-')
        BpmRp_tic268 = 13.4400 - 12.4347
        _x = BpmRp_tic268 - AVG_EBpmRp
    ax.plot(
        _x, P_ROT,
        alpha=1, mew=0.5, zorder=8, label='TOI 1937', markerfacecolor='yellow',
        markersize=18, marker='*', color='black', lw=0
    )

    ax.legend(loc='best', handletextpad=0.1, fontsize='x-small', framealpha=0.7)
    ax.set_ylabel('Rotation Period [days]', fontsize='large')
    if not BpmRp:
        ax.set_xlabel('Effective Temperature [K]', fontsize='large')
        ax.set_xlim((4900, 6600))
    else:
        ax.set_xlabel('(Bp-Rp)$_0$ [mag]', fontsize='large')
        ax.set_xlim((0.5, 1.5))

    ax.set_ylim((0,14))

    format_ax(ax)
    outstr = '_vs_BpmRp' if BpmRp else '_vs_Teff'
    if include_ngc2516:
        outstr += '_include_ngc2516'
    if ngc_core_halo:
        outstr += '_corehalosplit'
    outpath = os.path.join(outdir, f'rotation{outstr}.png')
    savefig(f, outpath)


def plot_skypositions_x_rotn(outdir):

    from earhart.priors import AVG_EBpmRp

    rotdir = os.path.join(DATADIR, 'rotation')
    df = pd.read_csv(
        os.path.join(rotdir, 'ngc2516_rotation_periods.csv')
    )
    BpmRp_0 = (df['phot_bp_mean_mag'] - df['phot_rp_mean_mag'] - AVG_EBpmRp)
    sel = (BpmRp_0 > 0.5) & (BpmRp_0 < 1.2)
    df = df[sel]

    set_style()

    plt.close('all')

    f, ax = plt.subplots(figsize=(4,3))

    xv, yv = 'ra', 'dec'

    sel = (df.subcluster == 'halo')
    ax.scatter(
        df[sel][xv], df[sel][yv], c='lightskyblue', alpha=0.9, zorder=4, s=7,
        rasterized=True, linewidths=0.15, label='Halo', marker='.',
        edgecolors='white'
    )
    sel = (df.subcluster == 'halo') & (df.Tags == 'gold')
    ax.scatter(
        df[sel][xv], df[sel][yv], c='lightskyblue', alpha=0.9, zorder=6, s=7,
        rasterized=True, linewidths=0.15, label='Halo + P$_\mathrm{rot}$', marker='.',
        edgecolors='black'
    )

    sel = (df.subcluster == 'core')
    ax.scatter(
        df[sel][xv], df[sel][yv], c='k', alpha=0.9, zorder=2, s=5, rasterized=True,
        linewidths=0, label='Core', marker='.'
    )

    nbhd_df, cg18_df, kc19_df, target_df = _get_nbhd_dataframes()
    ax.plot(
        target_df[xv], target_df[yv], alpha=1, mew=0.5,
        zorder=3, label='TOI 1937', markerfacecolor='yellow',
        markersize=14, marker='*', color='black', lw=0
    )

    ax.set_xlabel(r'$\alpha$ [deg]')
    ax.set_ylabel(r'$\delta$ [deg]')
    ax.set_xlim([108, 132])
    ax.set_ylim([-76, -45])

    # words = ['Halo', 'Core'][::-1]
    # colors = ['lightskyblue', 'k'][::-1]
    # rainbow_text(0.98, 0.02, words, colors, size='medium', ax=ax)

    # NOTE: hack size of legend markers
    leg = ax.legend(loc='upper right', handletextpad=0.1, fontsize='small',
                    framealpha=0.9)
    leg.legendHandles[0]._sizes = [18]
    leg.legendHandles[1]._sizes = [25]
    leg.legendHandles[2]._sizes = [25]
    leg.legendHandles[3]._sizes = [25]

    ax.set_title('$0.5 < (\mathrm{Bp}-\mathrm{Rp})_0 < 1.2$')

    f.tight_layout(w_pad=2)

    outpath = os.path.join(outdir, 'skypositions_x_rotn.png')
    savefig(f, outpath)


def plot_auto_rotation(outdir, runid, E_BpmRp, core_halo=0):

    set_style()

    from earhart.paths import DATADIR
    rotdir = os.path.join(DATADIR, 'rotation')

    # make plot
    plt.close('all')

    f, ax = plt.subplots(figsize=(4,3))

    classes = ['pleiades', 'praesepe', f'{runid}']
    colors = ['k', 'gray', 'C0']
    zorders = [3, 2, 4]
    markers = ['o', 'x', 'o']
    lws = [0, 0., 0]
    mews= [0.5, 0.5, 0.5]
    ss = [3.0, 6, 3]
    labels = ['Pleaides', 'Praesepe', f'{runid}']

    # plot vals
    for _cls, _col, z, m, l, lw, s, mew in zip(
        classes, colors, zorders, markers, labels, lws, ss, mews
    ):

        if f'{runid}' not in _cls:
            df = pd.read_csv(os.path.join(rotdir, f'curtis19_{_cls}.csv'))

        else:
            df = pd.read_csv(
                os.path.join(rotdir, f'{runid}_rotation_periods.csv')
            )

            # automatic selection criteria for viable rotation periods
            sel = (
                (df.period < 15)
                &
                (df.lspval > 0.08)
                &
                (df.nequal <= 1)
            )
            df = df[sel]

            print(42*'-')
            print(f'Applying E(Bp-Rp) = {E_BpmRp:.4f}')
            print(42*'-')

        if f'{runid}' not in _cls:
            xval = get_interp_BpmRp_from_Teff(df['teff'])
            df['BpmRp_interp'] = xval
            df.to_csv(
                os.path.join(rotdir, f'curtis19_{_cls}_BpmRpinterp.csv'),
                index=False
            )
        else:
            xval = (
                df['phot_bp_mean_mag'] - df['phot_rp_mean_mag'] - E_BpmRp
            )

        ykey = 'prot' if f'{runid}' not in _cls else 'period'

        if core_halo and f'{runid}' in _cls:
            sel = (df.subcluster == 'core')
            ax.plot(
                xval[sel],
                df[sel][ykey],
                c='C0', alpha=1, zorder=z, markersize=s, rasterized=False,
                lw=lw, label=f"{runid.replace('_','')} core", marker=m,
                mew=mew, mfc='C0'
            )

            sel = (df.subcluster == 'halo')
            ax.plot(
                xval[sel],
                df[sel][ykey],
                c='C1', alpha=1, zorder=z, markersize=s, rasterized=False,
                lw=lw, label=f"{runid.replace('_','')} halo", marker=m,
                mew=mew, mfc='C1'
            )

        else:
            ax.plot(
                xval, df[ykey], c=_col, alpha=1, zorder=z, markersize=s,
                rasterized=False, lw=lw, label=l, marker=m, mew=mew, mfc=_col
            )


    ax.legend(loc='best', handletextpad=0.1, fontsize='x-small', framealpha=0.7)
    ax.set_ylabel('Rotation Period [days]', fontsize='large')

    ax.set_xlabel('(Bp-Rp)$_0$ [mag]', fontsize='large')
    ax.set_xlim((0.5, 1.5))

    ax.set_ylim((0,15))

    format_ax(ax)
    outstr = '_vs_BpmRp'
    if core_halo:
        outstr += '_corehalosplit'
    outpath = os.path.join(outdir, f'rotation{outstr}.png')
    savefig(f, outpath)


def plot_galah_dr3_lithium(outdir, vs_rotators=1, corehalosplit=0):

    from earhart.lithium import get_GalahDR3_lithium

    g_tab = get_GalahDR3_lithium()
    scols = ['source_id', 'teff', 'e_teff', 'fe_h', 'e_fe_h', 'flag_fe_h',
             'Li_fe', 'e_Li_fe', 'nr_Li_fe', 'flag_Li_fe', 'ruwe']
    g_dict = {k:np.array(g_tab[k]).byteswap().newbyteorder() for k in scols}
    g_df = pd.DataFrame(g_dict)

    if vs_rotators:
        rotdir = os.path.join(DATADIR, 'rotation')
        rot_df = pd.read_csv(
            os.path.join(rotdir, 'ngc2516_rotation_periods.csv')
        )
        comp_df = rot_df[rot_df.Tags == 'gold']
        print('Comparing vs the "gold" NGC2516 rotators sample (core + halo)...')
    else:
        # nbhd_df, cg18_df, kc19_df, target_df = _get_nbhd_dataframes()
        raise NotImplementedError

    mdf = comp_df.merge(g_df, on='source_id', how='left')

    smdf = mdf[~pd.isnull(mdf.Li_fe)]

    print(f'Number of comparison stars: {len(comp_df)}')
    print(f'Number of comparison stars with finite lithium'
          f'(detection or limit): {len(smdf)}')

    ##########
    # make tha plot 
    ##########

    from earhart.priors import TEFF, P_ROT, AVG_EBpmRp

    set_style()

    plt.close('all')

    f, ax = plt.subplots(figsize=(4,3))

    if not corehalosplit:
        sel = (smdf.flag_Li_fe == 0)
        ax.scatter(
            smdf[sel]['phot_bp_mean_mag'] - smdf[sel]['phot_rp_mean_mag'] - AVG_EBpmRp,
            smdf[sel]['Li_fe'], c='k', alpha=1, zorder=2, s=10, rasterized=False,
            linewidths=0, label='Li detection'
        )
        sel = (smdf.flag_Li_fe == 1)
        ax.plot(
            smdf[sel]['phot_bp_mean_mag'] - smdf[sel]['phot_rp_mean_mag'] - AVG_EBpmRp,
            smdf[sel]['Li_fe'], c='k', alpha=1,
            zorder=3, label='Li limit', ms=10, mfc='white', marker='v', lw=0
        )
    else:
        sel = (smdf.flag_Li_fe == 0) & (smdf.subcluster == 'core')
        ax.scatter(
            smdf[sel]['phot_bp_mean_mag'] - smdf[sel]['phot_rp_mean_mag'] - AVG_EBpmRp,
            smdf[sel]['Li_fe'], c='C0', alpha=1, zorder=2, s=10, rasterized=False,
            linewidths=0, label='Li detection (core)'
        )
        sel = (smdf.flag_Li_fe == 0) & (smdf.subcluster == 'halo')
        ax.scatter(
            smdf[sel]['phot_bp_mean_mag'] - smdf[sel]['phot_rp_mean_mag'] - AVG_EBpmRp,
            smdf[sel]['Li_fe'], c='C1', alpha=1, zorder=2, s=10, rasterized=False,
            linewidths=0, label='Li detection (halo)'
        )

        sel = (smdf.flag_Li_fe == 1) & (smdf.subcluster == 'core')
        ax.plot(
            smdf[sel]['phot_bp_mean_mag'] - smdf[sel]['phot_rp_mean_mag'] - AVG_EBpmRp,
            smdf[sel]['Li_fe'], c='C0', alpha=1,
            zorder=3, label='Li limit (core)', ms=10, mfc='white', marker='v', lw=0
        )
        sel = (smdf.flag_Li_fe == 1) & (smdf.subcluster == 'halo')
        ax.plot(
            smdf[sel]['phot_bp_mean_mag'] - smdf[sel]['phot_rp_mean_mag'] - AVG_EBpmRp,
            smdf[sel]['Li_fe'], c='C1', alpha=1,
            zorder=3, label='Li limit (halo)', ms=10, mfc='white', marker='v', lw=0
        )


    # from timmy.priors import TEFF, LI_EW
    # ax.plot(
    #     TEFF,
    #     LI_EW,
    #     alpha=1, mew=0.5, zorder=8, label='TOI 837', markerfacecolor='yellow',
    #     markersize=18, marker='*', color='black', lw=0
    # )

    ax.legend(loc='best', handletextpad=0.1, fontsize='x-small', framealpha=0.7)
    ax.set_ylabel('[Li/Fe]', fontsize='large')
    ax.set_xlabel('(Bp-Rp)$_0$ [mag]', fontsize='large')

    ax.set_title('Gold rotators, core+halo, x GALAH DR3')

    # ax.set_xlim((4900, 6600))

    format_ax(ax)
    outname = 'galah_dr3_lithium'
    if corehalosplit:
        outname += '_corehalosplit'
    outpath = os.path.join(outdir, f'{outname}.png')
    savefig(f, outpath)



def _make_Randich18_goldrot_xmatch(datapath, vs_rotators=1):
    """
    For every Randich+18 Gaia-ESO star with a spectrum, look for a gold rotator
    match within 1 arcsecond.  If you find it, pull its data. If there are
    multiple, take the closest.
    """

    from earhart.lithium import get_Randich18_NGC2516

    rdf = get_Randich18_NGC2516()

    if vs_rotators:
        rotdir = os.path.join(DATADIR, 'rotation')
        rot_df = pd.read_csv(
            os.path.join(rotdir, 'ngc2516_rotation_periods.csv')
        )
        comp_df = rot_df[rot_df.Tags == 'gold']
        print('Comparing vs the "gold" NGC2516 rotators sample (core + halo)...')
    else:
        # nbhd_df, cg18_df, kc19_df, target_df = _get_nbhd_dataframes()
        raise NotImplementedError

    c_rot = SkyCoord(ra=nparr(comp_df.ra)*u.deg, dec=nparr(comp_df.dec)*u.deg)
    c_r18 = SkyCoord(ra=nparr(rdf._RA)*u.deg, dec=nparr(rdf._DE)*u.deg)

    cutoff_radius = 1*u.arcsec
    has_matchs, match_idxs, match_rows = [], [], []
    for ix, _c in enumerate(c_r18):
        if ix % 100 == 0:
            print(f'{ix}/{len(c_r18)}')
        seps = _c.separation(c_rot)
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

    mdf = pd.concat((left_df.reset_index(), right_df.reset_index()), axis=1)

    print(f'Got {len(mdf)} gold rot matches from {len(rdf)} Randich+18 shots.')

    mdf.to_csv(datapath, index=False)



def plot_randich_lithium(outdir, vs_rotators=1, corehalosplit=0):
    """
    Plot Li EW vs color for Randich+18's Gaia ESO lithium stars, crossmatched
    against the gold rotator sample.

    Somewhat surprisingly, the hit rate is rather poor:
    Got 203 gold rot matches from 796 Randich+18 shots.

    R+18 columns (cf.
    http://vizier.u-strasbg.fr/viz-bin/VizieR-3?-source=J/A%2bA/612/A99&-out.max=50&-out.form=HTML%20Table&-out.add=_r&-out.add=_RAJ,_DEJ&-sort=_r&-oc.form=sexa):
        'CName', 'Cluster', 'Inst', 'Teff', 'e_Teff', 'logg', 'e_logg',
        '__Fe_H_', 'e__Fe_H_', 'l_EWLi', 'EWLi', 'e_EWLi', 'f_EWLi', 'Gamma',
        'e_Gamma', 'RV', 'e_RV', 'MembPA', 'MembPB', 'ID', 'Simbad', '_RA',
        '_DE', 'recno'

    Gold rot columns:
        ['Name', 'Tags', 'source_id', 'period', 'solution_id', 'designation',
        'source_id_2', 'random_index', 'ref_epoch', 'ra', 'ra_error', 'dec',
        'dec_error', 'parallax', 'parallax_error', 'parallax_over_error',
        'pmra', 'pmra_error', 'pmdec', 'pmdec_error', 'ra_dec_corr',
        'ra_parallax_corr', 'ra_pmra_corr', 'ra_pmdec_corr',
        'dec_parallax_corr', 'dec_pmra_corr', 'dec_pmdec_corr',
        'parallax_pmra_corr', 'parallax_pmdec_corr', 'pmra_pmdec_corr',
        'astrometric_n_obs_al', 'astrometric_n_obs_ac',
        'astrometric_n_good_obs_al', 'astrometric_n_bad_obs_al',
        'astrometric_gof_al', 'astrometric_chi2_al',
        'astrometric_excess_noise', 'astrometric_excess_noise_sig',
        'astrometric_params_solved', 'astrometric_primary_flag',
        'astrometric_weight_al', 'astrometric_pseudo_colour',
        'astrometric_pseudo_colour_error', 'mean_varpi_factor_al',
        'astrometric_matched_observations', 'visibility_periods_used',
        'astrometric_sigma5d_max', 'frame_rotator_object_type',
        'matched_observations', 'duplicated_source', 'phot_g_n_obs',
        'phot_g_mean_flux', 'phot_g_mean_flux_error',
        'phot_g_mean_flux_over_error', 'phot_g_mean_mag', 'phot_bp_n_obs',
        'phot_bp_mean_flux', 'phot_bp_mean_flux_error',
        'phot_bp_mean_flux_over_error', 'phot_bp_mean_mag', 'phot_rp_n_obs',
        'phot_rp_mean_flux', 'phot_rp_mean_flux_error',
        'phot_rp_mean_flux_over_error', 'phot_rp_mean_mag',
        'phot_bp_rp_excess_factor', 'phot_proc_mode', 'bp_rp', 'bp_g', 'g_rp',
        'radial_velocity', 'radial_velocity_error', 'rv_nb_transits',
        'rv_template_teff', 'rv_template_logg', 'rv_template_fe_h',
        'phot_variable_flag', 'l', 'b', 'ecl_lon', 'ecl_lat', 'priam_flags',
        'teff_val', 'teff_percentile_lower', 'teff_percentile_upper',
        'a_g_val', 'a_g_percentile_lower', 'a_g_percentile_upper',
        'e_bp_min_rp_val', 'e_bp_min_rp_percentile_lower',
        'e_bp_min_rp_percentile_upper', 'flame_flags', 'radius_val',
        'radius_percentile_lower', 'radius_percentile_upper', 'lum_val',
        'lum_percentile_lower', 'lum_percentile_upper', 'datalink_url',
        'epoch_photometry_url', 'subcluster']
    """

    from earhart.priors import AVG_EBpmRp
    assert abs(AVG_EBpmRp - 0.1343) < 1e-4 # used by KC19

    set_style()

    datapath = os.path.join(DATADIR, 'lithium',
                            'randich_goldrot_xmatch_20201205.csv')

    if not os.path.exists(datapath):
        _make_Randich18_goldrot_xmatch(datapath, vs_rotators=vs_rotators)

    mdf = pd.read_csv(datapath)

    #
    # check crossmatch quality
    #
    plt.close('all')
    f, ax = plt.subplots(figsize=(4,3))

    detections = (mdf.f_EWLi == 0)
    upper_limits = (mdf.f_EWLi == 3)

    print(f'Got {len(mdf[detections])} kinematic X rotation X lithium detections')
    print(f'Got {len(mdf[upper_limits])} kinematic X rotation X lithium upper limits')

    if not corehalosplit:
        ax.plot(
            mdf[detections]['phot_bp_mean_mag'] - mdf[detections]['phot_rp_mean_mag'] - AVG_EBpmRp,
            mdf[detections]['EWLi'],
            c='k', alpha=1, zorder=2, ms=3, mfc='k', marker='o', lw=0, label='Detection'
        )
        ax.plot(
            mdf[upper_limits]['phot_bp_mean_mag'] - mdf[upper_limits]['phot_rp_mean_mag'] - AVG_EBpmRp,
            mdf[upper_limits]['EWLi'],
            c='k', alpha=1, zorder=3, ms=3, mfc='white', marker='v', lw=0, label="Limit"
        )
    else:
        iscore = mdf.subcluster == 'core'
        ishalo = mdf.subcluster == 'halo'

        ax.plot(
            mdf[detections & iscore]['phot_bp_mean_mag'] - mdf[detections & iscore]['phot_rp_mean_mag'] - AVG_EBpmRp,
            mdf[detections & iscore]['EWLi'],
            c='k', alpha=1, zorder=2, ms=3, mfc='C0', marker='o', lw=0, label='Detection + core', mew=0
        )
        ax.plot(
            mdf[upper_limits & iscore]['phot_bp_mean_mag'] - mdf[upper_limits & iscore]['phot_rp_mean_mag'] - AVG_EBpmRp,
            mdf[upper_limits & iscore]['EWLi'],
            c='k', alpha=1, zorder=3, ms=3, mfc='C0', marker='v', lw=0, label="Limit + core", mew=0
        )
        ax.plot(
            mdf[detections & ishalo]['phot_bp_mean_mag'] - mdf[detections & ishalo]['phot_rp_mean_mag'] - AVG_EBpmRp,
            mdf[detections & ishalo]['EWLi'],
            c='k', alpha=1, zorder=2, ms=3, mfc='C1', marker='o', lw=0, label='Detection + halo', mew=0
        )
        ax.plot(
            mdf[upper_limits & ishalo]['phot_bp_mean_mag'] - mdf[upper_limits & ishalo]['phot_rp_mean_mag'] - AVG_EBpmRp,
            mdf[upper_limits & ishalo]['EWLi'],
            c='k', alpha=1, zorder=3, ms=3, mfc='C1', marker='v', lw=0, label="Limit + halo", mew=0
        )

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), handletextpad=0.1)

    ax.set_title('Kinematic $\otimes$ Rotation $\otimes$ Lithium')

    ax.set_ylabel('Li$_{6708}$ EW [m$\mathrm{\AA}$]')
    ax.set_xlabel('(Bp-Rp)$_0$ [mag]')

    format_ax(ax)
    outstr = '_corehalosplit' if corehalosplit else ''
    outpath = os.path.join(outdir,
                           f'randich_lithium_vs_BpmRp_xmatch_goldrot{outstr}.png')
    savefig(f, outpath)


def plot_rotation_X_lithium(outdir, cmapname):
    """
    Plot Prot vs (Bp-Rp)_0, color points by Li EW.

    This is for Randich+18's Gaia ESO lithium stars, crossmatched against the
    gold rotator sample.
    """

    from earhart.priors import AVG_EBpmRp
    assert abs(AVG_EBpmRp - 0.1343) < 1e-4 # used by KC19

    set_style()

    datapath = os.path.join(DATADIR, 'lithium',
                            'randich_goldrot_xmatch_20201205.csv')

    if not os.path.exists(datapath):
        raise NotImplementedError('assumed plot_randich_lithium already run')

    mdf = pd.read_csv(datapath)

    #
    # check crossmatch quality
    #
    plt.close('all')
    f, ax = plt.subplots(figsize=(4.5,3))

    detections = (mdf.f_EWLi == 0)
    upper_limits = (mdf.f_EWLi == 3)

    print(f'Got {len(mdf[detections])} kinematic X rotation X lithium detections')
    print(f'Got {len(mdf[upper_limits])} kinematic X rotation X lithium upper limits')

    # not plotting upper limits b/c there arent any
    assert len(mdf[upper_limits]) == 0

    # color scheme
    if cmapname == 'nipy_spectral':
        cmap = mpl.cm.nipy_spectral
    elif cmapname == 'viridis':
        cmap = mpl.cm.viridis
    bounds = np.arange(0,260,20)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='max')
    cax = ax.scatter(
        mdf[detections]['phot_bp_mean_mag'] - mdf[detections]['phot_rp_mean_mag'] - AVG_EBpmRp,
        mdf[detections]['period'],
        c=nparr(mdf[detections]['EWLi']), alpha=1, zorder=2, s=10, edgecolors='k',
        marker='o', cmap=cmap, norm=norm, linewidths=0.3
    )

    cb = f.colorbar(cax, extend='max')
    cb.set_label("Li$_{6708}$ EW [m$\mathrm{\AA}$]")

    ax.set_title('Kinematic $\otimes$ Rotation $\otimes$ Lithium')

    ax.set_ylabel('Rotation Period [days]')
    ax.set_xlabel('(Bp-Rp)$_0$ [mag]')
    ax.set_xlim((0.5, 1.5))

    format_ax(ax)
    outstr = '_' + cmapname
    outpath = os.path.join(outdir,
                           f'rotation_vs_BpmRp_X_randich18_lithium{outstr}.png')
    savefig(f, outpath)



def _plot_detrending_check(time, flux, trend_flux, flat_flux, outpath):

    plt.close('all')
    set_style()
    fig, axs = plt.subplots(nrows=2, figsize=(4,3), sharex=True)
    t0 = np.nanmin(time)
    axs[0].scatter(time-t0, flux, c='k', zorder=1, s=2)
    axs[0].plot(time-t0, trend_flux, c='C0', zorder=2, lw=1.5)
    axs[1].scatter(time-t0, flat_flux, c='k', s=2)
    axs[1].set_xlabel('Days from start')
    fig.text(-0.01,0.5, 'Relative flux', va='center',
             rotation=90)
    fig.tight_layout(w_pad=0.2)
    savefig(fig, outpath, writepdf=0, dpi=300)
