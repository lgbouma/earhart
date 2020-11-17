"""
Plots:
    plot_full_kinematics
    plot_TIC268_nbhd_small
    plot_hr
    plot_rotation

Helpers:
    _get_nbhd_dataframes
"""
import os, corner, pickle
from glob import glob
from datetime import datetime
import numpy as np, matplotlib.pyplot as plt, pandas as pd, pymc3 as pm

from astropy import units as u, constants as const
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


def plot_rotation(outdir, BpmRp=0):

    set_style()

    from earhart.paths import DATADIR
    rotdir = os.path.join(DATADIR, 'rotation')

    # make plot
    plt.close('all')

    f, ax = plt.subplots(figsize=(4,3))

    classes = ['pleiades', 'praesepe']
    colors = ['k', 'gray']
    zorders = [3, 2]
    markers = ['o', 'x']
    lws = [0, 0.]
    mews= [0.5, 0.5]
    ss = [3.0, 6]
    labels = ['Pleaides', 'Praesepe']

    # plot vals
    for _cls, _col, z, m, l, lw, s, mew in zip(
        classes, colors, zorders, markers, labels, lws, ss, mews
    ):
        df = pd.read_csv(os.path.join(rotdir, f'curtis19_{_cls}.csv'))

        xval = df['teff']

        if BpmRp:
            xval = get_interp_BpmRp_from_Teff(df['teff'])
            df['BpmRp_interp'] = xval
            df.to_csv(
                os.path.join(rotdir, f'curtis19_{_cls}_BpmRpinterp.csv'),
                index=False
            )

        ax.plot(
            xval, df['prot'], c=_col, alpha=1, zorder=z, markersize=s,
            rasterized=False, lw=lw, label=l, marker=m, mew=mew,
            mfc=_col
        )

    from earhart.priors import TEFF, P_ROT, AVG_EBpmRp

    _x = TEFF
    if BpmRp:
        print(42*'-')
        print(f'Applying E(Bp-Rp) = {AVG_EBpmRp:.2f}')
        print(42*'-')
        _x = get_interp_BpmRp_from_Teff(TEFF) - AVG_EBpmRp
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
    outpath = os.path.join(outdir, f'rotation{outstr}.png')
    savefig(f, outpath)



