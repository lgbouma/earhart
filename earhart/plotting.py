"""
Contents:
    Kinematics / Gaia / HR:
        plot_full_kinematics
        plot_TIC268_nbhd_small
        plot_ngc2516_corehalo_3panel
        plot_hr
    Rotation / TESS:
        plot_rotation
        plot_auto_rotation
        plot_skypositions_x_rotn
        plot_rotation_X_RUWE
        plot_full_kinematics_X_rotation
        plot_physical_X_rotation
    Lithium:
        plot_randich_lithium
        plot_galah_dr3_lithium
        plot_rotation_X_lithium
    Other:
        plot_gaia_rv_scatter_vs_brightness
        plot_ruwe_vs_apparentmag
        plot_edr3_blending_vs_apparentmag
        plot_bisector_span_vs_RV
        plot_backintegration_ngc2516
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
from astropy.table import Table

import matplotlib.patheffects as pe
from matplotlib.ticker import MaxNLocator

from aesthetic.plot import savefig, format_ax, set_style

from astrobase.services.identifiers import gaiadr2_to_tic
from cdips.utils.gaiaqueries import (
    given_source_ids_get_gaia_data
)
from cdips.utils.tapqueries import given_source_ids_get_tic8_data
from cdips.utils.plotutils import rainbow_text
from cdips.utils.mamajek import get_interp_BpmRp_from_Teff

from earhart.paths import DATADIR, RESULTSDIR
from earhart.helpers import (
    get_gaia_basedata, get_autorotation_dataframe
)
from earhart.physicalpositions import (
    given_gaia_df_get_icrs_arr, calc_dist
)

def plot_TIC268_nbhd_small(outdir=RESULTSDIR):

    basedata = 'bright'
    nbhd_df, cg18_df, kc19_df, trgt_df = get_gaia_basedata(basedata)

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
        trgt_df[xv], trgt_df[yv], alpha=1, mew=0.5,
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
        get_xval(trgt_df), get_yval(trgt_df), alpha=1, mew=0.5,
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


def plot_full_kinematics(outdir, basedata='bright', show1937=1,
                         galacticframe=0):

    nbhd_df, cg18_df, kc19_df, trgt_df = get_gaia_basedata(basedata)

    if galacticframe:
        c_nbhd = SkyCoord(ra=nparr(nbhd_df.ra)*u.deg, dec=nparr(nbhd_df.dec)*u.deg)
        nbhd_df['l'] = c_nbhd.galactic.l.value
        nbhd_df['b'] = c_nbhd.galactic.b.value

    plt.close('all')

    rvkey = (
        'radial_velocity' if 'edr3' not in basedata else 'dr2_radial_velocity'
    )

    if galacticframe:
        xkey, ykey = 'l', 'b'
        xl, yl = r'$l$ [deg]', r'$b$ [deg]'
    else:
        xkey, ykey = 'ra', 'dec'
        xl, yl = r'$\alpha$ [deg]', r'$\delta$ [deg]'

    params = [xkey, ykey, 'parallax', 'pmra', 'pmdec', rvkey]
    # whether to limit axis by 5/95th percetile
    qlimd = {
        xkey: 0, ykey: 0, 'parallax': 0, 'pmra': 1, 'pmdec': 1, rvkey: 1
    }
    # whether to limit axis by 99th percentile
    nnlimd = {
        xkey: 1, ykey: 1, 'parallax': 1, 'pmra': 0, 'pmdec': 0, rvkey: 0
    }
    ldict = {
        xkey: xl, ykey: yl,
        'parallax': r'$\pi$ [mas]', 'pmra': r"$\mu_{{\alpha'}}$ [mas/yr]",
        'pmdec':  r'$\mu_{{\delta}}$ [mas/yr]', rvkey: 'RV [km/s]'
    }


    nparams = len(params)
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
            if show1937:
                axs[i,j].plot(
                    trgt_df[xv], trgt_df[yv], alpha=1, mew=0.5,
                    zorder=8, label='TOI 1937', markerfacecolor='yellow',
                    markersize=14, marker='*', color='black', lw=0
                )

            # set the axis limits as needed
            if qlimd[xv]:
                xlim = (np.nanpercentile(nbhd_df[xv], 5),
                        np.nanpercentile(nbhd_df[xv], 95))
                axs[i,j].set_xlim(xlim)
            if qlimd[yv]:
                ylim = (np.nanpercentile(nbhd_df[yv], 5),
                        np.nanpercentile(nbhd_df[yv], 95))
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
    if show1937:
        leg.legendHandles[3]._sizes = [20]

    for ax in axs.flatten():
        format_ax(ax)

    f.tight_layout(h_pad=0.05, w_pad=0.05)

    s = ''
    s += f'_{basedata}'
    if show1937:
        s += f'_show1937'
    if galacticframe:
        s += f'_galactic'
    else:
        s += f'_icrs'
    outpath = os.path.join(outdir, f'full_kinematics{s}.png')
    savefig(f, outpath)


def plot_gaia_rv_scatter_vs_brightness(outdir, basedata='fullfaint'):

    """
    basedata (str): any of ['bright', 'extinctioncorrected', 'fullfaint',
    'fullfaint_edr3'],
    """

    set_style()

    nbhd_df, cg18_df, kc19_df, trgt_df = get_gaia_basedata(basedata)

    plt.close('all')

    f, ax = plt.subplots(figsize=(4,3))

    ykey = 'radial_velocity_error' if 'edr3' not in basedata else 'dr2_radial_velocity_error'
    get_yval = (
        lambda _df: np.array(
            _df[ykey]
        )
    )
    get_xval = (
        lambda _df: np.array(
            _df['phot_g_mean_mag']
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
        get_xval(trgt_df), get_yval(trgt_df), alpha=1, mew=0.5,
        zorder=8, label='TOI 1937', markerfacecolor='yellow',
        markersize=10, marker='*', color='black', lw=0
    )

    leg = ax.legend(loc='upper left', handletextpad=0.1, fontsize='x-small',
                    framealpha=0.9)
    # NOTE: hack size of legend markers
    leg.legendHandles[0]._sizes = [18]
    leg.legendHandles[1]._sizes = [25]
    leg.legendHandles[2]._sizes = [25]
    leg.legendHandles[3]._sizes = [25]

    ax.set_xlabel('G [mag]', fontsize='large')
    ax.set_ylabel('GDR2 RV error [km$\,$s$^{-1}$]', fontsize='large')
    ax.set_yscale('log')
    ax.set_xlim([7,14.5])

    s = ''
    s += f'_{basedata}'
    outpath = os.path.join(outdir, f'gaia_rv_scatter_vs_brightness{s}.png')

    savefig(f, outpath, dpi=400)


def plot_ruwe_vs_apparentmag(outdir, basedata='fullfaint', smallylim=False):

    """
    basedata (str): any of ['extinctioncorrected', 'fullfaint_edr3'],
    """

    set_style()

    if not basedata == 'fullfaint_edr3':
        raise NotImplementedError('only EDR3 has ruwe built in')
    nbhd_df, cg18_df, kc19_df, trgt_df = get_gaia_basedata(basedata)

    plt.close('all')

    f, ax = plt.subplots(figsize=(4,3))

    ykey = 'ruwe'
    get_yval = (
        lambda _df: np.array(
            _df[ykey]
        )
    )
    get_xval = (
        lambda _df: np.array(
            _df['phot_g_mean_mag']
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
        get_xval(trgt_df), get_yval(trgt_df), alpha=1, mew=0.5,
        zorder=8, label='TOI 1937', markerfacecolor='yellow',
        markersize=10, marker='*', color='black', lw=0
    )

    leg = ax.legend(loc='upper left', handletextpad=0.1, fontsize='x-small',
                    framealpha=0.9)
    # NOTE: hack size of legend markers
    leg.legendHandles[0]._sizes = [18]
    leg.legendHandles[1]._sizes = [25]
    leg.legendHandles[2]._sizes = [25]
    leg.legendHandles[3]._sizes = [25]

    ax.set_xlabel('G [mag]', fontsize='large')
    ax.set_ylabel('EDR3 RUWE', fontsize='large')
    ax.set_yscale('log')
    ax.set_xlim([7,14.5])

    if smallylim:
        ax.set_yscale('linear')
        ax.set_ylim([0.5, 2])

    s = ''
    s += f'_{basedata}'
    if smallylim:
        s += f'_smallylim'
    outpath = os.path.join(outdir, f'ruwe_vs_apparentmag{s}.png')

    savefig(f, outpath, dpi=400)


def plot_edr3_blending_vs_apparentmag(outdir, basedata='fullfaint', num=None):

    """
    Look at how the blending changes vs apparent mag, to get context for the
    blending experienced by TOI 1937B.

    basedata (str): any of ['extinctioncorrected', 'fullfaint_edr3'],

    num = 'phot_bp_n_blended_transits', phot_bp_n_contaminated_transits', also
    rp ok.
    """

    set_style()

    if not basedata in ['fullfaint_edr3']:
        raise NotImplementedError('only EDR3 has n_blended_transits built in')
    nbhd_df, cg18_df, kc19_df, trgt_df = get_gaia_basedata(basedata)

    comp_arr = np.array([5489726768531118848]).astype(np.int64)
    runid = 'toi1937_companion_edr3'
    gaia_datarelease = 'gaiaedr3'
    comp_df = given_source_ids_get_gaia_data(
        comp_arr, runid, n_max=2, overwrite=False,
        enforce_all_sourceids_viable=True, gaia_datarelease=gaia_datarelease
    )

    ##########################################

    plt.close('all')

    f, ax = plt.subplots(figsize=(4,3))

    denom = 'phot_bp_n_obs' if '_bp_' in num else 'phot_rp_n_obs'
    get_yval = (
        lambda _df: np.array(
            _df[num] / _df[denom]
        )
    )
    get_xval = (
        lambda _df: np.array(
            _df['phot_g_mean_mag']
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
        get_xval(trgt_df), get_yval(trgt_df), alpha=1, mew=0.5,
        zorder=8, label='TOI 1937A', markerfacecolor='yellow',
        markersize=10, marker='*', color='black', lw=0
    )
    ax.plot(
        get_xval(comp_df), get_yval(comp_df), alpha=1, mew=0.5,
        zorder=8, label='TOI 1937B', markerfacecolor='yellow',
        markersize=10, marker='*', color='black', lw=0
    )

    leg = ax.legend(loc='upper left', handletextpad=0.1, fontsize='x-small',
                    framealpha=0.9)
    # NOTE: hack size of legend markers
    leg.legendHandles[0]._sizes = [18]
    leg.legendHandles[1]._sizes = [25]
    leg.legendHandles[2]._sizes = [25]
    leg.legendHandles[3]._sizes = [25]

    ax.set_xlabel('G [mag]')
    _c = 'Bp' if '_bp_' in num else 'Rp'
    _t = 'blended' if 'blended' in num else 'contaminated'
    ystr = 'N$_{\mathrm{'+_t+'}}^{\mathrm{'+_c+'}}$' + '/' + 'N$_{\mathrm{obs}}^{\mathrm{'+_c+'}}$'
    ax.set_ylabel(ystr)
    ax.set_xlim([7,19])

    s = ''
    s += f'_{basedata}'
    outpath = os.path.join(outdir, f'{num}_by_{denom}_vs_apparentmag{s}.png')

    savefig(f, outpath, dpi=400)


def plot_hr(outdir, isochrone=None, color0='phot_bp_mean_mag',
            basedata='fullfaint', highlight_companion=0, colorhalobyglat=0,
            show1937=1):
    """
    basedata (str): any of ['bright', 'extinctioncorrected', 'fullfaint',
    'fullfaint_edr3'], where each defines a different set of neighborhood /
    core / halo. The default is the "fullfaint" sample, which extends down to
    whatever cutoffs CG18 and KC19 used for members, and to even fainter for
    neighbors.  The "bright" sample uses the cutoffs from Bouma+19 (CDIPS-I),
    i.e., G_Rp<16, to require that a TESS light curve exists.
    """

    set_style()

    nbhd_df, cg18_df, kc19_df, trgt_df = get_gaia_basedata(basedata)

    comp_arr = np.array([5489726768531118848]).astype(np.int64)
    runid = (
        'toi1937_companion' if 'edr3' not in basedata
        else 'toi1937_companion_edr3'
    )
    gaia_datarelease = 'gaiadr2' if 'edr3' not in basedata else 'gaiaedr3'
    comp_df = given_source_ids_get_gaia_data(
        comp_arr, runid, n_max=2, overwrite=False,
        enforce_all_sourceids_viable=True, gaia_datarelease=gaia_datarelease
    )

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
    if 'denis' not in basedata:
        get_xval = (
            lambda _df: np.array(
                _df[color0] - _df['phot_rp_mean_mag']
            )
        )
        get_comp_xval = get_xval
    else:
        get_xval = (
            lambda _df: np.array(
                _df['phot_g_mean_mag'] - _df['Imag']
            )
        )
        # see /doc/20201118_STAR_INFO.txt. this is the DENIS Imag for primary,
        # plus the SOAR deltaI measured.
        comp_Imag = 12.26 + 4.3 # TODO: what's the uncertainty on this dMag?
        get_comp_xval = (
            lambda _df: np.array(
                _df['phot_g_mean_mag'] - comp_Imag
            )
        )


    if not colorhalobyglat:
        ax.scatter(
            get_xval(nbhd_df), get_yval(nbhd_df), c='gray', alpha=0.2, zorder=2,
            s=5, rasterized=True, linewidths=0, label='Field', marker='.'
        )
        ax.scatter(
            get_xval(kc19_df), get_yval(kc19_df), c='lightskyblue', alpha=1,
            zorder=3, s=5, rasterized=True, linewidths=0.15, label='Halo',
            marker='.', edgecolors='k'
        )

    else:
        glatkey = 'b'
        cmap = mpl.cm.viridis

        #bounds = np.arange(-25,-7.5,2.5)
        #bounds = np.arange(-20,-8, 1)
        bounds = np.arange(-20,-11, 1)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)#, extend='max')

        cax = ax.scatter(
            get_xval(kc19_df), get_yval(kc19_df), c=nparr(kc19_df[glatkey]),
            alpha=1, zorder=3, s=2, rasterized=True, linewidths=0.1,
            label='Halo', marker='o', edgecolors='none', cmap=cmap, norm=norm,
        )

        cb = f.colorbar(cax, extend='max')
        cb.set_label("Galactic lat [deg]")


    if not colorhalobyglat:
        ax.scatter(
            get_xval(cg18_df), get_yval(cg18_df), c='k', alpha=0.9,
            zorder=4, s=5, rasterized=True, linewidths=0, label='Core', marker='.'
        )
        _l = 'TOI 1937' if not highlight_companion else 'TOI 1937A'
        if show1937:
            ax.plot(
                get_xval(trgt_df), get_yval(trgt_df), alpha=1, mew=0.5,
                zorder=8, label=_l, markerfacecolor='yellow',
                markersize=10, marker='*', color='black', lw=0
            )

    if highlight_companion and show1937:
        # NOTE: NOTE: use the parallax from TOI 1937A, because it has higher
        # S/N
        _yval = np.array(
            comp_df['phot_g_mean_mag'] +
            5*np.log10(trgt_df['parallax']/1e3)
            + 5
        )

        ax.plot(
            get_comp_xval(comp_df), get_yval(comp_df), alpha=1, mew=0.5,
            zorder=8, label='TOI 1937B', markerfacecolor='yellow',
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


    if not colorhalobyglat:
        leg = ax.legend(loc='upper right', handletextpad=0.1, fontsize='x-small',
                        framealpha=0.9)
        # # NOTE: hack size of legend markers
        if show1937:
            leg.legendHandles[0]._sizes = [18]
            leg.legendHandles[1]._sizes = [25]
            leg.legendHandles[2]._sizes = [25]
            leg.legendHandles[3]._sizes = [25]
        else:
            leg.legendHandles[0]._sizes = [25]
            leg.legendHandles[1]._sizes = [25]
            leg.legendHandles[2]._sizes = [25]


    ax.set_ylabel('Absolute G [mag]', fontsize='large')
    if color0 == 'phot_bp_mean_mag':
        ax.set_xlabel('Bp - Rp [mag]', fontsize='large')
        c0s = '_Bp_m_Rp'
    elif color0 == 'phot_g_mean_mag':
        ax.set_xlabel('G - Rp [mag]', fontsize='large')
        c0s = '_G_m_Rp'
    elif color0 is None:
        ax.set_xlabel('G$_{\mathrm{EDR3}}$ - I$_\mathrm{DENIS}$ [mag]', fontsize='large')
        c0s = '_G_m_Idenis'
        ax.set_xlim([-0.7, 2.3])
    else:
        raise NotImplementedError

    ylim = ax.get_ylim()
    ax.set_ylim((max(ylim),min(ylim)))
    if basedata == 'denis_fullfaint_edr3':
        ax.set_ylim([12.8, -1.0])

    if basedata == 'fullfaint_edr3' and color0 == 'phot_bp_mean_mag':
        ax.set_xlim([-0.46, 3.54])

    format_ax(ax)
    if not isochrone:
        s = ''
    else:
        s = '_'+isochrone
    if colorhalobyglat:
        s = '_colorhalobyglat'
    if highlight_companion:
        c0s += '_highlight_companion'
    if show1937:
        c0s += '_show1937'
    c0s += f'_{basedata}'
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

    basedata = 'bright'
    nbhd_df, cg18_df, kc19_df, trgt_df = get_gaia_basedata(basedata)

    ax.plot(
        trgt_df[xv], trgt_df[yv], alpha=1, mew=0.5,
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


def plot_auto_rotation(outdir, runid, E_BpmRp, core_halo=0, yscale='linear'):
    """
    Plot rotation periods that satisfy:

        (df.period < 15)
        &
        (df.lspval > 0.08)
        &
        (df.nequal <= 1)

    (At most one equal-brightness star in the aperture that was used)
    """

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
    _s = 3 if runid != 'VelaOB2' else 1.2
    ss = [3.0, 6, _s]
    labels = ['Pleaides', 'Praesepe', f'{runid}']

    # plot vals
    for _cls, _col, z, m, l, lw, s, mew in zip(
        classes, colors, zorders, markers, labels, lws, ss, mews
    ):

        if f'{runid}' not in _cls:
            df = pd.read_csv(os.path.join(rotdir, f'curtis19_{_cls}.csv'))

        else:
            df = get_autorotation_dataframe(runid)

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


    loc = 'best' if yscale == 'linear' else 'lower right'
    ax.legend(loc=loc, handletextpad=0.1, fontsize='x-small', framealpha=0.7)
    ax.set_ylabel('Rotation Period [days]', fontsize='large')

    ax.set_xlabel('(Bp-Rp)$_0$ [mag]', fontsize='large')
    ax.set_xlim((0.25, 2.0))

    if yscale == 'linear':
        ax.set_ylim((0,15))
    elif yscale == 'log':
        ax.set_ylim((0.05,15))
    else:
        raise NotImplementedError
    ax.set_yscale(yscale)

    format_ax(ax)
    outstr = '_vs_BpmRp'
    if core_halo:
        outstr += '_corehalosplit'
    outstr += f'_{yscale}'
    outpath = os.path.join(outdir, f'{runid}_rotation{outstr}.png')
    savefig(f, outpath)


def plot_galah_dr3_lithium(outdir, vs_rotators=1, corehalosplit=0):

    from earhart.lithium import get_GalahDR3_lithium

    g_tab = get_GalahDR3_lithium(defaultflags=1)
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
        basedata = 'fullfaint'
        nbhd_df, cg18_df, kc19_df, trgt_df = get_gaia_basedata(basedata)
        cg18_df['subcluster'] = 'core'
        kc19_df['subcluster'] = 'halo'
        comp_df = pd.concat((cg18_df, kc19_df))
        print('Comparing vs the "fullfaint" kinematic NGC2516 rotators sample (core + halo)...')
        assert type(comp_df.source_id.iloc[0]) == np.int64

    mdf = comp_df.merge(g_df, on='source_id', how='inner')

    if not vs_rotators:
        outname = 'kinematic_X_galah_dr3'
        outpath = os.path.join(outdir, f'{outname}.csv')
        if not os.path.exists(outpath):
            mdf.to_csv(outpath, index=False)
            print(f'Made {outpath} with {len(mdf)} entries.')

    smdf = mdf
    smdf = mdf[~pd.isnull(mdf.Li_fe)]

    print(f'Number of comparison stars: {len(comp_df)}')
    print(f'Number of comparison stars w/ Galah DR3 matches '
          f'{len(mdf)}')
    print(f'Number of comparison stars w/ Galah DR3 matches in core '
          f'{len(mdf[mdf.subcluster == "core"])}')
    print(f'Number of comparison stars w/ Galah DR3 matches in halo '
          f'{len(mdf[mdf.subcluster == "halo"])}')

    print(f'Number of comparison stars w/ Galah DR3 matches and finite lithium '
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


def plot_randich_lithium(outdir, vs_rotators=1, corehalosplit=0):
    """
    Plot Li EW vs color for Randich+18's Gaia ESO lithium stars, crossmatched
    against either the gold rotator sample, or just the "fullfaint kinematic"
    sample.

    For gold rotator, the hit rate is rather poor:
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

    if vs_rotators:
        datapath = os.path.join(DATADIR, 'lithium',
                                'randich_goldrot_xmatch_20201205.csv')
    else:
        datapath = os.path.join(DATADIR, 'lithium',
                                'randich_fullfaintkinematic_xmatch_20210128.csv')

    if not os.path.exists(datapath):
        from earhart.lithium import _make_Randich18_xmatch
        _make_Randich18_xmatch(datapath, vs_rotators=vs_rotators)

    mdf = pd.read_csv(datapath)

    #
    # check crossmatch quality
    #
    plt.close('all')
    f, ax = plt.subplots(figsize=(4,3))

    detections = (mdf.f_EWLi == 0)
    upper_limits = (mdf.f_EWLi == 3)

    if vs_rotators:
        print(f'Got {len(mdf[detections])} kinematic X rotation X lithium detections')
        print(f'Got {len(mdf[upper_limits])} kinematic X rotation X lithium upper limits')
    else:
        print(f'Got {len(mdf[mdf.subcluster == "core"])} kinematic X lithium entries in core')
        print(f'Got {len(mdf[mdf.subcluster == "halo"])} kinematic X lithium entries in halo')
        print(f'Got {len(mdf[detections])} kinematic X lithium detections')
        print(f'Got {len(mdf[upper_limits])} kinematic X lithium upper limits')

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

    if vs_rotators:
        ax.set_title('Kinematic $\otimes$ Rotation $\otimes$ Lithium')
    else:
        ax.set_title('Kinematic $\otimes$ R+18 Lithium')

    ax.set_ylabel('Li$_{6708}$ EW [m$\mathrm{\AA}$]')
    ax.set_xlabel('(Bp-Rp)$_0$ [mag]')

    format_ax(ax)
    outstr = '_corehalosplit' if corehalosplit else ''
    xmstr = 'goldrot' if vs_rotators else 'fullfaintkinematic'
    outpath = os.path.join(outdir,
                           f'randich_lithium_vs_BpmRp_xmatch_{xmstr}{outstr}.png')
    savefig(f, outpath)

    if not vs_rotators:
        plt.close('all')
        f,ax = plt.subplots(figsize=(4,3))
        ax.plot(
            mdf[detections]['phot_bp_mean_mag'] - mdf[detections]['phot_rp_mean_mag'] - AVG_EBpmRp,
            mdf[detections]['Teff'],
            c='k', alpha=1, zorder=2, ms=3, mfc='k', marker='o', lw=0
        )
        outstr += 'colorvsteff_sanitycheck'
        outpath = os.path.join(outdir,
                               f'randich_lithium_vs_BpmRp_xmatch_{xmstr}{outstr}.png')
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



def plot_rotation_X_RUWE(outdir, cmapname, vs_rotators=1,
                         basedata='fullfaint_edr3', emph_1937=0):
    """
    Plot Prot vs (Bp-Rp)_0, color points by RUWE (i.e., Gaia EDR3 matched).

    This is for the "fullfaint_edr3" dataset, crossmatched against the
    gold rotator sample.
    """

    from earhart.priors import AVG_EBpmRp, P_ROT
    assert abs(AVG_EBpmRp - 0.1343) < 1e-4 # used by KC19

    set_style()

    assert basedata == 'fullfaint_edr3'
    nbhd_df, cg18_df, kc19_df, trgt_df = get_gaia_basedata(basedata)

    if vs_rotators:
        rotdir = os.path.join(DATADIR, 'rotation')
        rot_df = pd.read_csv(
            os.path.join(rotdir, 'ngc2516_rotation_periods.csv')
        )
        comp_df = rot_df[rot_df.Tags == 'gold']
        print('Comparing vs the "gold" NGC2516 rotators sample (core + halo)...')

        selcols = ['Name', 'period', 'Tags', 'source_id', 'subcluster']
        s_comp_df = comp_df[selcols]

    # merge, noting that the "ngc2516_rotation_periods.csv" measurements were
    # done based on the DR2 source_id list, and in this plot the basedata are
    # from EDR3 (so we use the DR2<->EDR3 crossmatch from
    # _get_fullfaint_edr3_dataframes)
    full_df = pd.concat((cg18_df,kc19_df))
    assert len(full_df) == len(cg18_df) + len(kc19_df)
    mdf = s_comp_df.merge(full_df, left_on='source_id',
                          right_on='dr2_source_id', how='left')
    assert len(mdf) == len(comp_df)

    #
    # check crossmatch quality
    #
    plt.close('all')
    f, ax = plt.subplots(figsize=(4.5,3))

    # color scheme
    if cmapname == 'nipy_spectral':
        cmap = mpl.cm.nipy_spectral
    elif cmapname == 'viridis':
        cmap = mpl.cm.viridis
    bounds = np.arange(0.9,1.2+0.1, 0.1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')
    if not emph_1937:
        cax = ax.scatter(
            mdf['phot_bp_mean_mag'] - mdf['phot_rp_mean_mag'] - AVG_EBpmRp,
            mdf['period'],
            c=nparr(mdf['ruwe']), alpha=1, zorder=2, s=10, edgecolors='k',
            marker='o', cmap=cmap, norm=norm, linewidths=0.3
        )
    else:
        _s = 5489726768531119616
        sel = (mdf.dr2_source_id == _s)
        cax = ax.scatter(
            mdf[~sel]['phot_bp_mean_mag'] - mdf[~sel]['phot_rp_mean_mag'] - AVG_EBpmRp,
            mdf[~sel]['period'],
            c=nparr(mdf[~sel]['ruwe']), alpha=1, zorder=2, s=10, edgecolors='k',
            marker='o', cmap=cmap, norm=norm, linewidths=0.3
        )


    cb = f.colorbar(cax, extend='both')
    cb.set_label("RUWE")

    if emph_1937:
        print(42*'-')
        print(f'Applying E(Bp-Rp) = {AVG_EBpmRp:.4f}')
        print(42*'-')

        ax.scatter(
                mdf[sel]['phot_bp_mean_mag'] - mdf[sel]['phot_rp_mean_mag'] - AVG_EBpmRp,
                P_ROT,
                c=nparr(mdf[sel]['ruwe']), alpha=1, zorder=2, s=40, edgecolors='k',
                marker='*', cmap=cmap, norm=norm, linewidths=0.3,
                label='TOI 1937'
            )

        ax.legend(loc='upper left', handletextpad=0.05, fontsize='x-small', framealpha=0.)

    ax.set_title('Kinematic $\otimes$ Rotation')

    ax.set_ylabel('Rotation Period [days]')
    ax.set_xlabel('(Bp-Rp)$_0$ [mag]')
    ax.set_xlim((0.5, 1.5))

    format_ax(ax)
    outstr = '_' + cmapname
    if emph_1937:
        outstr += '_emph1937'
    outpath = os.path.join(outdir,
                           f'rotation_vs_BpmRp_X_RUWE{outstr}.png')
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


def plot_bisector_span_vs_RV(outdir, which='Gummi'):
    """
    'Gummi' or 'Hartman'
    """

    # get data
    if which == 'Hartman':
        bisectorpath = os.path.join(
            DATADIR, 'spectra', 'PFS_bisector_spans_Hartman_20201211',
            'TOI1937.PFS_bs.txt'
        )
        bdf = pd.read_csv(bisectorpath, delim_whitespace=True)
        bis_key = 'BS[m/s]'
        ebis_key = 'eBS[m/s]'
        factor = 1
    elif which == 'Gummi':
        bisectorpath = os.path.join(
            DATADIR, 'spectra', 'PFS_bisector_spans_Gummi_20201223',
            '20201223_toi_1937_ccfs_bis.csv'
        )
        bdf = pd.read_csv(bisectorpath)
        bdf = bdf.rename({'ob_name':'Spectrum', 'rv':'rv_gummi'}, axis=1)
        bis_key = 'bis'
        ebis_key = 'e_bis'
        factor = 1000

    rvpath = os.path.join(
        RESULTSDIR, '20201110_butler_PFS_results', 'HD268301217_PFSBIN.vels'
    )
    rvdf = pd.read_csv(rvpath, delim_whitespace=True)

    mdf = rvdf.merge(bdf, how='left', on='Spectrum')

    # make plot

    set_style()

    plt.close('all')

    f, ax = plt.subplots(figsize=(4,3))

    ax.errorbar(
        factor*mdf[bis_key], mdf['rv'], yerr=mdf['rv_err'], xerr=factor*mdf[ebis_key],
        ls='none', color='k', elinewidth=1, capsize=1
    )

    ax.set_xlabel('BS [m$\,$s$^{-1}$]', fontsize='large')
    ax.set_ylabel('RV [m$\,$s$^{-1}$]', fontsize='large')

    s = '_'+which
    outpath = os.path.join(outdir, f'bisector_span_vs_RV{s}.png')

    savefig(f, outpath, dpi=400)


def plot_backintegration_ngc2516(basedata, fix_rvs=0):
    """
    Back-integrate the orbits of TOI 1937, the NGC 2516 cluster itself, and the
    CG18 and KC19 members, to see what happens. I back-integrate to 100 Myr,
    over 2e3 steps.

    We require 6d positions + kinematics (no NaN RVs allowed) for the core,
    halo, and field comparison samples. This implies a strong cut on the
    brightness of stars, and it also implies more precise parallax, position,
    and proper motion measurements.

    For the neighborhood/field samples, we also require S/N>10 on the parallax
    measurement --- otherwise we get stars with negative parallaxes, which lead
    to erroneous distance measurements when defining the orbits and doing the
    back-integration.

    `fix_rvs` is an option that sets all the RVs (except that of TOI1937) to
    the mean cluster RV.

    If this plot gets included in the paper, key citations will include a)
    Price-Whelan's `gala` code; b) Bovy 2015, who worked out the model for the
    galactic potential that is used in thisc calculation.

    This function makes three plots:
        check_backintegrate
        toi1937_to_ngc2516_mean_distance
        core_halo_to_ngc2516_separation
    """

    from earhart.backintegrate import backintegrate

    nbhd_df, cg18_df, kc19_df, trgt_df = get_gaia_basedata(basedata)

    rvkey = 'dr2_radial_velocity' if 'edr3' in basedata else 'radial_velocity'
    getcols = ['ra', 'dec', 'parallax', 'pmra', 'pmdec', rvkey]

    #
    # First, calculate separation between toi1937 and ngc2516 core median.
    # Then, do the same for the entire cluster, and the ngc2516 core median.
    n_steps = int(2e3)
    dt = -0.05*u.Myr

    # require 6d positions + kinematics (no NaN RVs allowed)
    # for neighborhood/field, also require S/N on parallax measurement
    s_cg18_df = cg18_df[getcols].dropna(axis=0)
    s_kc19_df = kc19_df[getcols].dropna(axis=0)
    sel_nbhd = (nbhd_df.parallax / nbhd_df.parallax_error) > 10
    nbhd_df = nbhd_df[sel_nbhd]
    s_nbhd_df = nbhd_df[getcols].dropna(axis=0)

    icrs_median_ngc2516_df = pd.DataFrame(s_cg18_df.median()).T
    icrs_toi1937_df = trgt_df[getcols]
    mdf = pd.concat((icrs_median_ngc2516_df, icrs_toi1937_df))

    s = '' # formatting string for output
    if fix_rvs:
        med_rv = float(icrs_median_ngc2516_df[rvkey])
        # mdf[rvkey] = med_rv
        s_cg18_df[rvkey] = med_rv
        s_kc19_df[rvkey] = med_rv
        s_nbhd_df[rvkey] = med_rv
        s += '_fix_rvs'

    icrs_arr = given_gaia_df_get_icrs_arr(mdf, zero_rv=0)
    orbits = backintegrate(icrs_arr, dt=dt, n_steps=n_steps)
    d = calc_dist(orbits.x[:,0], orbits.y[:,0], orbits.z[:,0],
                  orbits.x[:,1], orbits.y[:,1], orbits.z[:,1])

    icrs_arr_cg18 = given_gaia_df_get_icrs_arr(s_cg18_df, zero_rv=0)
    icrs_arr_kc19 = given_gaia_df_get_icrs_arr(s_kc19_df, zero_rv=0)
    icrs_arr_nbhd = given_gaia_df_get_icrs_arr(s_nbhd_df, zero_rv=0)

    orbits_cg18 = backintegrate(icrs_arr_cg18, dt=dt, n_steps=n_steps)
    orbits_kc19 = backintegrate(icrs_arr_kc19, dt=dt, n_steps=n_steps)
    orbits_nbhd = backintegrate(icrs_arr_nbhd, dt=dt, n_steps=n_steps)

    # cg18 (core) positions to ngc2516 median position
    d_cg18_full = calc_dist(
        orbits.x[:,0][:,None], orbits.y[:,0][:,None], orbits.z[:,0][:,None],
        orbits_cg18.x, orbits_cg18.y, orbits_cg18.z
    )
    d_cg18_med = np.nanmedian(d_cg18_full, axis=1)
    d_cg18_upper = np.nanpercentile(d_cg18_full, 68, axis=1)
    d_cg18_lower = np.nanpercentile(d_cg18_full, 32, axis=1)

    # ditto halo
    d_kc19_full = calc_dist(
        orbits.x[:,0][:,None], orbits.y[:,0][:,None], orbits.z[:,0][:,None],
        orbits_kc19.x, orbits_kc19.y, orbits_kc19.z
    )
    d_kc19_med = np.nanmedian(d_kc19_full, axis=1)
    d_kc19_upper = np.nanpercentile(d_kc19_full, 68, axis=1)
    d_kc19_lower = np.nanpercentile(d_kc19_full, 32, axis=1)

    # ditto nbhd
    d_nbhd_full = calc_dist(
        orbits.x[:,0][:,None], orbits.y[:,0][:,None], orbits.z[:,0][:,None],
        orbits_nbhd.x, orbits_nbhd.y, orbits_nbhd.z
    )
    d_nbhd_med = np.nanmedian(d_nbhd_full, axis=1)

    # first check
    outpath = f'../results/calc_backintegration_ngc2516/check_backintegrate{s}.png'
    set_style()
    fig = orbits.plot()
    savefig(fig, outpath)

    # second check: the distance between ngc2516 median positions and TOI 1937
    # positions
    plt.close('all')
    f, ax = plt.subplots(figsize=(4,3))
    set_style()
    ax.plot(-1*orbits.t.to(u.Myr).value, d.to(u.pc))
    ax.set_xlabel('Look-back time [Myr]')
    ax.set_ylabel('Separation [pc]')
    ax.set_title('TOI 1937 to NGC 2516 separation')
    outpath = f'../results/calc_backintegration_ngc2516/toi1937_to_ngc2516_mean_distance{s}.png'
    savefig(f, outpath)
    plt.close('all')

    # finally look at the Core and Halo median separations, and compare against
    # TOI 1937
    plt.close('all')
    fig, axs = plt.subplots(figsize=(4,7), sharex=True, nrows=3)
    set_style()

    # the core
    ax = axs[0]
    ax.plot(-1*orbits.t.to(u.Myr).value, d.to(u.pc), label='TOI 1937',
            zorder=4, c='yellow', lw=1,
            path_effects=[pe.Stroke(linewidth=2, foreground='k'), pe.Normal()])
    ax.plot(-1*orbits_cg18.t.to(u.Myr).value, d_cg18_med.to(u.pc),
            label=f'Core median (N={len(s_cg18_df)})', c='k', zorder=3, lw=1)
    ax.fill_between(
        -1*orbits_cg18.t.to(u.Myr).value,
        (d_cg18_lower).to(u.pc),
        (d_cg18_upper).to(u.pc),
        label='Core $\pm 1\sigma$', color='gray', alpha=0.7, zorder=2
    )
    leg = ax.legend(loc='upper left', handletextpad=0.2, fontsize='x-small',
                    framealpha=1)

    # the halo
    ax = axs[1]
    ax.plot(-1*orbits.t.to(u.Myr).value, d.to(u.pc), label='TOI 1937',
            zorder=4, c='yellow', lw=1,
            path_effects=[pe.Stroke(linewidth=2, foreground='k'), pe.Normal()])
    ax.plot(-1*orbits_kc19.t.to(u.Myr).value, d_kc19_med.to(u.pc),
            label=f'Halo median (N={len(s_kc19_df)})', c='C0', zorder=3, lw=1)
    ax.fill_between(
        -1*orbits_kc19.t.to(u.Myr).value,
        (d_kc19_lower).to(u.pc),
        (d_kc19_upper).to(u.pc),
        label='Halo $\pm 1\sigma$', color='lightskyblue', alpha=0.4, zorder=2
    )
    leg = ax.legend(loc='upper left', handletextpad=0.2, fontsize='x-small',
                    framealpha=1)

    # the nbhd
    ax = axs[2]
    ax.plot(-1*orbits.t.to(u.Myr).value, d.to(u.pc), label='TOI 1937',
            zorder=4, c='yellow', lw=1,
            path_effects=[pe.Stroke(linewidth=2, foreground='k'), pe.Normal()])
    ax.plot(-1*orbits_nbhd.t.to(u.Myr).value, d_nbhd_med.to(u.pc),
            label=f'Field median (N={len(s_nbhd_df)})', c='k', zorder=3, lw=1)

    ax.fill_between(
        -1*orbits_nbhd.t.to(u.Myr).value,
        (np.nanpercentile(d_nbhd_full, 32, axis=1)).to(u.pc),
        (np.nanpercentile(d_nbhd_full, 68, axis=1)).to(u.pc),
        label='Field $\pm 1\sigma$', color='gray', alpha=0.75, zorder=2
    )
    ax.fill_between(
        -1*orbits_nbhd.t.to(u.Myr).value,
        (np.nanpercentile(d_nbhd_full, 5, axis=1)).to(u.pc),
        (np.nanpercentile(d_nbhd_full, 95, axis=1)).to(u.pc),
        label='Field $\pm 2\sigma$', color='gray', alpha=0.5, zorder=2
    )
    ax.fill_between(
        -1*orbits_nbhd.t.to(u.Myr).value,
        (np.nanpercentile(d_nbhd_full, 0.3, axis=1)).to(u.pc),
        (np.nanpercentile(d_nbhd_full, 99.7, axis=1)).to(u.pc),
        label='Field $\pm 3\sigma$', color='gray', alpha=0.25, zorder=2
    )

    leg = ax.legend(loc='best', handletextpad=0.2, fontsize='x-small',
                    framealpha=1)

    # cleanup
    fig.text(-0.01,0.5, 'Distance from NGC 2516 trajectory [pc]', va='center',
             rotation=90, fontsize='large')

    ax.set_xlabel('Look-back time [Myr]', fontsize='large')

    for a in axs:
        a.set_xlim([0,100])
        a.set_ylim([0,700])
        # if fix_rvs:
        #     a.set_ylim([0,350])

    fig.tight_layout()

    outpath = f'../results/calc_backintegration_ngc2516/core_halo_to_ngc2516_separation{s}.png'
    savefig(fig, outpath)


def plot_ngc2516_corehalo_3panel(outdir=RESULTSDIR, emph_1937=0, basedata=None,
                                 corealpha=0.9):
    """
    # automatic selection criteria for viable rotation periods
    sel = (
        (df.period < 15)
        &
        (df.lspval > 0.08)
        &
        (df.nequal <= 1)
    )
    """

    nbhd_df, cg18_df, kc19_df, trgt_df = get_gaia_basedata(basedata)

    set_style()

    plt.close('all')

    f, axs = plt.subplots(figsize=(0.9*7,0.9*3), ncols=3)

    xv, yv = 'ra', 'dec'
    axs[0].scatter(
        nbhd_df[xv], nbhd_df[yv], c='gray', alpha=0.5, zorder=2, s=9,
        rasterized=True, linewidths=0, label='Field', marker='.'
    )
    axs[0].scatter(
        kc19_df[xv], kc19_df[yv], c='lightskyblue', alpha=0.9, zorder=3, s=9,
        rasterized=True, linewidths=0.15, label='Halo', marker='.',
        edgecolors='k'
    )
    axs[0].scatter(
        cg18_df[xv], cg18_df[yv], c='k', alpha=corealpha, zorder=4, s=6,
        rasterized=True, label='Core', marker='.'
    )
    if emph_1937:
        axs[0].plot(
            trgt_df[xv], trgt_df[yv], alpha=1, mew=0.5,
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
        get_xval(nbhd_df), get_yval(nbhd_df), c='gray', alpha=0.5, zorder=2,
        s=9, rasterized=True, linewidths=0, label='Field', marker='.'
    )
    axs[1].scatter(
        get_xval(kc19_df), get_yval(kc19_df), c='lightskyblue', alpha=1,
        zorder=3, s=9, rasterized=True, linewidths=0.15, label='Halo',
        marker='.', edgecolors='k'
    )
    axs[1].scatter(
        get_xval(cg18_df), get_yval(cg18_df), c='k', alpha=corealpha,
        zorder=4, s=6, rasterized=True, linewidths=0, label='Core', marker='.'
    )
    if emph_1937:
        axs[1].plot(
            get_xval(trgt_df), get_yval(trgt_df), alpha=1, mew=0.5,
            zorder=8, label='TOI 1937', markerfacecolor='yellow',
            markersize=14, marker='*', color='black', lw=0
        )

    axs[1].set_ylim(axs[1].get_ylim()[::-1])

    axs[1].set_xlabel('Bp - Rp [mag]')
    axs[1].set_ylabel('Absolute G [mag]')

    ##########

    from earhart.paths import DATADIR
    rotdir = os.path.join(DATADIR, 'rotation')
    runid = 'NGC_2516'

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

    xval = (
        df['phot_bp_mean_mag'] - df['phot_rp_mean_mag']
    )
    ykey = 'period'
    if emph_1937:
        from earhart.priors import AVG_EBpmRp, P_ROT
        BpmRp_tic268 = 13.4400 - 12.4347
        axs[2].plot(
            BpmRp_tic268, P_ROT,
            alpha=1, mew=0.5,
            zorder=8, label='TOI 1937', markerfacecolor='yellow',
            markersize=14, marker='*', color='black', lw=0
        )

    prefactor = 2
    sel = (df.subcluster == 'core')
    axs[2].scatter(
        xval[sel], df[sel][ykey], c='k', alpha=corealpha, zorder=4,
        s=prefactor*6, rasterized=True, linewidths=0, label='Core', marker='.'
    )

    sel = (df.subcluster == 'halo')
    axs[2].scatter(
        xval[sel], df[sel][ykey], c='lightskyblue', alpha=1, zorder=3,
        s=prefactor*9, rasterized=True, linewidths=0.15, label='Halo',
        marker='.', edgecolors='k'
    )

    axs[2].yaxis.set_major_locator(MaxNLocator(integer=True))

    from earhart.priors import TEFF, P_ROT, AVG_EBpmRp

    axs[2].set_ylabel('Rotation Period [days]')
    axs[2].set_xlabel('Bp-Rp [mag]')
    axs[2].set_xlim((0.21, 2.04))
    axs[2].set_ylim((0,14.2)) # linear

    ##########

    for ax in axs:
        format_ax(ax)

    f.tight_layout(w_pad=0.5)

    words = ['Field', 'Halo', 'Core'][::-1]
    colors = ['gray', 'lightskyblue', 'k'][::-1]
    # prefactor=4.6,  required for png
    rainbow_text(0.82, 0.08, words, colors, size='medium', ax=axs[0])

    outstr = f'_{basedata}'
    if emph_1937:
        outstr += '_emph1937'
    outpath = os.path.join(outdir, f'ngc2516_corehalo_3panel{outstr}.png')
    savefig(f, outpath)


def plot_full_kinematics_X_rotation(outdir, basedata='bright', show1937=0,
                                    galacticframe=0):
    """
    Match the kinematic members against the AUTOrotation sample.
    """

    nbhd_df, cg18_df, kc19_df, trgt_df = get_gaia_basedata(basedata)

    rot_df, lc_df = get_autorotation_dataframe(runid='NGC_2516', returnbase=True)

    dfs = [nbhd_df, cg18_df, kc19_df, trgt_df, rot_df]
    nbhd_df.source_id = nbhd_df.source_id.astype(np.int64)
    for _df in dfs:
        assert type(_df.source_id.iloc[0]) == np.int64

    if galacticframe:
        c_nbhd = SkyCoord(ra=nparr(nbhd_df.ra)*u.deg, dec=nparr(nbhd_df.dec)*u.deg)
        nbhd_df['l'] = c_nbhd.galactic.l.value
        nbhd_df['b'] = c_nbhd.galactic.b.value

    plt.close('all')

    rvkey = (
        'radial_velocity' if 'edr3' not in basedata else 'dr2_radial_velocity'
    )

    if galacticframe:
        xkey, ykey = 'l', 'b'
        xl, yl = r'$l$ [deg]', r'$b$ [deg]'
    else:
        xkey, ykey = 'ra', 'dec'
        xl, yl = r'$\alpha$ [deg]', r'$\delta$ [deg]'

    params = [xkey, ykey, 'parallax', 'pmra', 'pmdec', rvkey]
    # whether to limit axis by 5/95th percetile
    qlimd = {
        xkey: 0, ykey: 0, 'parallax': 0, 'pmra': 1, 'pmdec': 1, rvkey: 1
    }
    # whether to limit axis by 99th percentile
    nnlimd = {
        xkey: 1, ykey: 1, 'parallax': 1, 'pmra': 0, 'pmdec': 0, rvkey: 0
    }
    ldict = {
        xkey: xl, ykey: yl,
        'parallax': r'$\pi$ [mas]', 'pmra': r"$\mu_{{\alpha'}}$ [mas/yr]",
        'pmdec':  r'$\mu_{{\delta}}$ [mas/yr]', rvkey: 'RV [km/s]'
    }


    nparams = len(params)
    f, axs = plt.subplots(figsize=(6,6), nrows=nparams-1, ncols=nparams-1)

    from earhart.priors import AVG_EBpmRp
    get_BpmRp0 = lambda df: (df['phot_bp_mean_mag'] - df['phot_rp_mean_mag'] - AVG_EBpmRp)
    # this plot is for (Bp-Rp)_0 from 0.5-1.2, and will highlight which
    # kinematic members (for which we made light curves) are and are not rotators.
    sel_color = lambda df: (get_BpmRp0(df) > 0.5) & (get_BpmRp0(df) < 1.2)
    sel_autorot = lambda df: df.source_id.isin(rot_df.source_id)
    sel_haslc = lambda df: df.source_id.isin(lc_df.source_id)

    sel_comp = lambda df: (sel_color(df)) & (sel_haslc(df))
    sel_rotn =  lambda df: (sel_color(df)) & (sel_autorot(df))

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
                nbhd_df[sel_color(nbhd_df)][xv], nbhd_df[sel_color(nbhd_df)][yv],
                c='gray', alpha=0.9, zorder=2, s=5, rasterized=True,
                linewidths=0, label='Field', marker='.'
            )

            axs[i,j].scatter(
                kc19_df[sel_comp(kc19_df)][xv], kc19_df[sel_comp(kc19_df)][yv],
                c='orange', alpha=1, zorder=3, s=12, rasterized=True,
                label='Halo', linewidths=0.1, marker='.', edgecolors='k'
            )
            axs[i,j].scatter(
                kc19_df[sel_rotn(kc19_df)][xv], kc19_df[sel_rotn(kc19_df)][yv],
                c='lightskyblue', alpha=1, zorder=6, s=12, rasterized=True,
                label='Halo + P$_\mathrm{rot}$', linewidths=0.1, marker='.',
                edgecolors='k'
            )

            axs[i,j].scatter(
                cg18_df[sel_comp(cg18_df)][xv], cg18_df[sel_comp(cg18_df)][yv],
                c='k', alpha=0.9, zorder=7, s=2,
                rasterized=True, label='Core', marker='.', edgecolors='none'
            )

            if show1937:
                axs[i,j].plot(
                    trgt_df[xv], trgt_df[yv], alpha=1, mew=0.5,
                    zorder=8, label='TOI 1937', markerfacecolor='yellow',
                    markersize=7, marker='*', color='black', lw=0
                )

            # set the axis limits as needed
            if qlimd[xv]:
                xlim = (np.nanpercentile(nbhd_df[xv], 5),
                        np.nanpercentile(nbhd_df[xv], 95))
                axs[i,j].set_xlim(xlim)
            if qlimd[yv]:
                ylim = (np.nanpercentile(nbhd_df[yv], 5),
                        np.nanpercentile(nbhd_df[yv], 95))
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
    if show1937:
        leg.legendHandles[4]._sizes = [20]

    for ax in axs.flatten():
        format_ax(ax)

    f.tight_layout(h_pad=0.01, w_pad=0.01)

    s = ''
    s += f'_{basedata}'
    if show1937:
        s += f'_show1937'
    if galacticframe:
        s += f'_galactic'
    else:
        s += f'_icrs'
    outpath = os.path.join(outdir, f'full_kinematics_X_rotation{s}.png')
    savefig(f, outpath)


def plot_physical_X_rotation(outdir, basedata='bright', show1937=0,
                             do_histogram=1):
    """
    Same data as "full_kinematics_X_rotation", but in XYZ coordinates, and with
    physical (on-sky) velocity differences from the cluster mean.
    """

    nbhd_df, cg18_df, kc19_df, trgt_df = get_gaia_basedata(basedata)
    rot_df, lc_df = get_autorotation_dataframe(runid='NGC_2516', returnbase=True)

    # select the high probability CG18 members to get median parameters
    cg18_path = os.path.join(DATADIR, 'gaia', 'CantatGaudin2018_vizier_only_NGC2516.fits')
    t_df = Table(fits.open(cg18_path)[1].data).to_pandas()
    if basedata == 'fullfaint':
        assert len(t_df) == len(cg18_df)

    CUTOFF_PROB = 0.7
    sel = (t_df.PMemb > CUTOFF_PROB)

    s_t_df = t_df[sel]

    assert np.all(s_t_df.Source.isin(cg18_df.source_id))

    sel = cg18_df.source_id.isin(s_t_df.Source)

    s_cg18_df = cg18_df[sel]

    rvkey = (
        'radial_velocity' if 'edr3' not in basedata else 'dr2_radial_velocity'
    )
    getcols = ['ra', 'dec', 'parallax', 'pmra', 'pmdec', rvkey]

    med_df = pd.DataFrame(s_cg18_df[getcols].median()).T
    std_df = pd.DataFrame(s_cg18_df[getcols].std()).T

    from earhart.physicalpositions import append_physicalpositions
    cg18_df = append_physicalpositions(cg18_df, med_df)
    kc19_df = append_physicalpositions(kc19_df, med_df)
    nbhd_df = append_physicalpositions(nbhd_df, med_df)
    trgt_df = append_physicalpositions(trgt_df, med_df)

    # this plot is for (Bp-Rp)_0 from 0.5-1.2, and will highlight which
    # kinematic members (for which we made light curves) are and are not rotators.
    from earhart.priors import AVG_EBpmRp
    get_BpmRp0 = lambda df: (df['phot_bp_mean_mag'] - df['phot_rp_mean_mag'] - AVG_EBpmRp)
    sel_color = lambda df: (get_BpmRp0(df) > 0.5) & (get_BpmRp0(df) < 1.2)
    sel_autorot = lambda df: df.source_id.isin(rot_df.source_id)
    sel_haslc = lambda df: df.source_id.isin(lc_df.source_id)

    sel_comp = lambda df: (sel_color(df)) & (sel_haslc(df))
    sel_rotn =  lambda df: (sel_color(df)) & (sel_autorot(df))

    # make it!
    plt.close('all')
    f, axs = plt.subplots(figsize=(4,3), nrows=2, ncols=2)
    axs = axs.flatten()

    xytuples = [
        ('delta_x_pc', 'delta_y_pc'),
        ('delta_x_pc', 'delta_z_pc'),
        ('delta_y_pc', 'delta_z_pc'),
        ('delta_pmra_km_s', 'delta_pmdec_km_s')
    ]
    ldict = {
        'delta_x_pc':'$\Delta$X [pc]',
        'delta_y_pc':'$\Delta$Y [pc]',
        'delta_z_pc':'$\Delta$Z [pc]',
        'delta_pmra_km_s': r"$\Delta \mu_{{\alpha'}}$ [km$\,$s$^{-1}$]",
        'delta_pmdec_km_s': r"$\Delta \mu_{\delta}$ [km$\,$s$^{-1}$]"
    }

    # limit axis by 99th percentile or iqr
    qlimd = {
        'delta_x_pc': 0, 'delta_y_pc': 0, 'delta_z_pc': 0, 'delta_pmra_km_s': 0, 'delta_pmdec_km_s': 0
    }
    nnlimd = {
        'delta_x_pc': 1, 'delta_y_pc': 1, 'delta_z_pc': 1, 'delta_pmra_km_s': 1, 'delta_pmdec_km_s': 1
    }

    for i, xyt in enumerate(xytuples):

        xv, yv = xyt[0], xyt[1]

        # axs[i].scatter(
        #     nbhd_df[sel_color(nbhd_df)][xv], nbhd_df[sel_color(nbhd_df)][yv],
        #     c='gray', alpha=0.9, zorder=2, s=5, rasterized=True,
        #     linewidths=0, label='Field', marker='.'
        # )

        axs[i].scatter(
            kc19_df[sel_comp(kc19_df)][xv], kc19_df[sel_comp(kc19_df)][yv],
            c='orange', alpha=1, zorder=3, s=12, rasterized=True,
            label='Halo', linewidths=0.1, marker='.', edgecolors='k'
        )
        axs[i].scatter(
            kc19_df[sel_rotn(kc19_df)][xv], kc19_df[sel_rotn(kc19_df)][yv],
            c='lightskyblue', alpha=1, zorder=6, s=12, rasterized=True,
            label='Halo + P$_\mathrm{rot}$', linewidths=0.1, marker='.',
            edgecolors='k'
        )

        axs[i].scatter(
            cg18_df[sel_comp(cg18_df)][xv], cg18_df[sel_comp(cg18_df)][yv],
            c='k', alpha=1, zorder=7, s=2, edgecolors='none',
            rasterized=True, label='Core', marker='.'
        )

        if show1937:
            axs[i].plot(
                trgt_df[xv], trgt_df[yv], alpha=1, mew=0.5,
                zorder=8, label='TOI 1937', markerfacecolor='yellow',
                markersize=7, marker='*', color='black', lw=0
            )

        axs[i].set_xlabel(ldict[xv], fontsize='small')
        axs[i].set_ylabel(ldict[yv], fontsize='small')

    # axs[2,2].legend(loc='best', handletextpad=0.1, fontsize='medium', framealpha=0.7)
    # leg = axs[2,2].legend(bbox_to_anchor=(0.8,0.8), loc="upper right",
    #                       handletextpad=0.1, fontsize='medium',
    #                       bbox_transform=f.transFigure)

    # # NOTE: hack size of legend markers
    # leg.legendHandles[0]._sizes = [20]
    # leg.legendHandles[1]._sizes = [25]
    # leg.legendHandles[2]._sizes = [25]
    # leg.legendHandles[3]._sizes = [20]
    # if show1937:
    #     leg.legendHandles[4]._sizes = [20]

    for ax in axs.flatten():
        format_ax(ax)

    f.tight_layout(h_pad=0.1, w_pad=0.1)

    s = ''
    s += f'_{basedata}'
    if show1937:
        s += f'_show1937'
    outpath = os.path.join(outdir, f'physical_X_rotation{s}.png')
    savefig(f, outpath)


    # make the histogram too, of histogram_physical_X_rotation_fullfaint
    if do_histogram:

        if show1937:
            return

        # this is a plot of the 
        mdf = pd.concat((cg18_df, kc19_df))
        comp_df = mdf[sel_comp(mdf)]
        rot_df = mdf[sel_rotn(mdf)]

        plt.close('all')
        f, axs = plt.subplots(figsize=(4,3), nrows=1, ncols=2, sharey=True)
        axs = axs.flatten()

        #
        # first: delta_r_pc
        #
        delta_pc = 25
        bins = np.arange(0, 500+delta_pc, delta_pc)
        xvals = bins[:-1] + delta_pc/2

        h_comp, bins_comp = np.histogram(nparr(comp_df.delta_r_pc), bins=bins)
        h_rot, bins_rot = np.histogram(nparr(rot_df.delta_r_pc), bins=bins)

        axs[0].plot(
            xvals, h_rot/h_comp, marker='o', linewidth=1, linestyle='dashed',
            color='k', ms=5
        )
        axs[0].set_xlabel('$\Delta r$ [pc]')
        axs[0].set_ylabel('Fraction in bin with P$_\mathrm{rot}$')
        axs[0].set_xlim([-delta_pc, 400+delta_pc])

        print(h_rot)
        print(h_comp)

        #
        # then: delta_mu_km_s
        #
        delta_kms = 2.5
        bins = np.arange(0, 50+delta_kms, delta_kms)
        xvals = bins[:-1] + delta_kms/2

        h_comp, bins_comp = np.histogram(nparr(comp_df.delta_mu_km_s), bins=bins)
        h_rot, bins_rot = np.histogram(nparr(rot_df.delta_mu_km_s), bins=bins)

        axs[1].plot(
            xvals, h_rot/h_comp, marker='o', linewidth=1, linestyle='dashed',
            color='k', ms=5
        )
        axs[1].set_xlabel('$\Delta v$ [km$\,$s$^{-1}$]')
        axs[1].set_xlim([-delta_kms, 50+delta_kms])

        print(h_rot)
        print(h_comp)

        for ax in axs.flatten():
            format_ax(ax)

        f.tight_layout(h_pad=0.2, w_pad=0.2)

        s = ''
        s += f'_{basedata}'
        outpath = os.path.join(outdir, f'histogram_physical_X_rotation{s}.png')
        savefig(f, outpath)


