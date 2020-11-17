"""
The stellar spin periods and planetary orbital periods originally collected by
Penev et al (2018) are shown in Figure~\ref{fig:Pspin_vs_Porb}. We only show
hot Jupiter systems with spin period S/N ratios of at least 5, and have colored
the hot Jupiters by whether their stellar radius is below or above
$1.2R_\odot$. For a dwarf star, this radius corresponds to effective
temperatures above and below roughly 6000$\,$K, roughly at the F9V-G0V
boundary, and slightly below the Kraft break where the stellar spindown becomes
particularly inefficient.  This control between early and late spectral types
was apparently not needed, as four of the five other hot Jupiter host stars
with the shortest rotation periods have radii below $1.2R_\odot$ (CoRoT-18,
CoRoT-2, HAT-P-20, and HAT-P-23).  Similarly, the other hot Jupiters with
orbital periods below 1 day (HATS-18, WASP-19, and WASP-43) all also orbit G or
K dwarf hosts.
"""
import numpy as np, matplotlib.pyplot as plt
from astropy.io import fits
from aesthetic.plot import set_style, savefig

from scipy.stats import linregress

hdulist = fits.open('../data/Penev_2018_table1_vizier.fits')
d = hdulist[1].data
hdulist.close()

# 'ID', 'Per', 'E_Per', 'e_per_lc', 'SPer', 'E_SPer', 'e_sper_lc', 'M_', 'E_M_',
# 'e_m__lc', 'R_', 'E_R_', 'e_r__lc', 'Mp', 'E_Mp', 'e_mp_lc', 'Rp', 'E_Rp',
# 'e_rp_lc', 'l_logQ', 'logQ', 'E_logQ', 'e_logq_lc',

def plot_Porb_Prot(snr_cutoff=2, includefit=False):

    sel = (
        ((d['SPer'] / d['E_SPer']) > snr_cutoff)
        &
        (d['SPer'] < 30)
    )
    sel0 = (
        ((d['SPer'] / d['E_SPer']) > snr_cutoff)
        &
        (d['R_'] < 1.2)
        &
        (d['SPer'] < 30)
    )
    sel1 = (
        ((d['SPer'] / d['E_SPer']) > snr_cutoff)
        &
        (d['R_'] >= 1.2)
        &
        (d['SPer'] < 30)
    )
    sel2 = (
        ((d['SPer'] / d['E_SPer']) > snr_cutoff)
        &
        (d['R_'] < 1.2)
        &
        (d['SPer'] < 8)
    )
    sel3 = (
        ((d['SPer'] / d['E_SPer']) > snr_cutoff)
        &
        (d['R_'] < 1.2)
        &
        (d['Per'] < 1)
    )

    print(d['ID'][sel2])
    print(d['ID'][sel3])

    #
    # make the plot!
    #
    set_style()

    f, ax = plt.subplots(figsize=(4,3))

    Porb_1937 = 0.947
    Prot_1937 = 6.5

    ax.scatter(
        d['Per'][sel0], d['SPer'][sel0], zorder=2, c='k', alpha=0.9, s=9,
        linewidths=0,
        label='HJs ($\mathrm{R}_{\star} < 1.2 \mathrm{R}_{\odot}$)'
    )
    ax.scatter(
        d['Per'][sel1], d['SPer'][sel1], zorder=2, c='gray', alpha=0.9, s=9,
        linewidths=0,
        label='HJs ($\mathrm{R}_{\star} \geq 1.2 \mathrm{R}_{\odot}$)'
    )
    ax.plot(
        Porb_1937, Prot_1937, mew=0.5, zorder=3,
        markerfacecolor='yellow', markersize=18, marker='*',
        color='k', lw=0, label='TOI 1937b (1.1$\mathrm{R}_{\odot}$)'
    )

    if includefit:

        x = np.hstack([d['Per'][sel0], Porb_1937])
        y = np.hstack([d['SPer'][sel0], Prot_1937])

        slope, intercept, rvalue, pvalue, stderr = linregress(x, y)

        label = f'Slope={slope:.1f}$\pm${stderr:.1f}, p={pvalue:.1e}'

        ax.plot(x, intercept + slope*x, label=label)

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), handletextpad=0.1)

    ax.set_xlabel('Planet orbital period [days]')
    ax.set_ylabel('Stellar spin period [days]')

    ax.set_xlim([0.5, 4.2])
    ax.set_ylim([2, 28])

    extrastr = '_withfit' if includefit else '_nofit'
    figpath = (
        f'../results/Porb_Prot/Porb_Prot_snrcut{snr_cutoff}{extrastr}.png'
    )
    savefig(f, figpath)


if __name__ == "__main__":

    for includefit in [0, 1]:
        for snr_cutoff in [1,2,3,4,5]:
            plot_Porb_Prot(snr_cutoff=snr_cutoff, includefit=includefit)
