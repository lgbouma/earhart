"""
Similar to Prot vs Porb, but now vs. a tidal parameter, (Mp/Ms)*(Rs/a)^5.
The data were collected by Roberto Tejada.
"""
import numpy as np, matplotlib.pyplot as plt, pandas as pd
from astropy.io import fits
from aesthetic.plot import set_style, savefig
from astropy import units as u

from scipy.stats import linregress

df = pd.read_csv('../data/Tejada_HJs_phot_periods.csv')

def plot_Prot_vs_eps(includefit=False, xscale=None):

    #
    # make the plot!
    #
    set_style()

    f, ax = plt.subplots(figsize=(4,3))

    Porb_1937 = 0.947
    Prot_1937 = 6.5
    MpMs_1937 = ((1.79*u.Mjup) / (1.045*u.Msun)).cgs.value
    Rs_a_1937 = (1/4.26) # from MIT report
    eps_1937 = MpMs_1937 * Rs_a_1937**5

    ax.scatter(
        df['epsilon'], df['P_rot'], zorder=2, c='k', alpha=0.9, s=9,
        linewidths=0,
        label='HJs (phot P$_\mathrm{rot}$)'
    )
    # ax.scatter(
    #     d['Per'][sel1], d['SPer'][sel1], zorder=2, c='gray', alpha=0.9, s=9,
    #     linewidths=0,
    #     label='HJs ($\mathrm{R}_{\star} \geq 1.2 \mathrm{R}_{\odot}$)'
    # )
    ax.plot(
        eps_1937, Prot_1937, mew=0.5, zorder=3,
        markerfacecolor='yellow', markersize=18, marker='*',
        color='k', lw=0, label='TOI 1937b'
    )

    # if includefit:

    #     x = np.hstack([d['Per'][sel0], Porb_1937])
    #     y = np.hstack([d['SPer'][sel0], Prot_1937])

    #     slope, intercept, rvalue, pvalue, stderr = linregress(x, y)

    #     label = f'Slope={slope:.1f}$\pm${stderr:.1f}, p={pvalue:.1e}'

    #     ax.plot(x, intercept + slope*x, label=label)

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), handletextpad=0.1)

    ax.set_xlabel('$\epsilon \equiv (M_\mathrm{p}/M_{\star})(R_{\star}/a)^5$')
    ax.set_ylabel('Stellar spin period [days]')

    # ax.set_xlim([0.5, 4.2])
    ax.set_ylim([2, 28])

    extrastr = '_withfit' if includefit else '_nofit'

    if isinstance(xscale, str):
        ax.set_xscale(xscale)
        extrastr += f'_{xscale}'

    figpath = (
        f'../results/Prot_vs_eps/Prot_vs_eps_snrcut{extrastr}.png'
    )
    savefig(f, figpath)


if __name__ == "__main__":

    for includefit in [0]:
        for xscale in ['linear','log']:
            plot_Prot_vs_eps(includefit=includefit, xscale=xscale)
