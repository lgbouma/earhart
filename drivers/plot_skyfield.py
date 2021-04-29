"""
Use Eleanor Lutz's absolutely amazing
https://github.com/eleanorlutz/western_constellations_atlas_of_space to
visualize the NGC 2516 halo.
"""
import pandas as pd, numpy as np, matplotlib.pyplot as plt
import os
import cartopy.crs as ccrs
from numpy import array as nparr
import matplotlib.ticker as mticker

from earhart.skyfield_helpers import (
    get_hygdata, get_asterisms, get_constellation_boundaries,
    get_messier_data, radec_to_lb
)

from earhart.paths import DATADIR, RESULTSDIR
PROCESSED_DIR = os.path.join(DATADIR, 'skyfield', 'processed')

def gridlines_with_labels(ax, top=True, bottom=True, left=True,
                          right=True, **kwargs):
    """
    !!!
    Copy pasta from a github gist. It does not work.
    !!!

    Like :meth:`cartopy.mpl.geoaxes.GeoAxes.gridlines`, but will draw
    gridline labels for arbitrary projections.
    Parameters
    ----------
    ax : :class:`cartopy.mpl.geoaxes.GeoAxes`
        The :class:`GeoAxes` object to which to add the gridlines.
    top, bottom, left, right : bool, optional
        Whether or not to add gridline labels at the corresponding side
        of the plot (default: all True).
    kwargs : dict, optional
        Extra keyword arguments to be passed to :meth:`ax.gridlines`.
    Returns
    -------
    :class:`cartopy.mpl.gridliner.Gridliner`
        The :class:`Gridliner` object resulting from ``ax.gridlines()``.
    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>> import cartopy.crs as ccrs
    >>> plt.figure(figsize=(10, 10))
    >>> ax = plt.axes(projection=ccrs.Orthographic(-5, 53))
    >>> ax.set_extent([-10.0, 0.0, 50.0, 56.0], crs=ccrs.PlateCarree())
    >>> ax.coastlines('10m')
    >>> gridlines_with_labels(ax)
    >>> plt.show()
    """

    # Add gridlines
    gridliner = ax.gridlines(**kwargs)

    ax.tick_params(length=0)

    # Get projected extent
    xmin, xmax, ymin, ymax = ax.get_extent()

    # Determine tick positions
    sides = {}
    N = 500
    if bottom:
        sides['bottom'] = np.stack([np.linspace(xmin, xmax, N),
                                    np.ones(N) * ymin])
    if top:
        sides['top'] = np.stack([np.linspace(xmin, xmax, N),
                                np.ones(N) * ymax])
    if left:
        sides['left'] = np.stack([np.ones(N) * xmin,
                                  np.linspace(ymin, ymax, N)])
    if right:
        sides['right'] = np.stack([np.ones(N) * xmax,
                                   np.linspace(ymin, ymax, N)])

    # Get latitude and longitude coordinates of axes boundary at each side
    # in discrete steps
    gridline_coords = {}
    for side, values in sides.items():
        gridline_coords[side] = ccrs.PlateCarree().transform_points(
            ax.projection, values[0], values[1])

    lon_lim, lat_lim = gridliner._axes_domain(
        background_patch=ax.background_patch
    )
    ticklocs = {
        'x': gridliner.xlocator.tick_values(lon_lim[0], lon_lim[1]),
        'y': gridliner.ylocator.tick_values(lat_lim[0], lat_lim[1])
    }

    # Compute the positions on the outer boundary where
    coords = {}
    for name, g in gridline_coords.items():
        if name in ('bottom', 'top'):
            compare, axis = 'x', 0
        else:
            compare, axis = 'y', 1
        coords[name] = np.array([
            sides[name][:, np.argmin(np.abs(
                gridline_coords[name][:, axis] - c))]
            for c in ticklocs[compare]
        ])

    # Create overlay axes for top and right tick labels
    ax_topright = ax.figure.add_axes(ax.get_position(), frameon=False)
    ax_topright.tick_params(
        left=False, labelleft=False,
        right=True, labelright=True,
        bottom=False, labelbottom=False,
        top=True, labeltop=True,
        length=0
    )
    ax_topright.set_xlim(ax.get_xlim())
    ax_topright.set_ylim(ax.get_ylim())

    for side, tick_coords in coords.items():
        if side in ('bottom', 'top'):
            axis, idx = 'x', 0
        else:
            axis, idx = 'y', 1

        _ax = ax if side in ('bottom', 'left') else ax_topright

        ticks = tick_coords[:, idx]

        valid = np.logical_and(
            ticklocs[axis] >= gridline_coords[side][0, idx],
            ticklocs[axis] <= gridline_coords[side][-1, idx])

        if side in ('bottom', 'top'):
            _ax.set_xticks(ticks[valid])
            _ax.set_xticklabels([LONGITUDE_FORMATTER.format_data(t)
                                 for t in ticklocs[axis][valid]])
        else:
            _ax.set_yticks(ticks[valid])
            _ax.set_yticklabels([LATITUDE_FORMATTER.format_data(t)
                                 for t in ticklocs[axis][valid]])

    return gridliner


def plot_orthographic_references():

    _, stars = get_hygdata()
    asterisms = get_asterisms()
    const_names = pd.read_csv(
        os.path.join(PROCESSED_DIR,'centered_constellations.csv'),
        encoding="latin-1"
    )

    #names = ['Carina', 'Cancer', 'Corona Borealis', 'Crater',
    #         'Lepus', 'Gemini', 'Cygnus', 'Sagittarius', 'Orion']

    names = ['Carina', 'Orion', 'Corona Borealis', 'Lepus']
    gap = 50

    fig = plt.figure(figsize=(12, 12))

    for i, name in enumerate(names):
        valdf = const_names[const_names['name'] == name]
        clon = 360/24*valdf['ra'].tolist()[0]
        clat = valdf['dec'].tolist()[0]

        ax = fig.add_subplot(np.ceil(np.sqrt(len(names))), np.ceil(np.sqrt(len(names))), i+1,
                             projection=ccrs.Orthographic(central_longitude=clon, central_latitude=clat))

        ax.set_extent([clon-gap, clon+gap, clat+gap, clat-gap], ccrs.PlateCarree())
        for index, row in asterisms.iterrows():
            ras = [float(x)*360/24 for x in row['ra'].replace('[', '').replace(']', '').split(',')]
            decs = [float(x) for x in row['dec'].replace('[', '').replace(']', '').split(',')]
            for n in range(int(len(asterisms)/2)):
                ax.plot(ras[n*2:(n+1)*2], decs[n*2:(n+1)*2], transform=ccrs.Geodetic(), color='k', lw=0.5)

        ax.set_title(name)

        magnitude = stars['mag']
        limiting_magnitude = 6.5
        marker_size = (0.5 + limiting_magnitude - magnitude) ** 1.7

        ax.scatter(360/24*stars['ra'], stars['dec'],
                   transform=ccrs.PlateCarree(), s=marker_size, alpha=0.5, lw=0)
        stars_names = stars[pd.notnull(stars['proper'])]
        stars_names = stars_names[stars_names['dec'].between(clat-gap, clat+gap)]
        stars_names = stars_names[stars_names['ra'].between((clon-gap)/(360/24), (clon+gap)/(360/24))]
        for index, row in stars_names.iterrows():
            ax.text(360/24*row['ra'], row['dec'], row['proper'], ha='left', va='center',
                    transform=ccrs.Geodetic(), fontsize=5)

        ax.set_xlim(ax.get_xlim()[::-1])

        # gridlines_with_labels(ax, {'color':'gray', 'linestyle':'--'})
        gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=1,
                          color='gray', alpha=0.5, linestyle='--',
                          draw_labels=True)


    outpath = os.path.join(RESULTSDIR, 'skyfield_ngc2516', 'references.png')
    plt.savefig(outpath, format='png', dpi=100)
    plt.close('all')



def plot_carina():

    clon, clat = 277, -15 # rough ngc 2516 location (not used if ra/dec)
    gap = 20 # size of field, deg
    factor = 0.8 # for scaling star sizes
    limiting_magnitude = 6.5

    name = 'Carina'
    _, stars = get_hygdata()
    asterisms = get_asterisms()
    messiers = get_messier_data()
    const_names = pd.read_csv(
        os.path.join(PROCESSED_DIR,'centered_constellations.csv'),
        encoding="latin-1"
    )

    # if using RA/dec
    valdf = const_names[const_names['name'] == name]
    clon = 360/24*valdf['ra'].tolist()[0]
    clat = valdf['dec'].tolist()[0]


    fig = plt.figure(figsize=(4, 4))

    #ax = fig.add_subplot(
    #    projection=ccrs.PlateCarree(
    #    )
    #)
    # ax = fig.add_subplot(
    #     projection=ccrs.Orthographic(
    #         central_longitude=clon, central_latitude=clat
    #     )
    # )
    ax = fig.add_subplot(
        projection=ccrs.AzimuthalEquidistant(
            central_longitude=clon, central_latitude=clat
        )
    )


    ax.set_extent([clon-gap, clon+gap, clat+gap, clat-gap], ccrs.PlateCarree())

    for index, row in asterisms.iterrows():
        ras = [float(x)*360/24 for x in
               row['ra'].replace('[', '').replace(']', '').split(',')]

        decs = [float(x) for x in
                row['dec'].replace('[', '').replace(']', '').split(',')]

        ls,bs = radec_to_lb(ras, decs)

        for n in range(int(len(asterisms)/2)):
            ax.plot(ras[n*2:(n+1)*2], decs[n*2:(n+1)*2],
                    transform=ccrs.Geodetic(), color='k', lw=0.5)
            #ax.plot(ls[n*2:(n+1)*2], bs[n*2:(n+1)*2],
            #        transform=ccrs.Geodetic(), color='k', lw=0.5)

    # overplot messier objects!
    # for index, row in messiers.iterrows():
    #     ax.text(row['ra']*360/24, row['dec'], row['name_2'], color='C0',
    #             ha='left', va='center', transform=ccrs.Geodetic())

    magnitude = stars['mag']
    marker_size = (0.5 + limiting_magnitude - magnitude) ** 1.5

    ras, decs = 360/24*nparr(stars['ra']), nparr(stars['dec'])
    ls, bs = radec_to_lb(ras, decs)
    stars['l'], stars['b'] = ls, bs
    ax.scatter(ras, decs, transform=ccrs.PlateCarree(), s=marker_size,
               alpha=0.5, lw=0, c='k')

    stars_names = stars[pd.notnull(stars['proper'])]
    stars_names = stars_names[
        stars_names['ra'].between(
            clat-factor*gap, clat+factor*gap
        )
    ]
    stars_names = stars_names[
        stars_names['dec'].between(
            (clon-factor*gap), (clon+factor*gap)
        )
    ]

    for index, row in stars_names.iterrows():
        ax.text(row['ra'], row['dec'], row['proper'], ha='left',
                va='center', transform=ccrs.Geodetic(), fontsize=5)

    ax.set_xlim(ax.get_xlim()[::-1])

    # method1
    # gridlines_with_labels(ax, {'color':'gray', 'linestyle':'--'})

    # method 2
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

    gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=0.5,
                      color='gray', alpha=0.5, linestyle='--',
                      x_inline=False, y_inline=False, dms=True,
                      draw_labels=True)

    gl.left_labels=True
    gl.bottom_labels=True
    gl.top_labels=False
    gl.right_labels=False
    # xlocs = [240-360, 260-360, 280-360, 300-360, 320-360]
    # gl.xlocator = mticker.FixedLocator(xlocs)
    # gl.xformatter = LONGITUDE_FORMATTER
    # ylocs = [-20,-15,-10,-5]
    # gl.ylocator = mticker.FixedLocator(ylocs)
    # gl.yformatter = LATITUDE_FORMATTER


    # # method 3
    # from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    # # first do "fake gridlines" for the lines themselves
    # gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=0.5,
    #                   color='gray', alpha=0.5, linestyle='--')
    # gl.xlines = False
    # xlocs = [240-360, 260-360, 280-360, 300-360, 320-360]
    # gl.xlocator = mticker.FixedLocator(xlocs)
    # # then do them for the labels
    # gl2 = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5,
    #                    color='gray', alpha=0.0, linestyle='--')
    # gl2.xlocator = mticker.FixedLocator(xlocs)
    # #gl2.left_labels=True
    # #gl2.bottom_labels=True
    # #gl2.top_labels=False
    # #gl2.right_labels=False
    # gl2.xformatter = LONGITUDE_FORMATTER
    # gl2.yformatter = LATITUDE_FORMATTER
    # # gl2.xlabel_style = {'size': 15, 'color': 'gray'}
    # # gl2.xlabel_style = {'color': 'red', 'weight': 'bold'}

    # # method 4 -- only on rectangular projections
    # xlocs = np.arange(240,340,20)
    # xticklocs = xlocs-360
    # ylocs = np.arange(-30,0,5)
    # ax.set_xticks(xticklocs)
    # ax.set_xticklabels(xlocs)
    # ax.set_yticks(ylocs)
    # ax.set_yticklabels(ylocs)
    # ax.grid(True, color='gray', linestyle='--', linewidth=0.5, zorder=-1)

    # ax.text(-0.07, 0.55, 'latitude', va='bottom', ha='center',
    #                 rotation='vertical', rotation_mode='anchor',
    #                 transform=ax.transAxes)
    # ax.text(0.5, -0.2, 'longitude', va='bottom', ha='center',
    #                 rotation='horizontal', rotation_mode='anchor',
    #                 transform=ax.transAxes)


    outpath = os.path.join(RESULTSDIR, 'skyfield_ngc2516', 'carina.png')
    plt.savefig(outpath, format='png', dpi=200, bbox_inches='tight')
    plt.close('all')


if __name__ == "__main__":

    do_references = 0
    do_carina = 1
    do_skyfield = 0

    # debugging
    if do_references:
        plot_orthographic_references()

    if do_carina:
        plot_carina()

    # for paper / talks
    if do_skyfield:
        from earhart.plotting import plot_skyfield
        outdir = os.path.join(RESULTSDIR, 'skyfield_ngc2516')

        plot_skyfield(outdir, plot_starnames=0, plot_constnames=1, plot_core=1,
                      plot_halo=1)
        plot_skyfield(outdir, plot_starnames=0, plot_constnames=1, plot_core=1,
                      plot_halo=0)
        plot_skyfield(outdir, plot_starnames=0, plot_constnames=1, plot_core=0,
                      plot_halo=0)
