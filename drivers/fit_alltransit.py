"""
Fit the TESS + groundbased TOI 1937 data for {"period", "t0", "log_r", "b",
"u0", "u1"}, plus instrument means.  Believe the error bars, and fix the depth.
"""

import numpy as np, matplotlib.pyplot as plt, pandas as pd, pymc3 as pm
import pickle, os, corner, pytest, socket
from collections import OrderedDict
from pymc3.backends.tracetab import trace_to_dataframe

if 'PU' in socket.gethostname():
    # NOTE: require theano.gcc.cxxflags set to "-Wno-c++11-narrowing"
    assert os.path.exists(
        os.path.join(os.path.expanduser('~'), '.theanorc')
    )

import exoplanet as xo

from os.path import join
from importlib.machinery import SourceFileLoader

try:
    import betty.plotting as bp
except ModuleNotFoundError as e:
    print(f'WRN! {e}')
    pass

from earhart.helpers import get_toi1937_lightcurve, get_groundphot
from earhart.paths import RESULTSDIR, DATADIR, LOCALDIR
from earhart.plotting import _plot_detrending_check

from betty.helpers import (
    _subset_cut, _get_flux_err_as_stdev
)
from betty.posterior_table import make_posterior_table
from betty.modelfitter import ModelFitter

from astrobase.services.identifiers import (
    simbad_to_tic
)
from astrobase.services.tesslightcurves import (
    get_two_minute_spoc_lightcurves
)

from cdips.lcproc import (
    detrend as dtr,
    mask_orbit_edges as moe
)

EPHEMDICT = {
    'TOI_1937': {'t0': 1492.3525, 'per': 0.94667, 'tdur':1.1/24},
}

def fit_alltransit(starid='TOI_1937', N_samples=1000):

    modelid = 'alltransit'
    PLOTDIR = join(RESULTSDIR, f'{starid}_{modelid}_results')
    if not os.path.exists(PLOTDIR):
        os.mkdir(PLOTDIR)

    datasets = OrderedDict()
    if starid == 'TOI_1937':
        time, flux, flux_err, tess_texp = get_toi1937_lightcurve()
    else:
        raise NotImplementedError

    # hard detrend the stellar variability for now, since this is a
    # "alltransit".
    outpath = join(PLOTDIR, 'detrending_check.png')
    flat_flux, trend_flux = dtr.detrend_flux(time, flux)
    _plot_detrending_check(time, flux, trend_flux, flat_flux, outpath)

    # set "flux" to be the detrended value.
    flux = flat_flux
    flux = flux.astype(np.float64)

    time, flux, flux_err = _subset_cut(
        time, flux, flux_err, n=3.0, t0=EPHEMDICT[starid]['t0'],
        per=EPHEMDICT[starid]['per'], tdur=EPHEMDICT[starid]['tdur']
    )

    # NOTE the flux errors from CDIPS are apparently wrong, so set them to be
    # the stdev of the out of transit
    flux_err = _get_flux_err_as_stdev(
        time, flux, t0=EPHEMDICT[starid]['t0'], per=EPHEMDICT[starid]['per'],
        tdur=EPHEMDICT[starid]['tdur']
    )

    datasets['tess'] = [time, flux, flux_err, tess_texp]

    # TODO: include the Evans r-band, 20200214
    # 20200122_ip_TIC268301217.epdlc
    # 20200129_ip_TIC268301217.epdlc
    # 20200131_ip_TIC268301217.epdlc
    # 20200213_gp_TIC268301217.epdlc
    # 20200422_zs_TIC268301217.epdlc
    datestrs = ['20200122', '20200129', '20200131', '20200213', '20200422']
    groundkeytobandpass = {'ground_0': r'i$_\mathrm{LCOGT}$',
                           'ground_1': r'i$_\mathrm{LCOGT}$',
                           'ground_2': r'i$_\mathrm{LCOGT}$',
                           'ground_3': r'g$_\mathrm{LCOGT}$',
                           'ground_4': r'z$_\mathrm{LCOGT}$'}
    for ix, d in enumerate(datestrs):
        x_obs, y_obs, y_err, t_exp = get_groundphot(datestr=d)
        x_obs -= 2457000 # convert to BTJD
        datasets[f'ground_{ix}'] = [x_obs, y_obs, y_err, t_exp]

    priorpath = join(DATADIR, f'{starid}_{modelid}_priors.py')
    if not os.path.exists(priorpath):
        raise FileNotFoundError(f'need to create {priorpath}')
    priormod = SourceFileLoader('prior', priorpath).load_module()
    priordict = priormod.priordict

    pklpath = join(LOCALDIR, f'fit_{starid}_{modelid}.pkl')

    m = ModelFitter(modelid, datasets, priordict, plotdir=PLOTDIR,
                    pklpath=pklpath, overwrite=0, N_samples=N_samples,
                    N_cores=os.cpu_count(), target_accept=0.8)

    print(pm.summary(m.trace, var_names=list(priordict)))

    summdf = pm.summary(m.trace, var_names=list(priordict), round_to=10,
                        kind='stats', stat_funcs={'median':np.nanmedian},
                        extend=True)

    # require: precise
    params = ['t0', 'period']
    for _p in params:
        assert summdf.T[_p].loc['sd'] * 10 < priordict[_p][2]

    # require: priors were accurate
    for _p in params:
        absdiff = np.abs(summdf.T[_p].loc['mean'] - priordict[_p][1])
        priorwidth = priordict[_p][2]
        assert absdiff < priorwidth

    fitindiv = 1
    phaseplot = 1
    cornerplot = 1
    posttable = 1
    grounddepth = 1

    if grounddepth:
        outpath = join(PLOTDIR, f'{starid}_{modelid}_grounddepth.png')
        bp.plot_grounddepth(m, summdf, outpath, modelid=modelid,
                            showerror=0,
                            groundkeytobandpass=groundkeytobandpass,
                            tdur=EPHEMDICT[starid]['tdur'])
        outpath = join(PLOTDIR, f'{starid}_{modelid}_grounddepth_talkaspect.png')
        bp.plot_grounddepth(m, summdf, outpath, modelid=modelid,
                            showerror=0, talkaspect=1,
                            groundkeytobandpass=groundkeytobandpass,
                            tdur=EPHEMDICT[starid]['tdur'])

    if posttable:
        outpath = join(PLOTDIR, f'{starid}_{modelid}_posteriortable.tex')
        make_posterior_table(pklpath, priordict, outpath, modelid, makepdf=1)

    if phaseplot:
        outpath = join(PLOTDIR, f'{starid}_{modelid}_phaseplot.png')
        bp.plot_phasefold(m, summdf, outpath, modelid=modelid, inppt=1,
                          xlim=(-2.1,2.1), binsize_minutes=5, savepdf=1)

    if cornerplot:
        outpath = join(PLOTDIR, f'{starid}_{modelid}_cornerplot.png')
        bp.plot_cornerplot(list(priordict), m, outpath)

    if fitindiv:
        raise NotImplementedError("not important for the alltransit model case")
        outpath = join(PLOTDIR, f'{starid}_{modelid}_fitindiv.png')
        bp.plot_fitindiv(m, summdf, outpath, modelid=modelid)


if __name__ == "__main__":
    fit_alltransit(starid='TOI_1937', N_samples=4000)
