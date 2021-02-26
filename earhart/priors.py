import numpy as np
from collections import OrderedDict

# FROM ZASPE BASED ON PFS SPECTRA (Cf. BRAHM 20201116)
RSTAR = 1.058
RSTAR_STDEV = 0.06 # systematic unc based on comparison to SPECMATCH
MSTAR = 1.045
MSTAR_STDEV = 0.05 # systematic unc based on comparison to SPECMATCH
LOGG = 4.409
LOGG_STDEV = 0.03 # systematic unc based on comparison to ZHOU
TEFF = 5880
TEFF_STDEV = 100 # systematic unc based on comparison to SPECMATCH

LI_EW = 30 # TODO: this is a guess at upper limit
P_ROT = 6.5 # TODO: improve guesstimate. allvariability_report gives 7.08 from s9 only... earlier guess was 6.5d given s7+s9.  still should run it w/ masked
ROT_AMP = 0.01 # TODO: guess of 10 parts per thousand, peak-to-peak, by eye

VSINI = 9.01 # km/s, from ZASPE+PFS
VSINI_STDEV = 0.13 # km/s, ditto.

RV_PEAK_TO_PEAK_JITTER = VSINI*ROT_AMP
K_JITTER = RV_PEAK_TO_PEAK_JITTER * 0.5

# estimated from S98 in
# /Users/luke/Dropbox/proj/cdips_followup/results/TIC_268_neighborhood/LGB_extinction
AVG_EBmV = 0.21

DISTANCE_PC = 1/(2.4e-3) # average distance to NGC2516 members
b_2516 = -15.86 # degrees; SIMBAD

from earhart.extinction import given_S98_EBmV_correct
AVG_EBmV_corr = given_S98_EBmV_correct(AVG_EBmV, distance_pc=DISTANCE_PC, b=b_2516)

AVG_EBpmRp = 1.31 * AVG_EBmV_corr
AVG_AG = 2.72 * AVG_EBmV_corr

# R_V = A(V) / E(B-V)
AVG_AV_corr = 3.1 * AVG_EBmV_corr


def initialize_prior_d(modelcomponents, datasets=None):

    raise NotImplementedError

    # P_orb = 8.32467 # SPOC, +/- 4e-4
    # t0_orb = 1574.2738 # SPOC, +/- 1e-3

    P_orb = 8.32489
    t0_orb = 1574.27380

    # P_orb = 8.3248972 # SG1, +/- 3.4e-4
    # t0_orb = 1574.2738304  # SG1, +/- 1.1e-3,  BTJD

    # P_orb = 8.328 # TESS + El Sauce fit
    # t0_orb = 1574.2646 # plausible transit

    rp_rs = 0.0865 # +/- 0.0303

    # Visual inspection
    P_rot = 3.3 # +/- 1 (by eye)
    t0_rot = None # idk, whatever phase. it's not even a model parameter
    amp = 1e3*1.5e-2 # primary mode amplitude [ppt]. peak to peak is a bit over 2%
    amp_mix = 0.5 # between 0 and 1
    log_Q0 = np.log(1e1) # Q of secondary oscillation. wide prior
    log_deltaQ = np.log(1e1) # primary mode gets higher quality

    prior_d = OrderedDict()

    for modelcomponent in modelcomponents:

        if 'alltransit' in modelcomponent:
            prior_d['period'] = P_orb
            prior_d['t0'] = t0_orb
            if 'quaddepthvar' not in modelcomponents:
                prior_d['log_r'] = np.log(rp_rs)
            else:
                # NOTE: this implementation is very 837-specific.
                prior_d['log_r_Tband'] = np.log(rp_rs)
                prior_d['log_r_Bband'] = np.log(rp_rs)
                prior_d['log_r_Rband'] = np.log(rp_rs)
            prior_d['b'] = 0.5  # initialize for broad prior

            prior_d['r_star'] = RSTAR
            prior_d['logg_star'] = LOGG

            # T-band Teff 5900K, logg 4.50 (Claret+18)
            prior_d['u[0]'] = 0.3249
            prior_d['u[1]'] = 0.235


            for n, (name, (x, y, yerr, texp)) in enumerate(datasets.items()):
                prior_d[f'{name}_mean'] = 1

        if 'quad' in modelcomponent:
            for n, (name, (x, y, yerr, texp)) in enumerate(datasets.items()):
                if name == 'tess':
                    pass
                else:
                    # model [per ground-transit] is :
                    # a0+ a1*(time-midtime) + a2*(time-midtime)^2.
                    # a0 is the mean, already above.
                    prior_d[f'{name}_a1'] = 0
                    prior_d[f'{name}_a2'] = 0

        if modelcomponent not in ['alltransit', 'onetransit',
                                  'allindivtransit', 'tessindivtransit',
                                  'oddindivtransit', 'evenindivtransit']:
            if 'transit' in modelcomponent:
                prior_d['period'] = P_orb
                prior_d['t0'] = t0_orb
                # prior_d['r'] = rp_rs
                prior_d['log_r'] = np.log(rp_rs)
                prior_d['b'] = 0.5  # initialize for broad prior
                prior_d['u[0]'] = 0.3249
                prior_d['u[1]'] = 0.235
                prior_d['mean'] = 1
                prior_d['r_star'] = RSTAR
                prior_d['logg_star'] = LOGG

        if modelcomponent == 'onetransit':
            prior_d['period'] = 8.32483
            prior_d['t0'] = 1574.27273
            # prior_d['r'] = rp_rs
            prior_d['log_r'] = np.log(rp_rs)
            prior_d['b'] = 0.5  # initialize for broad prior
            prior_d['u[0]'] = 0.3249
            prior_d['u[1]'] = 0.235
            prior_d['r_star'] = RSTAR
            prior_d['logg_star'] = LOGG
            for n, (name, (x, y, yerr, texp)) in enumerate(datasets.items()):
                if name == 'tess':
                    raise NotImplementedError
                # model [per ground-transit] is :
                # a0+ a1*(time-midtime) + a2*(time-midtime)^2.
                # a0 is the mean, already above.
                prior_d[f'{name}_mean'] = 1
                prior_d[f'{name}_a1'] = 0
                prior_d[f'{name}_a2'] = 0

        if modelcomponent in ['allindivtransit', 'tessindivtransit',
                              'oddindivtransit', 'evenindivtransit']:
            prior_d['period'] = P_orb
            prior_d['t0'] = t0_orb
            prior_d['log_r'] = np.log(rp_rs)
            prior_d['b'] = 0.95

            prior_d['r_star'] = RSTAR
            prior_d['logg_star'] = LOGG

            # T-band Teff 6000K, logg 4.50 (Claret+18)
            prior_d['u[0]'] = 0.3249
            prior_d['u[1]'] = 0.235
            # NOTE: deprecated; if using xo.distributions.QuadLimbDark
            # prior_d['u'] = [0.3249, 0.235]

            for n, (name, (x, y, yerr, texp)) in enumerate(datasets.items()):
                # mean + a1*(time-midtime) + a2*(time-midtime)^2.
                prior_d[f'{name}_mean'] = 1
                prior_d[f'{name}_a1'] = 0
                prior_d[f'{name}_a2'] = 0

        if 'rv' in modelcomponent:
            raise NotImplementedError

        if 'gp' in modelcomponent:
            raise NotImplementedError

        if 'sincos' in modelcomponent:
            raise NotImplementedError('try billy')

    return prior_d
