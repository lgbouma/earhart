"""
Tools for working in XYZ, UVW, and Δμ_δ, Δμ_α' [km/s] coordinates.

Contents:
    append_physicalpositions: (α,δ,π,μ_α,μ_δ)->(X,Y,Z,v*_tang).
    given_gaia_df_get_icrs_arr: DataFrame -> SkyCoord conversion
    calc_dist: between two 3d cartesian points.
    calc_vl_vb_physical: on-sky galactic velocity conversion
    get_vl_lsr_corr: get vLSR correction given glon
    calculate_XYZ_given_RADECPLX: (α,δ,π)->(X,Y,Z).
"""

import numpy as np
from numpy import array as nparr
import astropy.units as u
import astropy.constants as const

from astropy.coordinates import Galactocentric
import astropy.coordinates as coord
_ = coord.galactocentric_frame_defaults.set('v4.0')

from astropy.coordinates import SkyCoord

VERBOSE = 1
if VERBOSE:
    print(Galactocentric())


def append_physicalpositions(df, reference_df):
    """
    Given a pandas dataframe `df` with [ra, dec, parallax, pmra, pmdec], and a
    reference dataframe `reference_df` with [ra, dec, parallax, pmra, pmdec,
    (dr2_)radial_velocity], calculate the XYZ coordinates, the
    "delta_pmra_prime_km_s" and "delta_pmdec_prime_km_s" coordinates
    (tangential velocity with respect to reference_df), the "delta_r_pc" (3d
    separation) and "delta_mu_km_s" (2d tangential velocity separation)
    coordinates, and append them to the dataframe.

    Returns:
        new DataFrame with ['x_pc', 'y_pc', 'z_pc', 'delta_r_pc',
        'delta_mu_km_s', 'delta_pmra_km_s' ,'delta_pmdec_km_s'] appended.
    """

    cols1 = ["ra","dec","parallax","pmra","pmdec"]
    assert np.sum(df.columns.str.match('|'.join(cols1))) >= 5

    cols2 = ["ra","dec","parallax","pmra","pmdec"]
    assert np.sum(reference_df.columns.str.match('|'.join(cols1))) >= 5

    assert len(df) > 0

    assert (
        'radial_velocity' in reference_df.columns
        or
        'dr2_radial_velocity' in reference_df.columns
    )

    # get XYZ in galactocentric
    get_galcen = lambda _c : _c.transform_to(coord.Galactocentric())
    c_median = get_galcen(given_gaia_df_get_icrs_arr(reference_df, zero_rv=1))
    c_df = get_galcen(given_gaia_df_get_icrs_arr(df, zero_rv=1))

    # ditto, but in icrs. used in the comoving projection trick.
    # as in plotting.plot_vtangential_projection
    icrs_c_median = given_gaia_df_get_icrs_arr(reference_df, zero_rv=0)
    icrs_c_df = given_gaia_df_get_icrs_arr(df, zero_allvelocities=1)

    # get 3D separation from cluster median
    sep_3d_position_pc = (c_median.separation_3d(c_df)).to(u.pc).value

    df['x_pc'] = c_df.x.to(u.pc).value
    df['y_pc'] = c_df.y.to(u.pc).value
    df['z_pc'] = c_df.z.to(u.pc).value
    df['delta_r_pc'] = sep_3d_position_pc

    reference_df['x_pc'] = c_median.x.to(u.pc).value
    reference_df['y_pc'] = c_median.y.to(u.pc).value
    reference_df['z_pc'] = c_median.z.to(u.pc).value

    # get observed differences from cluster
    target_cols = ['ra', 'dec', 'parallax', 'pmra', 'pmdec', 'x_pc', 'y_pc', 'z_pc']
    for c in target_cols:
        new_key = 'delta_'+c
        new_val = df[c] - nparr(reference_df[c]) # careful, without nparr does it by iloc
        df[new_key] = new_val

        if c in ['pmra', 'pmdec']:
            #
            # naive approach: just take the different to the median
            # pmra/pmdec of the cluster. (ignoring the spatial
            # projection effect).
            #
            # recall:
            # θ_arcsec = r_AU / d_pc
            # mu_arcsec/yr = v_AU/yr / d_pc
            # v_AU/yr = mu_arcsec/yr * d_pc
            #
            d_pc = (nparr(1/(df.parallax*1e-3))*u.pc).value
            c_AU_per_yr = (
                ((( nparr(df[c]) - nparr(reference_df[c]) )*1e-3) * d_pc)
                *(u.AU/u.yr)
            )
            c_km_per_sec = c_AU_per_yr.to(u.km/u.second)

            new_key = 'delta_'+c+'_km_s'
            new_val = c_km_per_sec
            df[new_key] = new_val

            #
            # sophisticated thing: actually correct for the projection
            # effect.
            #
            vel_to_add = np.ones(len(df))*icrs_c_median.velocity
            _ = icrs_c_df.data.to_cartesian().with_differentials(vel_to_add)
            c_full = icrs_c_df.realize_frame(_)

            if c == 'pmra':
                c_AU_per_yr = (
                    ((( nparr(df[c]) - nparr(c_full.pm_ra_cosdec) )*1e-3) * d_pc)
                    *(u.AU/u.yr)
                )
            elif c == 'pmdec':
                c_AU_per_yr = (
                    ((( nparr(df[c]) - nparr(c_full.pm_dec) )*1e-3) * d_pc)
                    *(u.AU/u.yr)
                )

            c_km_per_sec = c_AU_per_yr.to(u.km/u.second)
            new_key = 'delta_'+c+'_prime_km_s'
            new_val = c_km_per_sec
            df[new_key] = new_val

    df['delta_mu_km_s'] = np.sqrt(
        df['delta_pmra_km_s']**2 + df['delta_pmdec_km_s']**2
    )
    df['delta_mu_prime_km_s'] = np.sqrt(
        df['delta_pmra_prime_km_s']**2 + df['delta_pmdec_prime_km_s']**2
    )

    return df


def given_gaia_df_get_icrs_arr(df, zero_rv=0, zero_allvelocities=0):
    """
    Given a pandas dataframe with [ra, dec, parallax, pmra, pmdec] and
    optionally [dr2_radial_velocity], return an initialized astropy SkyCoord
    instance of the corresponding ICRS coordinates.
    """

    if zero_allvelocities:
        return coord.SkyCoord(
            ra=nparr(df.ra)*u.deg,
            dec=nparr(df.dec)*u.deg,
            distance=nparr(1/(df.parallax*1e-3))*u.pc,
            pm_ra_cosdec=None,
            pm_dec=None,
            radial_velocity=None
        )

    if zero_rv:
        return coord.SkyCoord(
            ra=nparr(df.ra)*u.deg,
            dec=nparr(df.dec)*u.deg,
            distance=nparr(1/(df.parallax*1e-3))*u.pc,
            pm_ra_cosdec=nparr(df.pmra)*u.mas/u.yr,
            pm_dec=nparr(df.pmdec)*u.mas/u.yr,
            radial_velocity=0*u.km/u.s
        )

    rvkey = 'dr2_radial_velocity'
    if rvkey in df:
        pass
    else:
        rvkey = 'radial_velocity'
    return coord.SkyCoord(
        ra=nparr(df.ra)*u.deg,
        dec=nparr(df.dec)*u.deg,
        distance=nparr(1/(df.parallax*1e-3))*u.pc,
        pm_ra_cosdec=nparr(df.pmra)*u.mas/u.yr,
        pm_dec=nparr(df.pmdec)*u.mas/u.yr,
        radial_velocity=nparr(df[rvkey])*u.km/u.s
    )


def calc_dist(x0, y0, z0, x1, y1, z1):

    d = np.sqrt(
        (x0-x1)**2 + (y0-y1)**2 + (z0-z1)**2
    )

    return d


def calc_vl_vb_physical(ra, dec, pmra, pmdec, parallax,
                        gaia_datarelease='gaia_edr3'):
    """
    Given RA, DEC, pmRA, pmDEC, and parallax, calculate
    (v_l_cosb_km_per_sec, v_b_km_per_sec)
    the velocities along the galacitc directions in physical units of km/s.

    gaia_datarelease must be in ['gaia_edr3', 'gaia_dr3', 'gaia_dr2', 'none'],
    where "none" means the parallax offset correction is not applied.
    """

    from cdips.utils.gaiaqueries import parallax_to_distance_highsn

    sc = SkyCoord(
        ra*u.deg, dec*u.deg, pm_ra_cosdec=pmra*u.mas/u.yr,
        pm_dec=pmdec*u.mas/u.yr
    )
    pm_l_cosb = sc.galactic.pm_l_cosb.to(u.mas/u.yr)
    pm_b = sc.galactic.pm_b.to(u.mas/u.yr)

    d_pc = parallax_to_distance_highsn(
        parallax, gaia_datarelease=gaia_datarelease
    )

    pm_l_cosb_AU_per_yr = (pm_l_cosb.value*1e-3) * d_pc * (1*u.AU/u.yr)
    v_l_cosb_km_per_sec = pm_l_cosb_AU_per_yr.to(u.km/u.second)

    pm_b_AU_per_yr = (pm_b.value*1e-3) * d_pc * (1*u.AU/u.yr)
    v_b_km_per_sec = pm_b_AU_per_yr.to(u.km/u.second)

    return v_l_cosb_km_per_sec, v_b_km_per_sec


def get_vl_lsr_corr(lons, get_errs=False):
    """
    Given a vector of galactic longitudes, return the curve of
    longitudinal v_LSR.  (Mean, +1sigma, -1sigma).

    If get_errs is false still returns 3-tuple, but with nones.
    """

    from astropy.coordinates import (
        Galactic, CartesianDifferential
    )

    # Schonrich+2010 solar velocity wrt local standard of rest
    # (LSR) - note the systemic uncertainties are not included
    U = 11.10 * u.km/u.s
    U_hi = 0.69 * u.km/u.s
    U_lo = 0.75 * u.km/u.s
    V = 12.24 * u.km/u.s
    V_hi = 0.47 * u.km/u.s
    V_lo = 0.47 * u.km/u.s
    W = 7.25 * u.km/u.s
    W_hi = 0.37 * u.km/u.s
    W_lo = 0.36 * u.km/u.s

    _lat = 42   # doesn't matter
    _dist = 300 # doesn't matter

    lats = _lat*np.ones_like(lons)
    dists_pc = _dist*np.ones_like(lons)

    for _U, _V, _W, label in zip(
        [U, U+U_hi, U-U_lo],
        [V, V+V_hi, V-V_lo],
        [W, W+W_hi, W-W_lo],
        [0, 1, 2]
    ):
        print(42*'-')
        print(label)

        if not get_errs and label == 1:
            v_l_cosb_kms_upper = None
            continue
        if not get_errs and label == 2:
            v_l_cosb_kms_lower = None
            continue

        v_l_cosb_kms_list = []
        for ix, (lon, lat, dist_pc) in enumerate(zip(lons, lats, dists_pc)):
            #print(f'{ix}/{len(lons)}')
            gal = Galactic(lon*u.deg, lat*u.deg, distance=dist_pc*u.pc)
            vel_to_add = CartesianDifferential(_U, _V, _W)
            newdata = gal.data.to_cartesian().with_differentials(vel_to_add)
            newgal = gal.realize_frame(newdata)
            pm_l_cosb_AU_per_yr = (newgal.pm_l_cosb.value*1e-3) * dist_pc * (1*u.AU/u.yr)
            v_l_cosb_kms = pm_l_cosb_AU_per_yr.to(u.km/u.second)
            v_l_cosb_kms_list.append(v_l_cosb_kms.value)

        v_l_cosb_kms = -np.array(v_l_cosb_kms_list)

        if label == 0:
            v_l_cosb_kms_mid = v_l_cosb_kms*1.
        elif label == 1:
            v_l_cosb_kms_upper = v_l_cosb_kms*1.
        elif label == 2:
            v_l_cosb_kms_lower = v_l_cosb_kms*1.

    return [v_l_cosb_kms_mid,
            v_l_cosb_kms_upper,
            v_l_cosb_kms_lower]


def calculate_XYZ_given_RADECPLX(ra, dec, plx):
    """
    Given numpy arrays of right ascension, declination, and parallax, calculate
    the corresponding galactic XYZ coordinates.  NOTE: this code converts from
    parallax to distance assuming parallax [arcsec] = 1/distance[pc].  This
    assumption is wrong in the regime of low signal-to-noise parallaxes (very
    faint stars).

    Args:
        ra/dec/plx: np.ndarray of right ascension, declination, and parallax.
        RA/DEC in units of degrees.  Parallax in units of milliarcseconds.

    Returns:
        X, Y, Z: tuple of corresponding physical coordinates.  In this
        coordinate system the Sun is at {X,Y,Z}={-8122,0,+20.8} parsecs.
    """

    # convert from parallax to distance assuming high S/N parallaxes.
    import numpy as np
    from numpy import array as nparr
    import astropy.units as u

    from astropy.coordinates import Galactocentric
    import astropy.coordinates as coord
    _ = coord.galactocentric_frame_defaults.set('v4.0')

    distance = nparr(1/(plx*1e-3))*u.pc

    _c = coord.SkyCoord(
        ra=nparr(ra)*u.deg, dec=nparr(dec)*u.deg, distance=distance,
        pm_ra_cosdec=None, pm_dec=None, radial_velocity=None
    )

    # get XYZ in galactocentric
    c = _c.transform_to(coord.Galactocentric())

    x = c.x.to(u.pc).value
    y = c.y.to(u.pc).value
    z = c.z.to(u.pc).value

    return x,y,z
