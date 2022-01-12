"""
Tools for working in XYZ, UVW, and Δμ_δ, Δμ_α' [km/s] coordinates.

Contents:
    append_physicalpositions: (α,δ,π,μ_α,μ_δ)->(X,Y,Z,v*_tang).
    given_gaia_df_get_icrs_arr: DataFrame -> SkyCoord conversion
    calc_dist: between two 3d cartesian points.
"""

import numpy as np
from numpy import array as nparr
import astropy.units as u
import astropy.constants as const

from astropy.coordinates import Galactocentric
import astropy.coordinates as coord
_ = coord.galactocentric_frame_defaults.set('v4.0')

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



