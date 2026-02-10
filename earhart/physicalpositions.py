"""
Tools for working in XYZ, UVW, and Δμ_δ, Δμ_α' [km/s] coordinates.

Contents:
    append_physicalpositions: (α,δ,π,μ_α,μ_δ)->(X,Y,Z,v*_tang).
    given_gaia_df_get_icrs_arr: DataFrame -> SkyCoord conversion
    calc_dist: between two 3d cartesian points.
    calc_vl_vb_physical: on-sky galactic velocity conversion
    get_vl_lsr_corr: get vLSR correction given glon
    calculate_XYZ_given_RADECPLX: (α,δ,π)->(X,Y,Z).
    calculate_XYZUVW_given_RADECPLXPMRV: (α,δ,π,μ_α,μ_δ,RV)->(X,Y,Z,U,V,W).
    calculate_RPHIZ_given_RADECPLXPMRV: cylindrical (α,δ,π,μ_α,μ_δ,RV)->(R,φ,z,v_r,v_φ,v_z).
    proper_motion_position_angle_deg: get PM position angle wrt. equatorial N & E
    proper_motion_position_angle_galactic_deg: ditto, wrt. galactic N & E
    proper_motion_to_tangential_velocity: (μ_α,μ_δ,π)->(v_α,v_δ)
"""

from __future__ import annotations

import numpy as np

from typing import Union, Any, Optional, Tuple
ArrayLike = Union[float, int, np.ndarray]

from numpy import array as nparr
import astropy.units as u
import astropy.constants as const

from astropy.coordinates import Galactocentric, SkyCoord, Galactic, ICRS
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
    faint stars).  Negative distances become NaNs.

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

    # Convert inputs to numpy arrays
    ra = nparr(ra)
    dec = nparr(dec)
    plx = nparr(plx)

    # Initialize output arrays with NaNs
    x = np.full_like(ra, np.nan, dtype=float)
    y = np.full_like(ra, np.nan, dtype=float)
    z = np.full_like(ra, np.nan, dtype=float)

    # Create a mask for valid parallaxes
    valid = plx > 0

    if np.any(valid):
        # Process only valid entries
        # Convert from parallax to distance assuming high S/N parallaxes.
        distance = (1 / (plx[valid] * 1e-3)) * u.pc

        # Create a SkyCoord object with the provided data
        _c = coord.SkyCoord(
            ra=ra[valid] * u.deg,
            dec=dec[valid] * u.deg,
            distance=distance,
            pm_ra_cosdec=None,
            pm_dec=None,
            radial_velocity=None
        )

        # Transform to the Galactocentric frame
        c = _c.transform_to(coord.Galactocentric())

        # Extract galactic positions
        x_valid = c.x.to(u.pc).value
        y_valid = c.y.to(u.pc).value
        z_valid = c.z.to(u.pc).value

        # Place the valid computed values into the output arrays
        x[valid] = x_valid
        y[valid] = y_valid
        z[valid] = z_valid

    # For entries where plx <= 0, outputs remain as NaN

    return x, y, z


def calculate_XYZUVW_given_RADECPLXPMRV(ra, dec, plx, pm_ra, pm_dec, radial_velocity):
    """
    NOTE: This function returns UVW values different from the Gagne+
    convention, and I am not sure why.

    Given numpy arrays of right ascension, declination, parallax, proper
    motions, and radial velocities, calculate the corresponding galactic XYZ
    positions and UVW velocities. NOTE: this code converts from parallax to
    distance assuming parallax [arcsec] = 1/distance[pc]. This assumption is
    incorrect in the regime of low signal-to-noise parallaxes (very faint
    stars).

    Args:
        ra (np.ndarray): Right ascension in degrees.
        dec (np.ndarray): Declination in degrees.
        plx (np.ndarray): Parallax in milliarcseconds.
        pm_ra (np.ndarray): Proper motion in right ascension (mas/yr).
        pm_dec (np.ndarray): Proper motion in declination (mas/yr).
        radial_velocity (np.ndarray): Radial velocity in km/s.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X, Y, Z positions in parsecs and U, V, W velocities in km/s.
        In this coordinate system, the Sun is at {X, Y, Z} = {-8122, 0, +20.8} parsecs.
        U is positive towards the Galactic center, V is positive in the
        direction of Galactic rotation, and W is positive towards the north
        Galactic pole.

        The returned UVW system is set in a manner that subtracts out the Sun's
        peculiar motion relative to the LSR (Schonrich+2010), and in a way that
        subtracts ou tthe circular velocity at the solar radius (assuming
        238km/s, astropy4.0 default).
    """
    print(42*'-')
    print(
    "WARNING: calculate_XYZUVW_given_RADECPLXPMRV returns UVW values " +
    "different from the Gagne+ convention, and I am not sure why."
    )
    print(42*'-')


    import numpy as np
    from numpy import array as nparr
    import astropy.units as u
    from astropy.coordinates import Galactocentric
    import astropy.coordinates as coord

    # Set the Galactocentric frame parameters to default v4.0 (appropriate for Gaia data)
    _ = coord.galactocentric_frame_defaults.set('v4.0')

    # Convert inputs to numpy arrays
    ra = np.array(ra)
    dec = np.array(dec)
    plx = np.array(plx)
    pm_ra = np.array(pm_ra)
    pm_dec = np.array(pm_dec)
    radial_velocity = np.array(radial_velocity)

    # Initialize output arrays with NaNs
    x = np.full_like(ra, np.nan, dtype=float)
    y = np.full_like(ra, np.nan, dtype=float)
    z = np.full_like(ra, np.nan, dtype=float)
    U_LSR = np.full_like(ra, np.nan, dtype=float)
    V_LSR = np.full_like(ra, np.nan, dtype=float)
    W_LSR = np.full_like(ra, np.nan, dtype=float)

    # Create a mask for valid parallaxes
    valid = plx > 0

    if np.any(valid):
        # Process only valid entries
        # Convert from parallax to distance assuming high S/N parallaxes.
        distance = (1 / (plx[valid] * 1e-3)) * u.pc

        # Convert proper motions and radial velocity to astropy quantities
        pm_ra_cosdec = pm_ra[valid] * u.mas / u.yr
        pm_dec_valid = pm_dec[valid] * u.mas / u.yr
        rv = radial_velocity[valid] * u.km / u.s

        # Create a SkyCoord object with the provided data
        _c = coord.SkyCoord(
            ra=ra[valid] * u.deg,
            dec=dec[valid] * u.deg,
            distance=distance,
            pm_ra_cosdec=pm_ra_cosdec,
            pm_dec=pm_dec_valid,
            radial_velocity=rv,
            frame='icrs'
        )

        # Transform to the Galactocentric frame
        c = _c.transform_to(coord.Galactocentric())

        # Extract galactic positions
        x_valid = c.x.to(u.pc).value
        y_valid = c.y.to(u.pc).value
        z_valid = c.z.to(u.pc).value

        # Extract galactic velocities
        vx = c.v_x.to(u.km / u.s).value
        vy = c.v_y.to(u.km / u.s).value
        vz = c.v_z.to(u.km / u.s).value

        # Compute U, V, W velocities (U is towards the Galactic center)
        U = -vx  # U positive towards Galactic center
        V = vy   # V positive in direction of Galactic rotation
        W = vz   # W positive towards North Galactic Pole

        # Sun's peculiar motion relative to the LSR, plus the circular motion
        # (~240km/s).  Uses astropy 4.0 defaults
        U_sun = 12.9   # km/s
        V_sun = 245.6  # km/s
        W_sun = 7.78   # km/s

        # Correct for the Sun's peculiar motion
        U_LSR_valid = U - U_sun
        V_LSR_valid = V - V_sun
        W_LSR_valid = W - W_sun

        # Place the valid computed values into the output arrays
        x[valid] = x_valid
        y[valid] = y_valid
        z[valid] = z_valid
        U_LSR[valid] = U_LSR_valid
        V_LSR[valid] = V_LSR_valid
        W_LSR[valid] = W_LSR_valid

    # For entries where plx <= 0, outputs remain as NaN

    return x, y, z, U_LSR, V_LSR, W_LSR


def proper_motion_position_angle_deg(pmra: ArrayLike,
                                     pmdec: ArrayLike) -> np.ndarray:
    """Compute proper-motion position angle (PA) in the standard convention.

    The PA is measured in degrees East of North (from +Dec toward +RA),
    on the tangent plane. Inputs should be Gaia-style components:
    pmra = mu_alpha* = (d(alpha)/dt) * cos(delta), and pmdec = mu_delta.

    Returns an array of PA values in [0, 360). If both components are zero
    for an element, that element is set to np.nan.

    Args:
        pmra: Proper motion in RA, mu_alpha* (mas/yr); scalar or array.
        pmdec: Proper motion in Dec, mu_delta (mas/yr); scalar or array.

    Returns:
        Numpy array of position angles (deg) East of North, in [0, 360).
    """
    pmra_arr = np.asarray(pmra, dtype=float)
    pmdec_arr = np.asarray(pmdec, dtype=float)

    pa_deg = (np.degrees(np.arctan2(pmra_arr, pmdec_arr)) % 360.0)

    zero_mask = (pmra_arr == 0.0) & (pmdec_arr == 0.0)
    if np.any(zero_mask):
        pa_deg = np.array(pa_deg, copy=True)
        pa_deg[zero_mask] = np.nan

    return pa_deg


def calculate_RPHIZ_given_RADECPLXPMRV(
    ra_deg: Any,
    dec_deg: Any,
    plx_mas: Any,
    pmra_masyr: Any,
    pmdec_masyr: Any,
    rv_kms: Any,
    galactocentric_frame: Optional[Galactocentric] = None,
    phi_unit: u.UnitBase = u.deg,
) -> Tuple[u.Quantity, u.Quantity, u.Quantity, u.Quantity, u.Quantity,
           u.Quantity]:
    """Convert ICRS 6D phase space to Galactocentric cylindrical coordinates.

    Outputs are Galactocentric cylindrical (R, phi, z) and corresponding
    velocities (v_R, v_phi, v_z), using Astropy's Galactocentric frame.

    Parallax is converted to distance via d[pc] = 1 / plx[arcsec]. For
    non-positive parallax values, distance becomes NaN and outputs will be NaN.

    Generated by ChatGPT GPT-5.2 Thinking on February 10 2026.

    Args:
        ra_deg: ICRS right ascension (deg); scalar or array-like.
        dec_deg: ICRS declination (deg); scalar or array-like.
        plx_mas: Parallax (mas); scalar or array-like.
        pmra_masyr: Proper motion in RA, mu_alpha* (mas/yr).
        pmdec_masyr: Proper motion in Dec, mu_delta (mas/yr).
        rv_kms: Radial velocity (km/s); scalar or array-like.
        galactocentric_frame: Optional Astropy Galactocentric frame instance
            controlling Solar position/velocity assumptions. If None, uses
            Galactocentric() defaults.
        phi_unit: Unit for returned azimuth angle (u.deg or u.rad).

    Returns:
        r: Cylindrical radius R (kpc).
        phi: Azimuth angle (phi_unit).
        z: Height z (kpc).
        v_r: Radial velocity v_R (km/s).
        v_phi: Azimuthal velocity v_phi (km/s).
        v_z: Vertical velocity v_z (km/s).
    """
    plx = np.asarray(plx_mas, dtype=float) * u.mas
    plx_arcsec = plx.to(u.arcsec).value
    dist_pc = np.where(plx_arcsec > 0.0, 1.0 / plx_arcsec, np.nan) * u.pc

    from astropy.coordinates import Galactocentric
    import astropy.coordinates as coord
    _ = coord.galactocentric_frame_defaults.set('v4.0')

    galcen = galactocentric_frame if galactocentric_frame is not None else (
        Galactocentric()
    )

    c_icrs = SkyCoord(
        ra=np.asarray(ra_deg, dtype=float) * u.deg,
        dec=np.asarray(dec_deg, dtype=float) * u.deg,
        distance=dist_pc,
        pm_ra_cosdec=np.asarray(pmra_masyr, dtype=float) * u.mas / u.yr,
        pm_dec=np.asarray(pmdec_masyr, dtype=float) * u.mas / u.yr,
        radial_velocity=np.asarray(rv_kms, dtype=float) * u.km / u.s,
        frame="icrs",
    )

    c_galcen = c_icrs.transform_to(galcen)
    cyl = c_galcen.cylindrical

    r = cyl.rho.to(u.kpc)
    phi = cyl.phi.to(phi_unit)
    z = cyl.z.to(u.kpc)

    v_cart = c_galcen.velocity
    cyl_vel = v_cart.represent_as("cylindrical").differentials["s"]
    v_r = cyl_vel.d_rho.to(u.km / u.s)
    v_phi = (cyl.rho * cyl_vel.d_phi).to(u.km / u.s)
    v_z = cyl_vel.d_z.to(u.km / u.s)

    return r, phi, z, v_r, v_phi, v_z


def proper_motion_position_angle_galactic_deg(
    ra_deg: Any,
    dec_deg: Any,
    pmra_masyr: Any,
    pmdec_masyr: Any,
) -> np.ndarray:
    """Compute proper-motion position angle after rotating into Galactic (l, b).

    This returns the position angle of the proper-motion vector expressed in
    Galactic components (mu_l* and mu_b), measured in degrees East of North in
    Galactic coordinates (from +b toward +l), i.e. PA = atan2(mu_l*, mu_b).

    The inputs are ICRS ra/dec plus Gaia-style pm components:
    pmra = mu_alpha* = (d(alpha)/dt) * cos(dec), pmdec = mu_delta.

    Elements with both (mu_l*, mu_b) == (0, 0) are returned as np.nan.

    Generated by ChatGPT GPT-5.2 Thinking on February 10 2026.

    Args:
        ra_deg: ICRS right ascension (deg); scalar or array-like.
        dec_deg: ICRS declination (deg); scalar or array-like.
        pmra_masyr: Proper motion in RA, mu_alpha* (mas/yr); scalar/array.
        pmdec_masyr: Proper motion in Dec, mu_delta (mas/yr); scalar/array.

    Returns:
        Numpy array of PA values (deg) East of North in Galactic coordinates,
        in [0, 360).
    """
    c = SkyCoord(
        ra=np.asarray(ra_deg, dtype=float) * u.deg,
        dec=np.asarray(dec_deg, dtype=float) * u.deg,
        pm_ra_cosdec=np.asarray(pmra_masyr, dtype=float) * u.mas / u.yr,
        pm_dec=np.asarray(pmdec_masyr, dtype=float) * u.mas / u.yr,
        frame="icrs",
    ).galactic

    mu_l_cosb = c.pm_l_cosb.to_value(u.mas / u.yr)
    mu_b = c.pm_b.to_value(u.mas / u.yr)

    pa_deg = (np.degrees(np.arctan2(mu_l_cosb, mu_b)) % 360.0)

    zero_mask = (mu_l_cosb == 0.0) & (mu_b == 0.0)
    if np.any(zero_mask):
        pa_deg = np.array(pa_deg, copy=True)
        pa_deg[zero_mask] = np.nan

    return pa_deg

def residual_proper_motion_lsr(
    ra_deg: Any,
    dec_deg: Any,
    pmra_masyr: Any,
    pmdec_masyr: Any,
    parallax_mas: Any,
    U_sun: float = 12.9,
    V_sun: float = 245.6,
    W_sun: float = 7.78,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute residual proper motions after subtracting Solar reflex motion.

    Assumes stars are comoving with the LSR and ignores differential rotation
    (appropriate within ~500 pc). Uses a constant Solar velocity (U, V, W)
    in Galactic Cartesian (U to GC, V to rotation, W to NGP). The expected
    proper motion is the reflex of this Solar velocity.

    Generated by ChatGPT GPT-5.2 Thinking on February 10 2026.

    Args:
        ra_deg: ICRS right ascension (deg); scalar or array-like.
        dec_deg: ICRS declination (deg); scalar or array-like.
        pmra_masyr: Observed mu_alpha* (mas/yr); scalar or array-like.
        pmdec_masyr: Observed mu_delta (mas/yr); scalar or array-like.
        parallax_mas: Parallax (mas); scalar or array-like.
        U_sun: Solar U velocity (km/s), positive toward Galactic center.
        V_sun: Solar V velocity (km/s), positive in direction of rotation.
        W_sun: Solar W velocity (km/s), positive toward North Galactic Pole.

    Returns:
        pmra_res: Residual mu_alpha* (mas/yr).
        pmdec_res: Residual mu_delta (mas/yr).
        pm_res_amp: Residual proper-motion amplitude (mas/yr).
        pa_res_deg: Position angle of residual PM (deg E of N), in [0, 360).
    """
    ra, dec, pmra, pmdec, plx = np.broadcast_arrays(
        np.asarray(ra_deg, dtype=float),
        np.asarray(dec_deg, dtype=float),
        np.asarray(pmra_masyr, dtype=float),
        np.asarray(pmdec_masyr, dtype=float),
        np.asarray(parallax_mas, dtype=float),
    )

    plx_arcsec = (plx * u.mas).to_value(u.arcsec)
    dist_pc = np.where(plx_arcsec > 0.0, 1.0 / plx_arcsec, np.nan) * u.pc
    dist_kpc = dist_pc.to_value(u.kpc)

    v_reflex_gal = np.array(
        [-U_sun, -V_sun, -W_sun],
        dtype=float,
    ) * (u.km / u.s)

    v_vec_icrs = (
        SkyCoord(
            u=0.0 * u.pc,
            v=0.0 * u.pc,
            w=0.0 * u.pc,
            U=v_reflex_gal[0],
            V=v_reflex_gal[1],
            W=v_reflex_gal[2],
            frame="galactic",
            representation_type="cartesian",
        )
        .transform_to(ICRS())
        .velocity.d_xyz.to_value(u.km / u.s)
    )

    v_x, v_y, v_z = (float(v_vec_icrs[0]), float(v_vec_icrs[1]),
                     float(v_vec_icrs[2]))

    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)

    sin_ra = np.sin(ra_rad)
    cos_ra = np.cos(ra_rad)
    sin_dec = np.sin(dec_rad)
    cos_dec = np.cos(dec_rad)

    e_ra_x = -sin_ra
    e_ra_y = cos_ra
    e_ra_z = 0.0

    e_dec_x = -sin_dec * cos_ra
    e_dec_y = -sin_dec * sin_ra
    e_dec_z = cos_dec

    v_ra = v_x * e_ra_x + v_y * e_ra_y + v_z * e_ra_z
    v_dec = v_x * e_dec_x + v_y * e_dec_y + v_z * e_dec_z

    conv = 4.74047  # km/s per (mas/yr * kpc)
    mu_ra_exp = v_ra / (conv * dist_kpc)
    mu_dec_exp = v_dec / (conv * dist_kpc)

    mu_ra_exp = np.where(np.isfinite(dist_kpc), mu_ra_exp, np.nan)
    mu_dec_exp = np.where(np.isfinite(dist_kpc), mu_dec_exp, np.nan)

    pmra_res = pmra - mu_ra_exp
    pmdec_res = pmdec - mu_dec_exp

    pm_res_amp = np.sqrt(pmra_res**2 + pmdec_res**2)

    pa_res_deg = (np.degrees(np.arctan2(pmra_res, pmdec_res)) % 360.0)
    zero_mask = (pmra_res == 0.0) & (pmdec_res == 0.0)
    if np.any(zero_mask):
        pa_res_deg = np.array(pa_res_deg, copy=True)
        pa_res_deg[zero_mask] = np.nan

    return pmra_res, pmdec_res, pm_res_amp, pa_res_deg


def proper_motion_to_tangential_velocity(
    pmra_masyr: Any,
    pmdec_masyr: Any,
    parallax_mas: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert proper-motion components to tangential velocity components.

    Uses v_t[km/s] = 4.74047 * mu[mas/yr] * d[kpc], with d from parallax:
    d[pc] = 1 / plx[arcsec]. Non-positive parallaxes yield NaNs.

    Generated by ChatGPT GPT-5.2 Thinking on February 10 2026.

    Args:
        pmra_masyr: mu_alpha* (mas/yr); scalar or array-like.
        pmdec_masyr: mu_delta (mas/yr); scalar or array-like.
        parallax_mas: Parallax (mas); scalar or array-like.

    Returns:
        v_ra_kms: Tangential velocity along +RA direction (km/s).
        v_dec_kms: Tangential velocity along +Dec direction (km/s).
    """
    pmra, pmdec, plx = np.broadcast_arrays(
        np.asarray(pmra_masyr, dtype=float),
        np.asarray(pmdec_masyr, dtype=float),
        np.asarray(parallax_mas, dtype=float),
    )

    plx_arcsec = plx / 1000.0
    dist_kpc = np.where(plx_arcsec > 0.0, (1.0 / plx_arcsec) / 1000.0, np.nan)

    conv = 4.74047  # km/s per (mas/yr * kpc)
    v_ra_kms = conv * pmra * dist_kpc
    v_dec_kms = conv * pmdec * dist_kpc

    return v_ra_kms, v_dec_kms
