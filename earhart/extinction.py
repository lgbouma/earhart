"""
Tools for working out the extinction.
"""

import numpy as np

def Bonifacio2000_EBmV_correction(EBmV):
    """
    Bonifacio+2000 note that Schlegel+98 overestimate the reddening values when
    the color excess exceeds about 0.10 mag.  This applies their Equation 1 to
    "correct" this.
    """
    EBmV = np.array(EBmV)
    sel = (EBmV > 0.10)
    EBmV[sel] = (
        0.10 + 0.65 * (EBmV[sel] - 0.10)
    )
    EBmV_adopted = EBmV
    return EBmV_adopted


def get_dist_corr(distance_pc, b, h=125):
    # h = 125pc, scale height of MW
    return (
            1 - np.exp(
                - np.abs( distance_pc * np.sin(np.deg2rad(b)) ) / h
            )
    )


def given_S98_EBmV_correct(EBmV, distance_pc=None, b=None):
    """
    EBmV: float or array of E(B-V) color excess values.
    distance_pc: float of distance to each EBmV value.
    b: galactic latitudes of the same, in units of degrees.
    """

    if distance_pc is None:
        return Bonifacio2000_EBmV_correction(EBmV)

    elif isinstance(distance_pc, float) and isinstance(b, float):
        h = 125 # pc; scale height of the milky way
        dist_corr = get_dist_corr(distance_pc, b)
        return dist_corr * Bonifacio2000_EBmV_correction(EBmV)
