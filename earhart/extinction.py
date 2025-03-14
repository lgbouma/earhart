"""
Tools for working out the extinction.

EBmV_to_AV
AV_to_EBpmRp
Bonifacio2000_EBmV_correction
get_dist_corr
given_S98_EBmV_correct
"""

import numpy as np

def EBmV_to_AV(EBmV):
    R_V = 3.1
    A_V = R_V * EBmV
    return A_V

def AV_to_EBmV(A_V):
    R_V = 3.1
    EBmV = A_V / R_V
    return EBmV

def EBmV_to_AG(EBmV):
    # Stassun+2019 TIC8
    return 2.72*EBmV

def AV_to_EBpmRp(A_V):
    """
    Convert A_V to E(Bp-Rp). NOTE: assumes A_V has been "corrected" for
    distance, galactic latitude, etc. So if you pull A_V from a total
    extinction in a LOS map (e.g., SF98) that doesn't do this correction,
    you'll get wrong answers.
    """
    # E(B-V) = A_V/R_V
    R_V = 3.1
    E_BmV = A_V/R_V
    # Stassun+2019 calibration
    E_BpmRp = 1.31*E_BmV
    return E_BpmRp


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
