"""
Data-getters:
    get_hygdata: All stars in Hipparcos, Yale Bright Star, and Gliese catalogs
    get_asterisms: Stellarium database asterisms
    get_constellation_boundaries: IAU constellation boundaries
    get_messier_data: The Messier catalog

All using data parsed from Eleanor Lutz's absolutely amazing
https://github.com/eleanorlutz/western_constellations_atlas_of_space
"""
import os
import numpy as np, pandas as pd

from astropy.coordinates import SkyCoord
from astropy import units as u, constants as const

from earhart.paths import DATADIR, RESULTSDIR
PROCESSED_DIR = os.path.join(DATADIR, 'skyfield', 'processed')

def radec_to_lb(ra,dec):
    c = SkyCoord(ra*u.deg, dec*u.deg, frame='icrs')
    return c.galactic.l.value, c.galactic.b.value


def get_const_names():
    const_names = pd.read_csv(
        os.path.join(PROCESSED_DIR,'centered_constellations.csv'),
        encoding="latin-1"
    )
    return const_names



def get_hygdata():
    """
    HYG 3.0: Database containing all stars in Hipparcos, Yale Bright Star, and
    Gliese catalogs (almost 120,000 stars, 14 MB)

    Acquired originally via via https://github.com/astronexus/HYG-Database.
    """

    # in 1_process_starbase_data.ipynb, Lutz removed the sun, tarnslated
    # plaintext Bayer designations into greek letters, and applied a
    # spectral-type to color mapper.
    # this gave the "processed" HYG stars. there's also the option of getting
    # the mag <=6.5 stars, i.e., the naked eye ones.

    orig_hygpath = os.path.join(
        DATADIR, 'skyfield', 'hygdata_v3', 'hygdata_v3.csv'
    )

    proc_hygpath = os.path.join(
        PROCESSED_DIR, 'hygdata_processed.csv'
    )

    m65proc_hygpath = os.path.join(
        PROCESSED_DIR, 'hygdata_processed_mag65.csv'
    )

    proc_df = pd.read_csv(proc_hygpath, low_memory=False)
    m65proc_df = pd.read_csv(m65proc_hygpath, low_memory=False)

    return proc_df, m65proc_df


def get_asterisms():
    """
    Get (cleaned) asterisms from Stellarium database.
    """

    asterism_path = os.path.join(
        PROCESSED_DIR, 'asterisms.csv'
    )

    return pd.read_csv(asterism_path)


def get_constellation_boundaries():
    """
    Get IAU constellation boundaries from 989 Catalogue of Constellation
    Boundary Data by A.C. Davenhall and S.K. Leggett
    http://cdsarc.u-strasbg.fr/viz-bin/Cat?VI/49#sRM2.2
    """

    csvpath = os.path.join(
        PROCESSED_DIR, 'constellations.csv'
    )

    return pd.read_csv(csvpath)


def get_messier_data():
    """
    Literally, the messier catalog. Brightnesses, names, positions, types.
    """

    csvpath = os.path.join(
        PROCESSED_DIR, 'messier_ngc_processed.csv'
    )

    return pd.read_csv(csvpath)
