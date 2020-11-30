import os
import numpy as np, pandas as pd
from astropy.io import fits
from astropy.table import Table

from earhart.paths import DATADIR, RESULTSDIR

def get_GalahDR3_lithium(verbose=1):
    """
    Get astropy table of stellar parameter flag == 0 stars from GALAH DR3, with
    lithium detections or upper limits.
    """

    # downloaded via wget, per
    # https://github.com/svenbuder/GALAH_DR3/blob/master/tutorials/tutorial1_dr3_main_catalog_overview.ipynb
    dr3path = os.path.join(DATADIR, 'lithium', 'GALAH_DR3_main_allstar_v1.fits')

    dr3_tab = Table.read(dr3path)

    unflagged_stellar_parameters = (dr3_tab['flag_sp'] == 0)

    Li_detections_and_upper_limits = (dr3_tab['flag_Li_fe'] <= 1)

    sel = (
        unflagged_stellar_parameters & Li_detections_and_upper_limits
    )

    if verbose:
        print(f'All stars in GALAH DR3: {len(dr3_tab)}')
        print(f'All stars in GALAH DR3 with stellar parameter flag == 0: {len(dr3_tab[unflagged_stellar_parameters])}')
        print(f'All stars in GALAH DR3 with stellar parameter flag == 0 and lithium detection or upper limit: {len(dr3_tab[sel])}')

    return dr3_tab[sel]

def get_Randich18_NGC2516():

    hl = fits.open(
        os.path.join(DATADIR, 'lithium',
                     'Randich_2018_NGC2516_all796_entries_vizier.fits')
    )

    t_df = Table(hl[1].data).to_pandas()

    return t_df
