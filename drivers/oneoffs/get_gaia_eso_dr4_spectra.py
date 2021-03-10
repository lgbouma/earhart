"""
Given the Gaia-ESO DR4 metadata table, which I from
http://archive.eso.org/programmatic/#TAP, by doing

    SELECT access_url, dp_id, em_max, em_min, em_res_power, em_xel, facility_name,
    gal_lat, gal_lon, instrument_name, obs_id, obs_release_date, s_dec, s_ra,
    obs_title, target_name FROM ivoa.ObsCore
    WHERE obs_collection='GAIAESO'
    AND publication_date between  '2020-12-09' and  '2020-12-10'

Then crossmatch against the "fullfaint" CG18+KC19+M21 NGC 2516 source list and
get all matching spectra.
"""

from astropy.io import fits
from astropy.table import Table
import pandas as pd, numpy as np
from earhart.paths import DATADIR
from earhart.helpers import get_gaia_basedata
from earhart.xmatch import xmatch_dataframes_using_radec


basedata = 'fullfaint'
nbhd_df, core_df, halo_df, full_df, trgt_df = get_gaia_basedata(basedata)

# made via the query in this driver's docstring.
inpath = '/Users/luke/Dropbox/proj/earhart/data/gaiaeso_dr4_spectra/gaiaeso_dr4_meta.fits'
gdf = Table(fits.open(inpath)[1].data).to_pandas()
assert len(gdf) == 190200 # agrees with the Gaia-ESO DR4 release notes

sel = (gdf.s_dec < -40) & (gdf.s_ra > 105) & (gdf.s_ra < 135)
gdf = gdf[sel]
print(f'Beginning xmatch vs {len(gdf)} Gaia-ESO positions...')

outpath = '/Users/luke/Dropbox/proj/earhart/data/gaiaeso_dr4_spectra/gaiaeso_dr4_meta_X_fullfaint_kinematic.csv'

if not os.path.exists(outpath):
    mdf = xmatch_dataframes_using_radec(
        gdf, full_df, 's_ra', 's_dec', 'ra', 'dec', RADIUS=0.5
    )
    mdf.to_csv(outpath, index=False)
else:
    mdf = pd.read_csv(outpath)


import IPython; IPython.embed()


