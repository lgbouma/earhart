"""
Given a list of Gaia DR2 stars and the "sobject_ids" from the GALAH
DR3 main table, get the corresponding GALAH DR3 spectra.

These were made via earhart.plotting.plot_galah_dr3_lithium_abundance,
which compared the GALAH DR3 table against the fullfaint sample:

    ```
    Number of comparison stars: 3298
    Number of comparison stars w/ Galah DR3 matches 107
    Number of comparison stars w/ Galah DR3 matches in core 51
    Number of comparison stars w/ Galah DR3 matches in halo 56
    Number of comparison stars w/ Galah DR3 matches and finite lithium (detection or limit): 78
    ```
Per
    https://docs.datacentral.org.au/galah/dr3/spectra-data-access/
Get the spectra by running this script, and then pasting the results to
    https://datacentral.org.au/services/download/
"""

import pandas as pd, numpy as np

# via earhart.plotting.plot_galah_dr3_lithium
inpath = '/Users/luke/Dropbox/proj/earhart/results/lithium/kinematic_X_galah_dr3.csv'
df = pd.read_csv(inpath)

sobject_ids = [i for i in df.sobject_id]

# the asterisk decomposes the list into separate arguments
print('Copy paste the following to https://datacentral.org.au/services/download/')
print(*sobject_ids, sep=', ')
