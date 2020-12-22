"""
[Still WIP; these demos were sketched out in an attempt to crossmatch Gaia DR2
source_ids against TIC8 in a vectorized way. It didn't work!]

Vizier TIC columns:

['TIC', 'RAJ2000', 'DEJ2000', 'HIP', 'TYC', 'UCAC4', '_2MASS', 'objID',
'WISEA', 'GAIA', 'APASS', 'KIC', 'S_G', 'Ref', 'r_Pos', 'pmRA',
'e_pmRA', 'pmDE', 'e_pmDE', 'r_pm', 'Plx', 'e_Plx', 'r_Plx', 'GLON',
'GLAT', 'ELON', 'ELAT', 'Bmag', 'e_Bmag', 'u_e_Bmag', 'Vmag', 'e_Vmag',
'u_e_Vmag', 'umag', 'e_umag', 'gmag', 'e_gmag', 'rmag', 'e_rmag',
'imag', 'e_imag', 'zmag', 'e_zmag', 'Jmag', 'e_Jmag', 'Hmag', 'e_Hmag',
'Kmag', 'e_Kmag', 'q_2MASS', 'W1mag', 'e_W1mag', 'W2mag', 'e_W2mag',
'W3mag', 'e_W3mag', 'W4mag', 'e_W4mag', 'Gmag', 'e_Gmag', 'Tmag',
'e_Tmag', 'f_Tmag', 'Flag', 'Teff', 's_Teff', 'logg', 's_logg',
'__M_H_', 'e__M_H_', 'Rad', 's_Rad', 'Mass', 's_Mass', 'rho', 's_rho',
'LClass', 'Lum', 's_Lum', 'Dist', 's_Dist', 'E_B-V_', 's_E_B-V_',
'Ncont', 'Rcont', 'Disp', 'm_TIC', 'Prior', 'e_E_B-V_', 'E_E_B-V_',
'f_E_B-V_', 'e_Mass', 'E_Mass', 'e_Rad', 'E_Rad', 'e_rho', 'E_rho',
'e_logg', 'E_logg', 'e_Lum', 'E_Lum', 'e_Dist', 'E_Dist', 'r_Dist',
'e_Teff', 'E_Teff', 'r_Teff', 'BPmag', 'e_BPmag', 'RPmag', 'e_RPmag',
'q_Gaia', 'r_Vmag', 'r_Bmag', 'Clist', 'e_RAJ2000', 'e_DEJ2000',
'RAOdeg', 'DEOdeg', 'e_RAOdeg', 'e_DEOdeg', 'RadFl', 'WDFl', 'ID']
"""

from cdips.utils.tapqueries import given_source_ids_get_tic8_data

# from astroquery.utils.tap.core import TapPlus
# from cdips.utils.gaiaqueries import given_votable_get_df
# 
# dlpath = '/Users/luke/Dropbox/proj/earhart/drivers/temp.xml.gz'
# 
# tap = TapPlus(url="http://TAPVizieR.u-strasbg.fr/TAPVizieR/tap")
# 
# 
# jobstr = (
# '''
# SELECT top 10
#     *
# FROM "IV/38/tic" as t
# '''
# )
# 
# query = jobstr
# 
# j = tap.launch_job(query=query, verbose=True, dump_to_file=True,
#                    output_file=dlpath)
# 
# 
# df = given_votable_get_df(dlpath, assert_equal=None)

import IPython; IPython.embed()

# # might do async if this times out. but it doesn't.
# j = Gaia.launch_job(query=query,
#                     upload_resource=xmltouploadpath,
#                     upload_table_name="foobar", verbose=True,
#                     dump_to_file=True, output_file=dlpath)
# 


