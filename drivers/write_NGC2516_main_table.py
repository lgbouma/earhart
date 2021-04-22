"""
Once cluster_rotation.get_auto_rotation_periods has been run, this script
populates results.tables with the table containing:

    * Rotation period + neighboring star count information.
    * Subcluster information.
    * "Cleaned subset" information ("subset A" or "subset B" of cleaning in the
    paper's terminology).
    * Lithium information.
"""
import os
import numpy as np, pandas as pd
from earhart.paths import DATADIR, RESULTSDIR
from earhart.helpers import (
    get_autorotation_dataframe, append_phot_binary_column, get_gaia_basedata
)
from earhart.lithium import _get_lithium_EW_df

include_lithium = False #FIXME: need this to be true!
N_members = 3298

# made by cluster_rotation.get_auto_rotation_periods
# already contains most information except for "clean subset" information.
df = pd.read_csv(
    os.path.join(DATADIR, 'rotation', 'NGC_2516_rotation_periods.csv')
)
assert len(df) == N_members

from earhart.priors import AVG_EBpmRp
df['(Bp-Rp)_0'] = df.phot_bp_mean_mag - df.phot_rp_mean_mag - AVG_EBpmRp

cleanings = ['defaultcleaning', 'periodogram_match', 'match234_alias']

# add the "in_{CLEANING}" columns
for c in cleanings:
    df_A = get_autorotation_dataframe('NGC_2516', cleaning=c)
    df[f'in_{c}'] = df.source_id.isin(df_A.source_id)

assert len(df) == N_members

dropcols = ['level_0', 'source_id_2', 'index', 'datalink_url',
            'epoch_photometry_url',  'priam_flags', 'teff_val',
            'teff_percentile_lower', 'teff_percentile_upper', 'a_g_val',
            'a_g_percentile_lower', 'a_g_percentile_upper', 'e_bp_min_rp_val',
            'e_bp_min_rp_percentile_lower', 'e_bp_min_rp_percentile_upper',
            'flame_flags', 'radius_val', 'radius_percentile_lower',
            'radius_percentile_upper', 'lum_val',
            'lum_percentile_lower', 'lum_percentile_upper',
            'ra_error', 'dec_error', 'parallax_over_error',
            'pmra_error', 'pmdec_error', 'ra_dec_corr',
            'ra_parallax_corr', 'ra_pmra_corr', 'ra_pmdec_corr',
            'dec_parallax_corr', 'dec_pmra_corr', 'dec_pmdec_corr',
            'parallax_pmra_corr', 'parallax_pmdec_corr',
            'pmra_pmdec_corr', 'astrometric_n_obs_al',
            'astrometric_n_obs_ac', 'astrometric_n_good_obs_al',
            'astrometric_n_bad_obs_al', 'astrometric_gof_al',
            'astrometric_chi2_al', 'astrometric_excess_noise',
            'astrometric_excess_noise_sig',
            'astrometric_params_solved', 'astrometric_primary_flag',
            'astrometric_weight_al', 'astrometric_pseudo_colour',
            'astrometric_pseudo_colour_error', 'mean_varpi_factor_al',
            'astrometric_matched_observations',
            'visibility_periods_used', 'astrometric_sigma5d_max',
            'frame_rotator_object_type', 'matched_observations',
            'duplicated_source', 'phot_g_n_obs', 'phot_g_mean_flux',
            'phot_g_mean_flux_error', 'phot_g_mean_flux_over_error',
            'phot_bp_n_obs', 'phot_bp_mean_flux',
            'phot_bp_mean_flux_error', 'phot_bp_mean_flux_over_error',
            'phot_rp_n_obs', 'phot_rp_mean_flux',
            'phot_rp_mean_flux_error', 'phot_rp_mean_flux_over_error',
            'phot_bp_rp_excess_factor', 'phot_proc_mode', 'bp_rp',
            'bp_g', 'g_rp',  'rv_nb_transits', 'rv_template_teff', 'rv_template_logg',
            'rv_template_fe_h', 'phot_variable_flag', 'solution_id',
            'designation', 'random_index', 'l', 'b', 'ecl_lon', 'ecl_lat']

df = df.drop(dropcols, axis=1)

# photometric binarity column
df = append_phot_binary_column(df)

# astrometric binarity column
basedata = 'fullfaint_edr3'
_, _, _, full_df, _ = get_gaia_basedata(basedata)

# merge, noting that the "ngc2516_rotation_periods.csv" measurements were
# done based on the DR2 source_id list, and in this plot the basedata are
# from EDR3 (so we use the DR2<->EDR3 crossmatch from
# _get_fullfaint_edr3_dataframes)
mdf = df.merge(full_df, left_on='source_id',
               right_on='dr2_source_id', how='left',
               suffixes=('_dr2', '_edr3'))

assert len(mdf) == len(df)

is_astrometric_binary = (mdf.ruwe > 1.2)
df['is_astrometric_binary'] = is_astrometric_binary


df = df.drop(['in_match234_alias'], axis=1)

df = df.rename(columns={
    'in_defaultcleaning': 'in_SetA',
    'in_periodogram_match': 'in_SetB',
    'is_phot_binary': 'is_phot_bin',
    'is_astrometric_binary': 'is_astrm_bin'
})

df = df.sort_values(by=['in_SetB'], ascending=False)

# by default, release all columns. it's machine-readable...
outpath = os.path.join(RESULTSDIR, 'tables', 'NGC_2516_Prot_cleaned.csv')

df.to_csv(outpath, index=False)
print(f'Wrote {outpath}')

namedesc_dict = {
'source_id': "Gaia DR2 source identifier.",
'n_cdips_sector': "Number of TESS sectors with CDIPS light curves.",
'period': "Lomb-Scargle best period.",
'lspval': "Lomb-Scargle periodogram value for best period.",
'nequal': "Number of stars brighter than the target in TESS aperture.",
'nclose': "Number of stars with $\Delta T > 1.25$ in TESS aperture.",
'nfaint': "Number of stars with $\Delta T > 2.5$ in TESS aperture.",
'spdmperiod': "Stellingwerf PDM best period.",
'spdmval': "Stellingwerf PDM periodogram value for best period.",
'ref_epoch': "Reference epoch for right ascension and declination.",
'ra': "Gaia DR2 right ascension.",
'dec': "Gaia DR2 declination.",
'parallax': "Gaia DR2 parallax.",
'parallax_error': "Gaia DR2 parallax uncertainty.",
'pmra': r"Gaia DR2 proper motion $\mu_\alpha \cos \delta$.",
'pmdec': "Gaia DR2 proper motion $\mu_\delta$.",
'phot_g_mean_mag': "Gaia DR2 $G$ magnitude.",
'phot_bp_mean_mag': "Gaia DR2 $G_\mathrm{BP}$ magnitude.",
'phot_rp_mean_mag': "Gaia DR2 $G_\mathrm{RP}$ magnitude.",
'radial_velocity': "Gaia DR2 heliocentric radial velocity.",
'radial_velocity_error': "Gaia DR2 radial velocity uncertainty.",
'subcluster': "Is star in core (CG18) or halo (KC19+M21)?",
'in_CG18': "Star in \\citet{cantatgaudin_gaia_2018}.",
'in_KC19': "Star in \\citet{kounkel_untangling_2019}.",
'in_M21': "Star in \\citet{meingast_2021}.",
'(Bp-Rp)_0': "Gaia Bp-Rp color, minus $E$($G_\mathrm{BP}$-$G_\mathrm{RP}$)=0.1343",
'in_SetA': "In Set $\mathcal{A}$ (LSP>0.08, P<15d, nequal==0, nclose>=1).",
'in_SetB': "In Set $\mathcal{B}$ (Set $\mathcal{A}$ and periods match).",
'is_phot_bin': "True if $>0.3$ mag above cluster isochrone.",
'is_astrm_bin': "True if Gaia EDR3 RUWE > 1.2."
}

keys = list(namedesc_dict.keys())
keys = ["\\texttt{"+k.replace("_", "\_")+"}" for k in keys]
vals = df.head(n=1).T.values.flatten()
descrs = list(namedesc_dict.values())

df_totex = pd.DataFrame({
    'Parameter': keys,
    'Example Value': vals,
    'Description': descrs
})

outpath = os.path.join(RESULTSDIR, 'tables', 'NGC_2516_Prot_cleaned_header.tex')

pd.set_option('display.max_colwidth',100)
# escape=False fixes "textbackslash"
df_totex.to_latex(outpath, index=False, escape=False)
print(f'Wrote {outpath}')



if include_lithium:
    #
    # TODO TODO: THE RANDICH+18 GAIA-ESO MERGE HAS NON-UNIQUE SOURCE_IDS. 
    # A FIXME IS TO DEBUG THIS
    #

    # get lithium data, first gaia-eso / R+18. rename columns to ensure that
    # sources with both GaiaESO and GALAH spectra get their EWs reported.
    ldf0 = _get_lithium_EW_df(1, 0)
    newcoldict = {}
    for c in ldf0.columns:
        if "Li" in c:
            newcoldict[c] =c+"_GaiaESO"
    ldf = ldf0.rename(newcoldict, axis='columns')
    mdf = df.merge(ldf0, how='left', on='source_id')

    ldf1 = _get_lithium_EW_df(0, 1)
    newcoldict = {}
    for c in ldf1.columns:
        if "Li" in c:
            newcoldict[c] =c+"_GALAH"
    outdf = mdf.merge(ldf1, how='left', on='source_id')

    assert len(outdf) == N_members

    # by default, release all columns. it's machine-readable...
    outpath = os.path.join(RESULTSDIR, 'tables', 'NGC_2516_Prot_cleaned.csv')

    outdf.to_csv(outpath, index=False)
    print(f'Wrote {outpath}')
