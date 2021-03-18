import os, shutil, socket, pickle
from glob import glob
import pandas as pd, numpy as np
from functools import reduce
from cdips.utils.pipelineutils import load_status

def get_auto_rotation_periods(
    runid='NGC_2516',
    get_spdm=True
):
    """
    Given a `runid` (an identifier string for a particular CDIPS
    "allvariability" sub-pipeline processing run), retrieve the following
    output and concatenate into a table, which is then saved to
    '../../data/rotation/{runid}_rotation_periods.csv':

        [
        'source_id': source_ids,
        'n_cdips_sector': nsectors,
        'period': periods,
        'lspval': lspvals,
        'nequal': nequal,
        'nclose': nclose,
        'nfaint': nfaint
        ]

    Valid runids include:
        IC_2602, CrA, kc19group_113, Orion, NGC_2516, ScoOB2, compstar_NGC_2516
    """

    # the allvariability logs, including the top 5 lomb-scargle periods, and
    # peak values, are here.
    logdir = f'/Users/luke/Dropbox/proj/cdips/results/allvariability_reports/{runid}/logs'
    logfiles = glob(os.path.join(logdir, '*status.log'))
    print(f'Got {len(logfiles)} log files.')

    if get_spdm:
        pkldir = f'/Users/luke/Dropbox/proj/cdips/results/allvariability_reports/{runid}/data'
        pklfiles = glob(os.path.join(pkldir, '*reportinfo.pkl'))
        N_pklfiles = len(pklfiles)
        print(f'Got {N_pklfiles} pickle files.')
        if N_pklfiles < 10:
            raise ValueError('Too few pickle files... Port from phtess2?')

    source_ids = np.array(
        [np.int64(os.path.basename(f).split('_')[0]) for f in logfiles]
    )

    # retrieve the LS periods. only top period; since we're not bothering with
    # the "second period" classification option.
    periods, lspvals, nequal, nclose, nfaint, nsectors = [], [], [], [], [], []
    if get_spdm:
        spdmperiods, spdmvals = [], []

    ix = 0
    for source_id, logpath in zip(source_ids, logfiles):

        s = load_status(logpath)

        if get_spdm:
            pklpath = os.path.join(pkldir, f'{source_id}_reportinfo.pkl')
            if not os.path.exists(pklpath):
                spdmperiods.append(np.nan)
                spdmvals.append(np.nan)
            else:
                with open(pklpath, 'rb') as f:
                    d = pickle.load(f)
                spdmperiods.append(float(d['spdm']['bestperiod']))
                spdmvals.append(float(d['spdm']['bestlspval']))

        n_sectors = int(s['lc_info']['n_sectors'])
        nsectors.append(n_sectors)
        try:
            periods.append(float(s['report_info']['ls_period']))
            lspvals.append(float(s['report_info']['bestlspval']))
            nequal.append(int(eval(s['report_info']['n_dict'])['equal']))
            nclose.append(int(eval(s['report_info']['n_dict'])['close']))
            nfaint.append(int(eval(s['report_info']['n_dict'])['faint']))
        except (TypeError, ValueError) as e:
            periods.append(np.nan)
            lspvals.append(np.nan)
            nequal.append(np.nan)
            nclose.append(np.nan)
            nfaint.append(np.nan)

    period_df = pd.DataFrame({
        'source_id': source_ids,
        'n_cdips_sector': nsectors,
        'period': periods,
        'lspval': lspvals,
        'nequal': nequal,
        'nclose': nclose,
        'nfaint': nfaint
    })
    if get_spdm:
        period_df['spdmperiod'] = spdmperiods
        period_df['spdmval'] = spdmvals

    print(f'Got {len(period_df[period_df.n_cdips_sector > 0])} sources with at least 1 cdips sector')
    print(f'Got {len(period_df[~pd.isnull(period_df.period)])} periods')

    # get the runid's source list
    if 'compstar' not in runid:
        if runid == 'NGC_2516':
            sourcelistpath = (
                '/Users/luke/Dropbox/proj/cdips/data/cluster_data/NGC_2516_full_fullfaint_20210305.csv'
            )
        else:
            sourcelistpath = os.path.join(
                '/Users/luke/Dropbox/proj/cdips/data/cluster_data/cdips_catalog_split',
                f'OC_MG_FINAL_v0.4_publishable_CUT_{runid}.csv'
            )
    else:
        sourcelistpath = (
            f'/Users/luke/Dropbox/proj/earhart/results/tables/{runid}_sourcelist.csv'
        )

    df = pd.read_csv(
        sourcelistpath
    )
    if 'compstar' in runid:

        print(42*'-')
        print(f'{len(df)} light curves made for stars in neighborhood (calib+cdips)')
        print(f'... for {len(np.unique(df.source_id))} unique stars')

        df = df[df.phot_rp_mean_mag<13]
        print(f'{len(df)} light curves made for stars in neighborhood (calib+cdips) w/ Rp<13')
        print(f'... for {len(np.unique(df.source_id))} unique stars')
        print(42*'-')

        df = df.drop_duplicates(subset='source_id', keep='first')

    mdf = period_df.merge(
        df, how='inner', on='source_id'
    )

    outpath = f'../../data/rotation/{runid}_rotation_periods.csv'
    mdf.to_csv(
        outpath, index=False
    )
    print(f'Made {outpath}')


if __name__ == "__main__":
    get_auto_rotation_periods()
