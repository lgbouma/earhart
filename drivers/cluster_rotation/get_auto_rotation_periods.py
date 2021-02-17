import os, shutil, socket, pickle
from glob import glob
import pandas as pd, numpy as np
from functools import reduce
from cdips.utils.pipelineutils import load_status

def ls_to_df(classfile):

    indf = pd.read_csv(classfile, names=['lsname'])

    pdfpaths = np.array(indf['lsname'])
    # e.g.,
    # vet_hlsp_cdips_tess_ffi_gaiatwo0002890062263458259968-0006_tess_v01_llc[gold PC].pdf

    classes = [p.split('[')[1].replace('].pdf','') for p in pdfpaths]

    pdfnames = [p.split('[')[0]+'.pdf' for p in pdfpaths]

    source_ids = [np.int64(p.split('_')[0]) for p in pdfpaths]

    df = pd.DataFrame({'Name':pdfnames, 'Tags':classes, 'source_id': source_ids})

    return df

def get_auto_rotation_periods(
    runid='NGC_2516',
    get_spdm=True
):
    """
    valid runids include:
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
    periods, lspvals, nequal, nclose, nfaint = [], [], [], [], []
    if get_spdm:
        spdmperiods, spdmvals = [], []

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
        'period': periods,
        'lspval': lspvals,
        'nequal': nequal,
        'nclose': nclose,
        'nfaint': nfaint
    })
    if get_spdm:
        period_df['spdmperiod'] = spdmperiods
        period_df['spdmval'] = spdmvals

    print(f'Got {len(period_df[~pd.isnull(period_df.period)])} periods')

    # get the runid's source list
    if 'compstar' not in runid:
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

    if 'compstar' not in runid:
        # create "subcluster" column
        core_sel = (
            mdf.reference.str.contains('CantatGaudin_2018')
        )
        halo_sel = ~core_sel

        corehalo_vec = np.ones(len(mdf)).astype(str)
        corehalo_vec[core_sel] = 'core'
        corehalo_vec[halo_sel] = 'halo'

        mdf['subcluster'] = corehalo_vec

    else:
        mdf['subcluster'] = 'nbhd'

    outpath = f'../../data/rotation/{runid}_rotation_periods.csv'
    mdf.to_csv(
        outpath, index=False
    )
    print(f'Made {outpath}')


if __name__ == "__main__":
    get_auto_rotation_periods()
