import os, shutil, socket
from glob import glob
import pandas as pd, numpy as np
from functools import reduce
from cdips.utils.pipelineutils import load_status
from earhart.plotting import _get_nbhd_dataframes

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
    runid='ScoOB2'
):
    """
    valid runid: IC_2602, CrA, kc19group_113, Orion
    """

    # the allvariability logs, including the top 5 lomb-scargle periods, and
    # peak values, are here.
    logdir = f'/Users/luke/Dropbox/proj/cdips/results/allvariability_reports/{runid}/logs'
    logfiles = glob(os.path.join(logdir, '*status.log'))
    print(f'Got {len(logfiles)} log files.')

    source_ids = np.array(
        [np.int64(os.path.basename(f).split('_')[0]) for f in logfiles]
    )

    # retrieve the LS periods. only top period; since we're not bothering with
    # the "second period" classification option.
    periods, lspvals, nequal, nclose, nfaint = [], [], [], [], []
    for source_id, logpath in zip(source_ids, logfiles):
        s = load_status(logpath)
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

    print(f'Got {len(period_df[~pd.isnull(period_df.period)])} periods')

    # get the runid's source list
    sourcelistpath = os.path.join(
        '/Users/luke/Dropbox/proj/cdips/data/cluster_data/cdips_catalog_split',
        f'OC_MG_FINAL_v0.4_publishable_CUT_{runid}.csv'
    )
    df = pd.read_csv(
        sourcelistpath
    )

    mdf = period_df.merge(
        df, how='left', on='source_id'
    )

    # create "subcluster" column
    core_sel = (
        mdf.reference.str.contains('CantatGaudin_2018')
    )
    halo_sel = ~core_sel

    corehalo_vec = np.ones(len(mdf)).astype(str)
    corehalo_vec[core_sel] = 'core'
    corehalo_vec[halo_sel] = 'halo'

    mdf['subcluster'] = corehalo_vec

    outpath = f'../../data/rotation/{runid}_rotation_periods.csv'
    mdf.to_csv(
        outpath, index=False
    )
    print(f'Made {outpath}')


if __name__ == "__main__":
    get_auto_rotation_periods()
