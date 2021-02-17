import pandas as pd, matplotlib.pyplot as plt, numpy as np
from earhart.helpers import get_autorotation_dataframe
from aesthetic.plot import savefig, format_ax, set_style

df = get_autorotation_dataframe(runid='NGC_2516', verbose=1, returnbase=0)

f, ax = plt.subplots(figsize=(4,4))
set_style()

ax.scatter(df.period, df.spdmperiod, c='k', s=1)

ax.plot(np.arange(0.1,30,0.1), np.arange(0.1,30,0.1), zorder=-1, lw=1, c='gray')
ax.plot(np.arange(0.1,30,0.1), 2*np.arange(0.1,30,0.1), zorder=-1, lw=1,
        c='gray', ls='--')
ax.plot(np.arange(0.1,30,0.1), 0.5*np.arange(0.1,30,0.1), zorder=-1, lw=1,
        c='gray', ls=':')
ax.plot(np.arange(0.1,30,0.1), 3*np.arange(0.1,30,0.1), zorder=-1, lw=1,
        c='gray', ls=':')

ax.set_xlabel('LS Period [days]')
ax.set_ylabel('SPDM Period [days]')

ax.set_xlim([0,30])
ax.set_ylim([0,30])

format_ax(ax)

outpath = '../results/rotation/NGC_2516/NGC_2516_LS_vs_SPDM_periods.png'
savefig(f, outpath, )

##########

sel_match = (
    (0.9 < (df.spdmperiod/df.period))
    &
    (1.1 > (df.spdmperiod/df.period))
)

sel_spdm2x = (
    (1.9 < (df.spdmperiod/df.period))
    &
    (2.1 > (df.spdmperiod/df.period))
)

sel_spdmpt5x = (
    (0.4 < (df.spdmperiod/df.period))
    &
    (0.6 > (df.spdmperiod/df.period))
)

sel_spdm3x = (
    (2.9 < (df.spdmperiod/df.period))
    &
    (3.1 > (df.spdmperiod/df.period))
)

sel_spdm4x = (
    (3.9 < (df.spdmperiod/df.period))
    &
    (4.1 > (df.spdmperiod/df.period))
)


print(42*'.')
print(f'{len(df)} entries met selection function...')
print(f'{len(df[sel_match])} have LS + SPDM periods agree within 10%')
print(f'{len(df[sel_spdm2x])} have SPDM period 2x that of LS (within 10%)')
print(f'{len(df[sel_spdm3x])} have SPDM period 3x that of LS (within 10%)')
print(f'{len(df[sel_spdm4x])} have SPDM period 4x that of LS (within 10%)')
print(f'{len(df[sel_spdmpt5x])} have SPDM period 0.5x that of LS (within 10%)')
print(42*'.')
