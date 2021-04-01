import numpy as np

from gilly.gyrochronology import (
  MamajekHillenbrand08_gyro,
  SpadaLanzafame20_gyro,
  Angus19_gyro
)

from cdips.utils.mamajek import get_interp_BmV_from_BpmRp

BpmRp = np.array([1.35])
Prots = np.array([7,8,9,10,11]) # 10 days is max NGC2516 at this color

for Prot in Prots:
    Prot = np.array([Prot])

    # BpmRp is the existing array of dereddened Gaia colors
    BmV = get_interp_BmV_from_BpmRp(BpmRp)

    # Prot is the array of periods
    mh08_age = MamajekHillenbrand08_gyro(BmV, Prot)
    a19_age = Angus19_gyro(BpmRp, Prot)

    # can set plot as true to write a sanity check
    sl20_age = SpadaLanzafame20_gyro(BmV=BmV, Prot=Prot, makeplot=False)

    print(42*'.')
    print(f'Bp-Rp={BpmRp}, Prot={Prot} d, interp B-V={BmV}')
    print(f'MamajekHillenbrand08: {mh08_age}')
    print(f'Angus+19: {a19_age}')
    print(f'SpadaLanzafame20: {sl20_age}')
    print(42*'.')
