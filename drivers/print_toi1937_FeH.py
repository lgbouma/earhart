import pandas as pd

csvpath = '../data/tic268_specmatch_params_20201113.csv'
df = pd.read_csv(csvpath)

print(42*'=')
print('SPECMATCH RESULTS FROM PFS SPECTRUM')
print(df.describe())

csvpath = '../data/ngc2516_magrini_2017_metallicities.csv'
df = pd.read_csv(csvpath)

print(42*'=')
print('MAGRINI+2017 METALLICITY MEASUREMENTS')
print(df.describe())
