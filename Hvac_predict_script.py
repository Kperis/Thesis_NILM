from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import pandas as pd
import numpy as np  

test_start = pd.Timestamp('2023-01-22')
test_end = pd.Timestamp('2023-02-27')

df = dfs[0].copy()
# df = df.dropna()

unique_days = df.index.normalize().unique()
second_day = unique_days[4]


start = pd.to_datetime('2023-02-02 00:00:00')
end   = pd.to_datetime('2023-02-02 23:59:59')
df_day = df[(df.index >= start) & (df.index <= end)]

print(df_day.shape[0])
lags_out = [1, 2, 3, 4, 96]
for lag in lags_out:
    df[f'Tout_lag{lag}'] = df['Tout'].shift(lag)
# df['Tin_lag4'] = df['Tin'].shift(4)
# df['Tout_lag1'] = df['Tout'].shift(1)
# df['Tout_lag2'] = df['Tout'].shift(2)
df['Tin_lag4'] = df['Tin'].shift(4)
df['Tin_lag1'] = df['Tin'].shift(1)
df['Tin_lag2'] = df['Tin'].shift(2)
df['Tin_lag3'] = df['Tin'].shift(3)
future_steps = 4
for i in range(1, future_steps + 1):
    df[f'Tout_fut{i}'] = df['Tout'].shift(-i)

df['Tin_target']  = df['Tin'].shift(-1).dropna()          
df['is_test'] = (df.index >= test_start) & (df.index <= test_end)
df['mode_heatcool'] = np.where(df['Tout'] > 15, 0, 1)

df['hvac_mode'] = 0 
on_mask = df['hvac_energy'] > 0
nonzero_power = df.loc[on_mask, 'hvac_energy']
bins = pd.qcut(nonzero_power, q=2, labels=[1, 2])
df.loc[on_mask, 'hvac_mode'] = bins.astype(int).values
df['hvac_mode_lag'] = df['hvac_mode'].shift(1)

df_on = df[df['hvac_energy'] > 0].copy()
nonzero_power = df_on['hvac_energy']
bins = pd.qcut(nonzero_power, q=2, labels=[1, 2])
df.loc[df['hvac_energy'] > 0, 'hvac_mode'] = bins.astype(int)
df_off = df[df.hvac_mode == 0].copy()