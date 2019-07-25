#%%
import pandas as pd
import os

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

# from pandas.api.types import CategoricalDtype
# from plotnine import *
# from plotnine.data import mpg
%matplotlib inline

#%% [markdown]
# Inspect the data

#%%
data = pd.read_csv(
    '../data/stock/price_history/1997-2006/19970102.TXT', 
    names=['Stock', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    )

#%% [markdown]
# We have end of day stock data in the form
#
# Stock | Date | Open | High | Low | Close | Volume
# --- | --- | --- | --- | --- | --- | ---  

#%% 
data.head()


#%% [markdown]
# need to load this data historically
#%%
dir = '../data/stock/price_history/'

# data = pd.DataFrame(columns=[0, 1, 2, 3, 4, 5, 6], dtype='int64')
data = pd.DataFrame()

for r, d, f in os.walk(dir):
    for file in f:
        temp = pd.read_csv(
        os.path.join(r, file), 
        names=['stock', 'date', 'open', 'high', 'low', 'close', 'volume']
        )
        data = data.append(temp)

    # break
data['datetime'] = pd.to_datetime(data['date'].astype(str), format='%Y%m%d')

# data.append(temp)

#%% [markdown]

# plot and inspect the data
sns.set_style("whitegrid")
sns.relplot(
    data = data[data.stock == 'BOQ'], 
    x = "datetime", 
    y = "close",
    kind = "line",
    style = "stock"
)




#%%

#%%
