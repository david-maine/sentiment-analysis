#%%
import pandas as pd
import os

from pymongo import MongoClient

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%
client = MongoClient()
db = client.stock

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

#%% write the data to mongo
for i, row in data.iterrows():
    price  = {
        'stock' : row['stock'],
        'date' : row['datetime'],
        'open' : row['open'],
        'high' : row['high'],
        'low' : row['low'],
        'close' : row['close'],
        'volume' : row['volume'],
    }

    prices = db.prices
    prices.insert_one(price)
    # break

#%% [markdown]

#%% plot and inspect the data
%matplotlib inline

# query the data
query = db.prices.find({"stock": { "$in": [ "ANZ", "BOQ", "CSL" ] }})
data = pd.DataFrame(list(query))

#%%
sns.set_style("darkgrid")
sns.relplot(
    data = data, 
    x = "date", 
    y = "close",
    kind = "line",
    hue = "stock"
)


#%% [markdown]

# going to want to squeeze increases/decreases into [0, 1] to code our NLP 

#%%
