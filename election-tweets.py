#%% imports
import numpy as np
import pandas as pd
import torch
import os

%matplotlib inline
# import plotnine as p9


import matplotlib.pyplot as plt

#%% data import
tweet_df = pd.read_csv("~/projects/personal/data/election-tweets/auspol2019.csv")
location_df = pd.read_csv("~/projects/personal/data/election-tweets/location_geocode.csv")


#%% explore data
tweet_df.head()
location_df.head()


#%%
tweet_df.hist()