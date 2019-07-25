#%% [markdown]
# # Scrape Morningstar Report Data

#%%
from pymongo import MongoClient
import pandas as pd
from data import morningstar

#%%
client = MongoClient()
db = client.stock

#%% get the codes we want to search
asx_df = pd.read_csv(
    "~/projects/personal/data/stock/ASXListedCompanies.csv",
    skiprows=2
    )

#%% iterate through all listed ASX
session = morningstar.create_session()

for i, row in asx_df.iterrows():
    code = row['ASX code']
    print('trying ' + code)

    # get the archive links from morningstar
    hrefs = morningstar.find_archives(session, code)

    # add this to our archives collection
    archive = {
        "code" : code,
        "links" : hrefs   
    }

    archives = db.archives
    archives.insert_one(archive)

client.close()    


#%%
