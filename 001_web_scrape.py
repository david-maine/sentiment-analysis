#%% [markdown]
# # Scrape Morningstar Report Data

#%%
from pymongo import MongoClient
import pandas as pd
from data import morningstar
from bs4 import BeautifulSoup as bs

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


#%% [markdown]
# Save all the pages for later decomposition

#%%
session = morningstar.create_session()
    
archives = db.archives.find()

for archive in archives:
    # print(archive)
    # page = morningstar.get_page(session, link)
    for url in archive['links']:
        print(url)
        page = morningstar.get_page(
            session, 
            "https://www.morningstar.com.au" + url
            )

        # soup = bs(page.text, "html.parser")

        post = {
            "url" : url,
            "html" : page.text
        }

        # print(post)
        pages = db.pages
        pages.insert_one(post)
    # break

client.close()    



#%%
