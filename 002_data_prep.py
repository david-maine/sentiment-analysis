#%% [markdown]
# # Scrape Morningstar Report Data

#%%
import requests
from bs4 import BeautifulSoup as bs
import json
import pandas as pd
import re
from pymongo import MongoClient
from data import morningstar

import datetime

#%%
client = MongoClient()
db = client.stock

#%% [markdown]
# Define the logic to pull out the Analyst Note components

# iterate through all the archives
pages = db.pages.find()

for page in pages:

    url = page['url']
    code = url[-3:]
    date_string = re.search('[0-9]{8}', url).group(0)
    date = datetime.datetime.strptime(date_string, '%Y%m%d')

    try:
        note = morningstar.get_analyst_note(page['html'])
    except:
        print("couldnt parse " + url)
        note = None
    article = {
        "code" : code,
        "date" : date,
        "note" : note
    }

    articles = db.morningstar.articles
    articles.insert_one(article)

