#%%
import requests
from bs4 import BeautifulSoup as bs
import json
import pandas as pd
import re


#%% get the codes we want to search
asx200_df = pd.read_csv(
    "~/projects/personal/data/stock/20190701-asx200.csv",
    skiprows=1
    )

#%%
session = requests.Session()

session.post('https://www.morningstar.com.au/Security/Login', data = dict(
    UserName = "",
    Password = ""
))

#%%
page = session.get('https://www.morningstar.com.au/Stocks/Archive/')
soup = bs(page.text, "html.parser")
#%%



driver = webdriver.Chrome()

driver.get("https://www.morningstar.com.au/Security/Login")

# login

driver.find_element_by_id("loginFormNew").submit()

def download_reports(driver, code):
    driver.get('https://www.morningstar.com.au/Stocks/Archive/' + code)

    # get list of all hyperlinks
    links = driver.find_elements_by_xpath(
        "//table[@class = 'table1 rarchivetable']//a[@class = 'plainlink']"
    )

    hrefs = list()
    # make a list of links:
    for link in links:
        href = link.get_attribute("href")
        hrefs.append(href)
    
    return(hrefs)

def download_report(driver, href):
    driver.get(href)

    text = driver.find_elements_by_xpath(
        "//div[@id = 'AnalystNote']//p[@class = 'commenttext']"
    )

    paragraphs = list()
    for p in text:
        paragraphs.append(p.text)
    
    return paragraphs



# downlaod all the reports
data = {}
data["companies"] = []
for i, row in asx200_df.iterrows():
    code = row['Code']

    hrefs = download_reports(driver, code)
    
    company = {}
    company[code] = {}

    reviews = []
    for j, href in enumerate(hrefs):
        # get the date from the href
        date = re.search('[0-9]{8}', href).group(0)
        review = {}
        review[date] = []
        paragraphs = download_report(driver, href)
        for p in paragraphs:
            review[date].append(p)
        reviews.append(review)

        # if j > 1:
        #     break

    if i > 10:
        break

    company[code]['reviews'] = reviews

    data["companies"].append(company) 

with open('data.json', 'w') as outfile:  
    json.dump(data, outfile, indent=4)


