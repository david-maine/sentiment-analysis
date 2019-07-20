#%%
import requests
from bs4 import BeautifulSoup as bs
import json
import pandas as pd
import re

from credentials import login

#%% get the codes we want to search
asx_df = pd.read_csv(
    "~/projects/personal/data/stock/ASXListedCompanies.csv",
    skiprows=2
    )

#%% try research archives
# saved the logged in session
def create_session():
    session = requests.Session()

    session.post('https://www.morningstar.com.au/Security/Login', data = dict(
        UserName = login['username'],
        Password = login['password']
    ))

    return session

#%% iterate through all listed ASX
session = create_session()
reports = {}

for i, row in asx_df.iterrows():
    code = row['ASX code']
    print('trying ' + code)
    # max sure we dont get redirected to
    try:
        page = session.get('https://www.morningstar.com.au/Stocks/Archive/' + code)
        assert(page.status_code == 200)
    except:
        # reset the session
        session = create_session
    finally:
        # try again
        try:
            page = session.get('https://www.morningstar.com.au/Stocks/Archive/' + code)
            assert(page.status_code == 200)
        except:
            print('failed to reopen the session')
            break

    soup = bs(page.text, "html.parser")

    # check if we have the reports table with links in it
    table = soup.find('table', class_ = 'table1 rarchivetable')
    if table == None:
        print("no table found")
        reports[code] = 'No data'
    else:
        links = table.find_all('a', class_ = 'plainlink')
        if not links:
            print('no archived reports')
            reports[code] = 'No reports'
        # save all the links in the dictionary
        else:
            hrefs = list()
            for link in links:
                href = link['href']
                hrefs.append(href)
            reports[code] = hrefs

#%% save those report links
with open('reports.json', 'w') as outfile:  
    json.dump(reports, outfile, indent=4)
            

#%% Get the analyst reports from the links
for stock in reports:
    if reports[stock] == 'No data' or reports[stock] == 'no archived reports':
        pass
    else 

#%%
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
        if hrefs == None:
    
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
for i, row in asx_df.iterrows():
    code = row['ASX code']

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




#%%
