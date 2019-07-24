#%% [markdown]
# # Scrape Morningstar Report Data

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

    # import credentials from credentials.py under gitignore
    session.post('https://www.morningstar.com.au/Security/Login', data = dict(
        UserName = login['username'],
        Password = login['password']
    ))

    return session

#%% iterate through all listed ASX
# session = create_session()
# reports = {}

# for i, row in asx_df.iterrows():
#     code = row['ASX code']
#     print('trying ' + code)
#     # max sure we dont get redirected to
#     try:
#         page = session.get('https://www.morningstar.com.au/Stocks/Archive/' + code)
#         assert(page.status_code == 200)
#     except:
#         # reset the session
#         session = create_session()
#     finally:
#         # try again
#         try:
#             page = session.get('https://www.morningstar.com.au/Stocks/Archive/' + code)
#             assert(page.status_code == 200)
#         except:
#             print('failed to reopen the session with return code ' + page.status_code)
#             raise

#     soup = bs(page.text, "html.parser")

#     # check if we have the reports table with links in it
#     table = soup.find('table', class_ = 'table1 rarchivetable')
#     if table == None:
#         print("no table found")
#         reports[code] = 'No data'
#     else:
#         links = table.find_all('a', class_ = 'plainlink')
#         if not links:
#             print('no archived reports')
#             reports[code] = 'No reports'
#         # save all the links in the dictionary
#         else:
#             hrefs = list()
#             for link in links:
#                 href = link['href']
#                 hrefs.append(href)
#             reports[code] = hrefs

#%% [markdown]
# Code blocks to save and read the dictionay to and from json
#%% 
# with open('reports.json', 'w') as outfile:  
#     json.dump(reports, outfile, indent=4)

#%%
with open('reports.json') as json_file:  
    archived_reports = json.load(json_file)        

#%% [markdown]
## Report Components
# We now need to extract our relevant coponents of the report page


#%% try getting the web page helper function
def get_page(session, url):
    try:
        page = session.get(url)
        assert(page.status_code == 200)
    except:
        # reset the session
        session = create_session()
    finally:
        # try again
        try:
            page = session.get(url)
            assert(page.status_code == 200)
        except:
            # print('failed to reopen the session with return code ' + page.status_code)
            raise
        return page

#%% [markdown]
# Define the logic to pull out the Analyst Note components

#%%
def get_analyst_note(div):
    '''
    Given the analyst note div return the relevant components in a dictionary

    Structure of the returned dictionary
    {
        "title" : <title>,
        "author" : <author>,
        "notes" : [<note>, <note>, ...]
    }
    '''
    # define the dictionary
    analyst_note = {}
    # pull out the title
    title = div.find('span', class_ = 'stockreportsubheader bold borderbtmD4').get_text()
    analyst_note['title'] = title

    # pull out the author - TO DO: no unique id or class on span element

    # pull out the comment blocks
    paragraphs = div.find_all('p', class_ = 'commenttext')
    notes = list()
    for paragraph in paragraphs:
        notes.append(paragraph.get_text())
    analyst_note['notes'] = notes

    return analyst_note

#%% [markdown]
# The analyst report

#%%
def get_analyst_report(div):
    '''
    Given the analyst report div return the relevant components in a dictionary

    Structure of the returned dictionary
    {
        <subsection> : [<note>, <note>, ...],
        <subsection> : [<note>, <note>, ...],
        ...
    }
    '''
    # define the dictionary
    analyst_report = {}

    # TO DO

    return analyst_report

#%% [markdown]
# The analyst valuation

#%%
def get_analyst_valuation(div):
    '''
    Given the analyst valuation div return the relevant components in a dictionary

    Structure of the returned dictionary
    {
        <subsection> : [<note>, <note>, ...],
        <subsection> : [<note>, <note>, ...],
        ...
    }
    '''
    # define the dictionary
    analyst_valuation = {}

    # TO DO

    return analyst_valuation


#%% [markdown]
# Now we want to get all the scrpaed componets into one dictionary

#%%
def get_research_report(page):
    '''
    Given a page object construct the research report in a dictionary format

    Structure of the returned dictionary
    {
        "date" : <research date>,
        "note" : <note dictionary>,
        "report" : <report dictionary,
        "valuation : <valuation dictionary,
        (TO DO - expand)
    }
    '''
    research = {}
    soup = bs(page.text, "html.parser")

    print(page.url)
    date = re.search('[0-9]{8}', page.url).group(0)
    research['date'] = date
                
    note_div = soup.find('div', id = 'AnalystNote')
    note = get_analyst_note(note_div)
    research['note'] = note

    report_div = soup.find('div', id = '')
    report = get_analyst_report(report_div)
    research['report'] = report

    valuation_div = soup.find('div', id = '')
    valuation = get_analyst_report(valuation_div)
    research['valuation'] = valuation

    return research

#%% Now scrape the reports
session = create_session()
i = 0
data = {}
for stock in archived_reports:
    if archived_reports[stock] == 'No data' or archived_reports[stock] == 'No reports':
        pass
    else:
        links = archived_reports[stock]
        # visit every archived report
        reports = []
        for link in links:
            page = get_page(session, 'https://www.morningstar.com.au' + link)
            research = get_research_report(page)

            reports.append(research)

        data[stock] = reports 
            
#%%
with open('data.json', 'w') as outfile:  
    json.dump(data, outfile, indent=4)

#%%
