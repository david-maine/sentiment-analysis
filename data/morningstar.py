import requests
from credentials import login
from bs4 import BeautifulSoup as bs


def create_session():
    '''
        Create a new session on the morningstar premium site
    '''
    session = requests.Session()

    # import credentials from credentials.py under gitignore
    session.post('https://www.morningstar.com.au/Security/Login', data = dict(
        UserName = login['morningstar']['username'],
        Password = login['morningstar']['password']
    ))

    return session

def get_page(session, url):
    '''
        build in logic to test if the page is redirected, and reopen if the session if so
    '''
    # max sure we dont get redirected to
    try:
        page = session.get(url)
        assert(page.status_code == 200)
    # otherwise reset the session
    except:
        # reset the session and try again
        session = create_session()
        try:
            page = session.get(url)
            assert(page.status_code == 200)
        except:
            # tidy this up
            print('failed to reopen the session with return code ' + page.status_code)
            raise
    # return the page
    finally:
        return page

def find_archives(session, code):
    '''
    return the list of archived report URLs provided the page
    '''
    try:
        page = get_page(session, 'https://www.morningstar.com.au/Stocks/Archive/' + code)
    except:
        raise

    soup = bs(page.text, "html.parser")

    # check if we have the reports table with links in it
    table = soup.find('table', class_ = 'table1 rarchivetable')
    if table == None:
        print("no table found")
        hrefs = []
    else:
        links = table.find_all('a', class_ = 'plainlink')
        if not links:
            print('no archived reports')
            hrefs = []
        # save all the links in the dictionary
        else:
            hrefs = list()
            for link in links:
                href = link['href']
                hrefs.append(href)
    return hrefs

def get_analyst_note(page_text):
    '''
    Given the analyst note div return the relevant components in a dictionary

    Structure of the returned dictionary
    {
        "title" : <title>,
        "author" : <author>,
        "notes" : [<note>, <note>, ...]
    }
    '''
    soup = bs(page_text, "html.parser")
    
    div = soup.find('div', id = 'AnalystNote')

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
        # remove whitespace 
        split = paragraph.get_text().split()
        string = ' '.join(split)
        # append if string not empty
        if string:
            notes.append(string)
    
    analyst_note['notes'] = notes

    return analyst_note


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
