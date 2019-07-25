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

