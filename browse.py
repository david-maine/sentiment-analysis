from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time

# waiting imports 
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import json
import pandas as pd


# get the codes we want to search
asx200_df = pd.read_csv(
    "~/projects/personal/data/stock/20190701-asx200.csv",
    skiprows=1
    )


driver = webdriver.Chrome()

driver.get("https://www.morningstar.com.au/Security/Login")

# login
driver.find_element_by_name("UserName").send_keys("daveanthony97@hotmail.com")
driver.find_element_by_name("Password").send_keys("Millie97")
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
        review = {}
        review[str(i)] = []
        paragraphs = download_report(driver, href)
        for p in paragraphs:
            review[str(i)].append(p)
        reviews.append(review)

        if j > 10:
            break

    if i > 3:
        break

    company[code]['reviews'] = reviews

    data["companies"].append(company) 

with open('data.txt', 'w') as outfile:  
    json.dump(data, outfile)


