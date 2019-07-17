from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time

# waiting imports 
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

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

    for p in text:
        paragraph = p.text
        print(paragraph)



hrefs = download_reports(driver, 'ABP')

for href in hrefs:
    download_report(driver, href)

