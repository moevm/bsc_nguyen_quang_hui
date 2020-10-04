import os
from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support.expected_conditions import url_to_be, element_to_be_clickable, invisibility_of_element
from fake_useragent import UserAgent
import time
import re
import json
import random
def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

ua = UserAgent()

options = Options()
options.headless = True
options.add_argument('--no-sandbox')
options.add_argument('--window-size=1920,1080')
options.add_argument('--allow-insecure-localhost')
options.add_experimental_option('excludeSwitches', ['enable-logging'])
driver = Chrome(options=options)
# driver.quit()
driver.get("https://cyberleninka.ru/article/c/computer-and-information-sciences")

last_page_id = driver.find_elements(By.CSS_SELECTOR, 'ul.paginator li')[
    -1].find_elements_by_tag_name("a")[0].get_attribute('href').split('/')[-1]
last_page_id = int(last_page_id)
# print(last_page_id)

cleanup_pattern = re.compile(r'\[.*?\]|[\+\*\/\\!@#$%^&*]+|lt;|gt;|±|-{2,}')

count = 0
with open("cyberleninka2.txt", "a+", encoding='utf-8') as output_file:
    output_file.seek(0)
    data = output_file.read(100)
    if len(data) == 0:
        output_file.write("[\n")

    for page_id in range(217, last_page_id+1):
        print("page start: "+str(page_id))

        driver.quit()
        options = Options()
        options.headless = True
        userAgent = ua.random
        options.add_argument(f'user-agent={userAgent}')
        options.add_argument('--no-sandbox')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--allow-insecure-localhost')
        options.add_experimental_option('excludeSwitches', ['enable-logging'])
        driver = Chrome(options=options)

        time.sleep(1)
        driver.get(
            "https://cyberleninka.ru/article/c/computer-and-information-sciences/"+str(page_id))
        article_url_list = [e.get_attribute('href') for e in driver.find_elements(
            By.CSS_SELECTOR, 'ul.list li a:first-child')]
        for article_url in article_url_list:
            if driver.find_element(By.CSS_SELECTOR, 'div.main h1:first-child').get_attribute('innerHTML')=="Вы точно человек?":
                time.sleep(2)
                WebDriverWait(driver, 10).until(EC.frame_to_be_available_and_switch_to_it((By.CSS_SELECTOR,"iframe[name^='a-'][src^='https://www.google.com/recaptcha/api2/anchor?']")))
                time.sleep(2)
                WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//span[@id='recaptcha-anchor']"))).click()
                time.sleep(2)
                driver.switch_to.default_content()
                WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "input[value='Продолжить']"))).click()

                
            time.sleep(random.randint(1,10))
            driver.get(article_url)
            keywords = driver.find_elements(
                By.CSS_SELECTOR, 'div.keywords i span')
            keywords = [kw_tag.get_attribute('innerHTML').lower()
                        for kw_tag in keywords]
            if(len(keywords) == 0):
                continue
            print(keywords)

            fulltext = driver.find_elements(
                By.CSS_SELECTOR, "div.ocr[itemprop='articleBody'] p")
            # cleanup text, remove number, special symbol (+-=[*])
            fulltext = [re.sub(cleanup_pattern, "", t.get_attribute(
                'innerHTML')) for t in fulltext]
            if(len(fulltext) == 0):
                continue

            # remove СПИСОК ЛИТЕРАТУРЫ
            for i in range(len(fulltext)-1, -1, -1):
                if(fulltext[i].lower().startswith('список литературы')
                   or fulltext[i].lower().startswith('библиографический список')
                   or fulltext[i].lower().startswith('литература')):
                    fulltext = fulltext[0:i]
                    break

            title = driver.find_elements(
                By.CSS_SELECTOR, "i[itemprop='headline']")
            title = [cleanhtml(t.get_attribute('innerHTML')) for t in title]
            if(len(title) != 0):
                title = title[0]
                # print(title)

            abstract = driver.find_elements(
                By.CSS_SELECTOR, "div.abstract p[itemprop='description']")
            abstract = [cleanhtml(t.get_attribute('innerHTML'))
                        for t in abstract]
            if(len(abstract) != 0):
                abstract = abstract[0]
                # print(abstract)

            views = driver.find_elements(
                By.CSS_SELECTOR, "div.views[title='Просмотры']")
            views = [cleanhtml(t.get_attribute('innerHTML')) for t in views]
            if(len(views) != 0):
                views = views[0]
                # print(views)

            downloads = driver.find_elements(
                By.CSS_SELECTOR, "div.downloads[title='Загрузки']")
            downloads = [cleanhtml(t.get_attribute('innerHTML'))
                         for t in downloads]
            if(len(downloads) != 0):
                downloads = downloads[0]
                # print(downloads)

            data_obj = {
                'keywords': keywords,
                'fulltext': fulltext,
                'title': title,
                'abstract': abstract,
                'views': views,
                'downloads': downloads
            }

            data_json = (json.dumps(data_obj, ensure_ascii=False)+',')
            output_file.write(data_json)
            count+=1
            print("article - "+str(count))
        print("page end: "+str(page_id))
    output_file.write(']')

# if(len(self.driver.find_element(By.TAG_NAME, 'body').get_attribute('innerHTML'))==0):
#     self.driver.get("http://web:5000/")

# self.driver.find_element(
#     By.CSS_SELECTOR, "#navbarNav ul li:first-child a").click()

# # print(self.driver.current_url)
# # self.driver.add_cookie({'name' : '_LOCALE_', 'value' : 'en'})
# self.driver.refresh()
# self.driver.find_element(
#     By.ID, "template-file").send_keys(os.path.join(self.template_folder,self.template))
# self.driver.find_element(By.ID, "btn-upload-template").click()

# # wait until upload complete (use this template button available to press)
# WebDriverWait(self.driver, timeout=10).until(
#     element_to_be_clickable((By.ID, "btn-use-template")))
# self.driver.find_element(By.ID, "btn-use-template").click()

# WebDriverWait(self.driver, timeout=10).until(
#     element_to_be_clickable((By.ID, "btn-upload-data")))
# self.driver.find_element(
#     By.ID, "data-table-file").send_keys(os.path.join(self.data_folder,self.table))
# self.driver.find_element(By.ID, "btn-upload-data").click()

# # wait until upload complete (use this table button available to press)
# WebDriverWait(self.driver, timeout=10).until(
#     element_to_be_clickable((By.ID, "btn-verify")))
# self.driver.find_element(By.ID, "btn-verify").click()

# WebDriverWait(self.driver, timeout=20).until(
#     element_to_be_clickable((By.ID, "btn-generate")))
# # check for verification result
# verification_result = self.driver.find_element(By.ID, 'verificationResult').get_attribute('innerHTML')
# self.assertEqual(verification_result, self.verification_mes)

# self.driver.find_element(By.ID, "btn-generate").click()
# WebDriverWait(self.driver, timeout=20).until(element_to_be_clickable(
#     (By.CSS_SELECTOR, "#pills-result .row .col-md .text-center input")))

# no_files = len(self.driver.find_element(By.CSS_SELECTOR, '#link-files ul').find_elements_by_tag_name('li'))
# self.assertEqual(no_files, self.no_output_files)

# self.driver.find_element(
#     By.CSS_SELECTOR, "#pills-result .row .col-md .text-center input").click()
