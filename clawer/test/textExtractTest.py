'''
Created on 2019年7月1日

@author: gaojiexcq
'''
import requests
from readability import Document
from html2text import HTML2Text
import sqlite3
from selenium import webdriver
# from selenium.webdriver.common.keys import Keys
import re
import os.path as ospath
import time
from clawer.textextract import readAbilityExtract
def getUrls(dbstr,tablename):
    conn = sqlite3.connect(dbstr)
    c = conn.cursor()
    data = c.execute("select link from "+tablename)
    urls = []
    for link in data:
        urls.append(link[0])
    return urls

def getHtml(url):
    browser = webdriver.Chrome()
#     browser.set_page_load_timeout(100)
    browser.get(url)
    time.sleep(90)
    html = browser.page_source
#     html = browser.find_element_by_tag_name("html").get_attribute("innerHTML")
#     print(html)
#     browser.quit()
    return html

def getTitle_Content(html):
    doc = Document(html)
    title = doc.title()
    h = HTML2Text()
#     print(doc.content())
    h.ignore_images = True
    h.ignore_links = True
#     print(doc.content().replace("\\n", "\n"))
    content = h.handle(doc.summary())
    return title,content

def validateTitle(title):
    rstr = r"[\s\/\\\:\*\?\"\<\>\|]"  # '/ \ : * ? " < > |'
    new_title = re.sub(rstr, "_", title)
    if len(new_title) > 255:
        new_title = new_title[0:255]
    return new_title

def saveContent(dirpath,id,titleForFile,title,content):
    path = ospath.join(dirpath,titleForFile+"-id-"+str(id)+".txt")
    with open(path,mode="w",encoding="utf-8") as f:
        f.write(title+"\n\n"+content+"\n")

dbstr = "../textextract/data/bad_boy_links.db"
tablename="tb_links"
dirpath = "content_for_url/"
# urls = getUrls(dbstr, tablename)
# 
h2t = readAbilityExtract.ReadAbilityExtract()
print(h2t)
# links_info_json = h2t.fit_jsonlize(urls[0:2])
# print(links_info_json)




# # for url in urls:
# #     print(url)
# for i in range(len(urls)):
# #     url = "http://worldeconomy-wingate.blogspot.com/2010/05"
#     url = "https://blog.csdn.net/weixin_36279318/article/details/79475388"
#     html = getHtml(url)
#     with open("test.html",mode="w",encoding="utf-8") as f:
#         f.write(html)
# #     print(html)
#     title,content = getTitle_Content(html)
#     print(title)
#     print(content)
#     break
# #     titleForFile = validateTitle(title)
# #     saveContent(dirpath,id,titleForFile,title,content)  