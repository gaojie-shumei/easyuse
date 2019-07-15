'''
Created on 2019年7月12日

@author: gaojiexcq
'''
import requests
from readability import Document
from html2text import HTML2Text
from selenium import webdriver
import re
import os.path as ospath
import time
import json

class ReadAbilityExtract(object):
    def __init__(self,file_encoding:str = None,second_delay:int=None):
        if file_encoding is None:
            self._file_encoding = "utf-8"
        else:
            self._file_encoding = file_encoding    
        if second_delay is None:
            self._second_delay = 90
        else:
            self._second_delay = second_delay
        self._links_info = []
            
    #根据link获取其浏览器渲染过后的html源码
    def get_html(self,source_link,browser):
        try:
            if browser is None:
                browser = webdriver.Chrome()
            browser.get(source_link)
            time.sleep(self._second_delay)
            html = browser.page_source
            print(html)
            browser.close()
        except:
            response = requests.get(source_link)
            html = response.text
            if html is None or html=="":
                print("get html error")
                html = ""
        return html
    
    #验证保存文件时的title
    def validate_title(self,title):
        rstr = r"[\s\/\\\:\*\?\"\<\>\|]"  # '/ \ : * ? " < > |'
        new_title = re.sub(rstr, "_", title)
        if len(new_title) > 255:
            new_title = new_title[0:255]
        return new_title
    
    def get_title(self,html):
        if html!="":
            doc = Document(html)
            title = doc.title()
        else:
            title = ""
        return title
    
    def get_clean_text(self,html):
        if html!="":
            doc = Document(html)
            h = HTML2Text()
            h.ignore_images = True
            h.ignore_links = True
            clean_text = h.handle(doc.summary())
        else:
            clean_text = ""
        return clean_text
    
    def fit(self,source_links:list,browser:[webdriver.Edge,webdriver.Chrome,
                                            webdriver.Firefox,webdriver.Opera,
                                            webdriver.PhantomJS,webdriver.Safari,
                                            webdriver.Android,webdriver.Ie]=None):
        for link in source_links:
            html = self.get_html(link,browser)
            title = self.get_title(html)
            file_title = self.validate_title(title)
            clean_text = self.get_clean_text(html)
            link_info = dict()
            link_info["html"] = html
            link_info["title"] = title
            link_info["file_title"] = file_title
            link_info["clean_text"] = clean_text
            link_info["source_link"] = link
            self._links_info.append(link_info)
        return self
    
    def fit_transform(self,source_links:list,browser:[webdriver.Edge,webdriver.Chrome,
                                                      webdriver.Firefox,webdriver.Opera,
                                                      webdriver.PhantomJS,webdriver.Safari,
                                                      webdriver.Android,webdriver.Ie]=None):
        self.fit(source_links,browser)
        return self._links_info
    
    def fit_jsonlize(self,source_links:list,browser:[webdriver.Edge,webdriver.Chrome,
                                                     webdriver.Firefox,webdriver.Opera,
                                                     webdriver.PhantomJS,webdriver.Safari,
                                                     webdriver.Android,webdriver.Ie]=None):
        self.fit(source_links,browser)
        return json.dumps(self._links_info)
    
    def fit2file(self,source_links,dirpath:list,browser:[webdriver.Edge,webdriver.Chrome,
                                                          webdriver.Firefox,webdriver.Opera,
                                                          webdriver.PhantomJS,webdriver.Safari,
                                                          webdriver.Android,webdriver.Ie]=None):
        i = 0
        for link in source_links:
            html = self.get_html(link,browser)
            title = self.get_title(html)
            file_title = self.validate_title(title)
            clean_text = self.get_clean_text(html)
            link_info = dict()
            link_info["html"] = html
            link_info["title"] = title
            link_info["file_title"] = file_title
            link_info["clean_text"] = clean_text
            link_info["source_link"] = link
            self._links_info.append(link_info)
            filepath = ospath.join(dirpath,file_title+"-id-"+str(i))
            with open(filepath,mode="w",encoding=self._file_encoding) as f:
                try:
                    f.write(json.dumps(link_info))
                except:
                    print("file IO error")
        return self

