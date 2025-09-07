import json
import re
from tqdm import tqdm

import requests
from bs4 import BeautifulSoup

def ieee_scrape(URL):

    headers =   {'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36'}

    ieee_content = requests.get(URL, timeout=300, headers= headers)
    soup = BeautifulSoup(ieee_content.content, "html.parser")
    scripts = soup.find_all("script")

    keyword_pattern = re.compile(r"(?<=\"keywords\":)\[{.*?}\]")
    author_pattern = re.compile(r"(?<=\"authorNames\":)\".*?\"")
    date_pattern = re.compile(r"(?<=\"displayPublicationDate\":)\".*?\"")
    publicationtitle_pattern = re.compile(r"(?<=\"displayPublicationTitle\":)\".*?\"")
    articleId_pattern = re.compile(r"(?<=\"articleId\":)\".*?\"")
    title_pattern = re.compile(r"(?<=\"title\":)\".*?\"")
    affiliation_pattern = re.compile(r"(?<=\"affiliation\":)\[.*?\]")
    abstract_pattern = re.compile(r"(?<=\"abstract\":)\".*?\"")

    IEEE_output_dict = {}
    keywords_dict = {}
    for i, script in enumerate(scripts):

        keywords = re.findall(keyword_pattern, str(script.string))
        authors = re.findall(author_pattern, str(script.string))
        date = re.findall(date_pattern, str(script.string))
        articleId = re.findall(articleId_pattern, str(script.string))
        title = re.findall(title_pattern, str(script.string))
        publicationtitle = re.findall(publicationtitle_pattern, str(script.string))
        affiliation = re.findall(affiliation_pattern, str(script.string))
        abstract = soup.select_one('meta[property="og:description"]')
        
        if len(keywords) == 1 and len(authors)!=0 and len(affiliation)!=0 and len(articleId)!=0 and len(publicationtitle)!=0 and len(date)!=0 and abstract!=None:

            raw_keywords_list = json.loads(keywords[0])
            for keyword_type in raw_keywords_list:
                keywords_dict[keyword_type["type"].strip()] = [kwd.strip() for kwd in keyword_type["kwd"]]
        
            
            IEEE_output_dict["Article_ID"] = articleId[0].strip("\"")
            IEEE_output_dict["Title"] = title[0].strip("\"")
            IEEE_output_dict["Published_in"] = publicationtitle[0].strip("\"")
            IEEE_output_dict["Date"] = date[0].strip("\"")
            IEEE_output_dict["Authors"] = authors[0].strip("\"[]").split(";")
            IEEE_output_dict["Affiliation"] = affiliation[0].strip("\"[]")
            IEEE_output_dict["Keywords"] = keywords_dict
            IEEE_output_dict["Abstract"] = abstract['content'].strip("\"\'")
    return IEEE_output_dict

if __name__ == "__main__":
    
    for i in tqdm(range(0, 10000000)):
        URL= "https://ieeexplore.ieee.org/document/" + str(i)
        ieee_output = ieee_scrape(URL)
    
