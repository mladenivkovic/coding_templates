#!/usr/bin/env python3


import requests
from bs4 import BeautifulSoup

#  res = requests.get('https://codedamn.com')
#
#  print(res.text)
#  print(res.status_code)

page = requests.get("https://codedamn.com")
soup = BeautifulSoup(page.content, 'html.parser')
title = soup.title.text
print(title)
#  print(soup)
