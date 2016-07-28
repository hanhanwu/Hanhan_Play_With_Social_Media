# cell 1

from yelp.client import Client
from yelp.oauth1_authenticator import Oauth1Authenticator

auth = Oauth1Authenticator(
    consumer_key="[your customer kwy]",
    consumer_secret="[your customer secret]",
    token="[your token]",
    token_secret="[your token secret]"
)

client = Client(auth)


# cell 2

import operator

category_dct = {}
US_search_city_lst = ['Seattle', 'Redmond', 'Bellevue']
snippet_text_set = set()
my_term = 'Pokemon Go'

params = {
    'term': my_term
}

for us_city in US_search_city_lst:
  us_response = client.search(us_city, **params)
  for biz in us_response.businesses:
    category_lst = [e[1] for e in biz.categories]
    if biz.snippet_text != None:
      snippet_text_set.add(biz.snippet_text)
    for c in category_lst:
      category_dct.setdefault(c, 0)
      category_dct[c] += 1
    
sorted_category_lst = sorted(category_dct.items(), key=operator.itemgetter(1), reverse = True)

for itm in sorted_category_lst:
  print itm
  
print
for t in list(snippet_text_set):
  if my_term.lower() in t.lower():
    print '*****',t
    print
    

# cell 3

import operator

category_dct = {}
US_search_city_lst = ['Seattle', 'Redmond', 'Bellevue']
snippet_text_set = set()
my_term = 'Pokemon'

params = {
    'term': my_term
}

for us_city in US_search_city_lst:
  us_response = client.search(us_city, **params)
  for biz in us_response.businesses:
    category_lst = [e[1] for e in biz.categories]
    if biz.snippet_text != None:
      snippet_text_set.add(biz.snippet_text)
    for c in category_lst:
      category_dct.setdefault(c, 0)
      category_dct[c] += 1
    
sorted_category_lst = sorted(category_dct.items(), key=operator.itemgetter(1), reverse = True)

for itm in sorted_category_lst:
  print itm
  
print
for t in list(snippet_text_set):
  print t
  if my_term.lower() in t.lower():
    print '*****',t
  print
