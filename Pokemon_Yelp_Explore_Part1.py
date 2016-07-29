# cell 1 - crtea Yelp client based on your keys

from yelp.client import Client
from yelp.oauth1_authenticator import Oauth1Authenticator

auth = Oauth1Authenticator(
    consumer_key="[your customer kwy]",
    consumer_secret="[your customer secret]",
    token="[your token]",
    token_secret="[your token secret]"
)

client = Client(auth)


# cell 2 - TYR1: try to find whether Yelp search is based on key words in current business content

params = {
    'term': 'Pokemon'
}
response = client.search('Seattle', **params)

ct = 0
for biz in response.businesses:
  print biz.name
  print biz.id
  print biz.image_url
  print biz.categories
  print biz.rating
  print biz.reviews
  
  print biz.location.address
  print biz.location.display_address  # full address
  print biz.location.cross_streets
  print biz.location.neighborhoods
  print biz.location.coordinate.latitude
  print biz.location.coordinate.longitude
  
  print biz.deals
  print biz.gift_certificates
  print biz.menu_provider
  print biz.reservation_url
  print biz.eat24_url
  print biz.reviews
  
  print biz.snippet_text
  print '********************'
  
  ct += 1
  if ct == 2: break



# cell 3 - TRY2: try to see whether 'Pokemon Go' and 'Pokemon' will find similar business categories

import operator

category_dct = {}
US_search_city_lst = ['Seattle', 'Redmond', 'Bellevue']
my_term = 'Pokemon Go'

params = {
    'term': my_term
}

for us_city in US_search_city_lst:
  us_response = client.search(us_city, **params)
  for biz in us_response.businesses:
    category_lst = [e[1] for e in biz.categories]
    for c in category_lst:
      category_dct.setdefault(c, 0)
      category_dct[c] += 1
    
sorted_category_lst = sorted(category_dct.items(), key=operator.itemgetter(1), reverse = True)

for itm in sorted_category_lst:
  print itm
    
##***************************##

my_term = 'Pokemon'

params = {
    'term': my_term
}

for us_city in US_search_city_lst:
  us_response = client.search(us_city, **params)
  for biz in us_response.businesses:
    category_lst = [e[1] for e in biz.categories]
    for c in category_lst:
      category_dct.setdefault(c, 0)
      category_dct[c] += 1
    
sorted_category_lst = sorted(category_dct.items(), key=operator.itemgetter(1), reverse = True)

for itm in sorted_category_lst:
  print itm

##***************************##

category_dct = {}
Canada_search_city_lst = ['Vancouver Canada', 'Burnaby Canada', 'Richmond Canada', 'Surrey Canada']
my_term = 'Pokemon Go'

params = {
    'term': my_term
}

for canada_city in Canada_search_city_lst:
  canada_response = client.search(us_city, **params)
  for biz in canada_response.businesses:
    category_lst = [e[1] for e in biz.categories]
    for c in category_lst:
      category_dct.setdefault(c, 0)
      category_dct[c] += 1
    
sorted_category_lst = sorted(category_dct.items(), key=operator.itemgetter(1), reverse = True)

for itm in sorted_category_lst:
  print itm
