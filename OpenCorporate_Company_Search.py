import requests 
import json

# cell 1 - search companies based on key words, search query should obey ISO 8601 format
open_corp_search_url = "https://api.opencorporates.com/v0.4/companies/search?q=vancity"

r1 = requests.get(open_corp_search_url)
print r1.status_code
print r1.json()
print


# cell 2 - get detailed info of each returned company
companies = r1.json()['results']['companies']

for c in companies:
  print c['company']['company_type']
  print c['company']['name']
  print c['company']['company_number']
  print c['company']['current_status']
  for pre_name in c['company']['previous_names']:
    print '  ', pre_name['company_name']
  print c['company']['registered_address_in_full']
  print c['company']['registry_url']
  break
  
  
# cell 3 - based on the above output, choose the company number and check details of this company
open_corp_url = "https://api.opencorporates.com/v0.4/companies/ca/2421593?sparse=true"

r2 = requests.get(open_corp_url)
print r2.status_code
print r2.json()
print
