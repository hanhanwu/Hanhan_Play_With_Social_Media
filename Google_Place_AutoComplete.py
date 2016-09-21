# Search Autocomplete
# A newly-added place does not appear in Text Search or Radar Search results, or to other applications, until it has been approved by the moderation process.

import requests 
from googleplaces import GooglePlaces, types, lang, GooglePlacesError, GooglePlacesAttributeError

API_KEY = "[YOUR API KEY]"

params1 = "input=Vancity&types=establishment&key="    # returns all palces with the input as keywords in their name
params2 = "input=183 Terminal Ave&types=geocode&key="    # returns only palces with keys words in input
params3 = "input=183 Terminal Ave&types=address&key="     # returns all the addresses contains the input as key words
params4 = "input=Vancouver, Canada&types=(regions)&key=" 
input_url = "https://maps.googleapis.com/maps/api/place/autocomplete/json?"+params2+API_KEY
print input_url

r = requests.post(input_url)
print r.status_code
print r.json()
