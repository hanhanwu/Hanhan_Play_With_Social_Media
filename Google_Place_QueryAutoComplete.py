import requests 
from googleplaces import GooglePlaces, types, lang, GooglePlacesError, GooglePlacesAttributeError

API_KEY = "[YOUR_API_KEY]"

params1 = "&input=Vancity Theater"
params2 = "&input=Vancity Theater&language=fr"
query_url = "https://maps.googleapis.com/maps/api/place/queryautocomplete/json?key="+API_KEY+params2
print query_url

r = requests.post(query_url)
print r.status_code
print r.json()
