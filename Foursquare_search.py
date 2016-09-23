# cell 0 - app ids, latitude and longitude
from geopy.geocoders import Nominatim
from geopy.distance import vincenty

CLIENT_ID = "[YOUR CLIENT ID]"
CLIENT_SECRET = "[YOUR CLIENT SECRET]"

geolocator = Nominatim()
location1 = geolocator.geocode("Redmond Town Center, WA")   # it seems that cannot write abbreviation here
al1 = (location1.latitude, location1.longitude)
print al1



# cell 1 - search request
import requests 

ids = "client_id=" + CLIENT_ID + "&client_secret=" + CLIENT_SECRET
params = "&v=20160921&ll=47.67051715, -122.119421574994&query=bubble team"
search_url = "https://api.foursquare.com/v2/venues/search?"+ids+params
  
print search_url

r = requests.post(search_url)
print r.status_code
print r.json()


# TO BE CONTINUED....
