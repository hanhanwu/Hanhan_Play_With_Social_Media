# cell 0 - app ids, latitude and longitude
from geopy.geocoders import Nominatim
from geopy.distance import vincenty

CLIENT_ID = "[YOUR CLIENT ID]"
CLIENT_SECRET = "[YOUR CLIENT SECRET]"

geolocator = Nominatim()
location1 = geolocator.geocode("Redmond Town Center, WA")   # it seems that cannot write abbreviation here
al1 = (location1.latitude, location1.longitude)
print al1



# cell 1 - search venues
import requests 

ids = "client_id=" + CLIENT_ID + "&client_secret=" + CLIENT_SECRET
params = "&v=20160921&ll=47.67051715, -122.119421574994&query=bubble team"
search_url = "https://api.foursquare.com/v2/venues/search?"+ids+params
  
print search_url

r = requests.post(search_url)
print r.status_code
print r.json()



# cell 2 - formatted venue
json_output = r.json()
all_venues = json_output['response']['venues']

for ve in all_venues:
  print ve['name']
  
  print 'here now: ', ve['hereNow']['count']
  print ve['hereNow']['summary']
  print 'specials: ', ve['specials']
  
  print 'distance: ', ve['location']['distance']
  print ve['location']['formattedAddress']
  print str(ve['location']['lat']) + ', ' + str(ve['location']['lng'])
  
  print 'tip count: ', ve['stats']['tipCount']
  print 'checkin count: ', ve['stats']['checkinsCount']
  
  print 'venue id: ', ve['id']
  print 'Categories: '
  for c in ve['categories']:
    print '  (', c['name'], ', ', c['id'], ')'
  break



# cell 3 - for more about each venue, you may need to get ACCESS_TOKEN first, before sending POST/GET requests
import requests 
import json

ACCESS_TOKEN = "[YOUR ACCESS TOKEN]"
PARAMS = "&v=YYYYMMDD"

likes_url = "https://api.foursquare.com/v2/venues/4ac83a79f964a520f7bb20e3/likes?oauth_token=" + ACCESS_TOKEN + PARAMS
print likes_url

hours_url = "https://api.foursquare.com/v2/venues/4ac83a79f964a520f7bb20e3/hours?oauth_token=" + ACCESS_TOKEN + PARAMS
print hours_url


r1 = requests.get(likes_url)
print r1.status_code
print r1.json()
print

r2 = requests.get(hours_url)
print r2.status_code
print r2.json()
print



# cell 4 - generate latitude, longitude based on Postal Code, latitude anf longitude play a significant role in many Foursquare requests
# latitude and longittude are important for Foursquare search
from geopy.geocoders import Nominatim
from geopy.distance import vincenty


def get_ll(postal_code):
  if postal_code == None: return None
  geolocator = Nominatim()
  location = geolocator.geocode(postal_code)   # it seems that cannot write abbreviation here
  if location == None: return None
  al = str(location.latitude) + ', ' + str(location.longitude)
  return al

llUDF = udf(lambda r: get_ll(r))
new_df = buz_member_df.withColumn("lat_lng", llUDF(buz_member_df.PostalCode)).cache()


