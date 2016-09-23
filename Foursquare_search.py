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


# TO BE CONTINUED....