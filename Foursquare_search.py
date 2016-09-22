# cell 0 - app ids
CLIENT_ID = "[YOUR CLIENT ID]"
CLIENT_SECRET = "[YOUR CLIENT SECRET]"


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
