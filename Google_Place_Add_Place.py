
# cell 1 - Add a place to Google API, it will be searchable immediately from Nearby Search method
from googleplaces import GooglePlaces, types, lang, GooglePlacesError, GooglePlacesAttributeError
import requests

API_KEY = "[YOUR API KEY]"

post_url = "https://maps.googleapis.com/maps/api/place/add/json?key=" + API_KEY

r = requests.post(post_url, json={
  "location": {
    "lat": 49.27336625,
    "lng": -123.101124271294
  },
  "accuracy": 50,
  "name": "Vancity HQ",
  "address": "183 Terminal Ave, Vancouver, Canada",
  "types": ["bank"],
  "website": "Vancity"
})

print r.status_code
print r.json()



# cell 2 - search the place you added
google_places = GooglePlaces(API_KEY)
query_result = google_places.nearby_search(location='183 Terminal Ave, Vancouver, Canada', name='Vancity HQ',radius=20, types=["bank"])

if query_result.has_attributions:
    print query_result.html_attributions
    
for place in query_result.places:
    # Returned places from a query are place summaries.
    print place.name
    print place.geo_location
    print place.place_id

    # The following method has to make a further API call.
    place.get_details()
    # Referencing any of the attributes below, prior to making a call to
    # get_details() will raise a googleplaces.GooglePlacesAttributeError.
    print place.details # A dict matching the JSON response from Google.
    print place.local_phone_number
    print place.international_phone_number
    print place.website
    print place.url
    
    break
    


#cell 3 - delete the added palce
delete_url = "https://maps.googleapis.com/maps/api/place/delete/json?key=" + API_KEY
pid = r.json()["place_id"]

r_delete = requests.post(post_url, json={
    "place_id": pid
  })

print r_delete.status_code
