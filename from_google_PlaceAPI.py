# cell 0
# Google Places API for Web Services
from googleplaces import GooglePlaces, types, lang

API_KEY = "[YOUR API KEY]"    # apply one in Google API console
google_places = GooglePlaces(API_KEY)

query_result = google_places.nearby_search(location='Redmond, United States', keyword='Bubble Tea',radius=20000, types=[types.TYPE_FOOD])

if query_result.has_attributions:
    print query_result.html_attributions



# cell 1
# output ratings, reviews, open time, address, website and other details of a place

from datetime import datetime
from pytz import timezone
import pytz
from pyspark.sql import Row


palces_lst = []
local_tz = timezone('US/Pacific')

for place in query_result.places:
  place.get_details()
  p_details = place.details
  
  print place.name
  print place.website
  print p_details["formatted_address"]
  print p_details["formatted_phone_number"]
  
  for r in p_details["reviews"]:
    print r["rating"]
    print r["author_name"]
    print r["text"]
    print datetime.fromtimestamp(r["time"]).replace(tzinfo=pytz.utc).astimezone(local_tz).strftime('%Y-%m-%d, %H:%M:%S, PST')
    
  
  for wd in p_details["opening_hours"]["weekday_text"]:
    print wd
  
  print "Open Now: ", p_details["opening_hours"]["open_now"]
  
  break
