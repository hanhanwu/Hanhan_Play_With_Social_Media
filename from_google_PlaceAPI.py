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
  print str(p_details["geometry"]["location"]["lat"])
  print str(p_details["geometry"]["location"]["lng"])
  
  for r in p_details["reviews"]:
    print r["rating"]
    print r["author_name"]
    print r["text"]
    print datetime.fromtimestamp(r["time"]).replace(tzinfo=pytz.utc).astimezone(local_tz).strftime('%Y-%m-%d, %H:%M:%S, PST')
    
  
  for wd in p_details["opening_hours"]["weekday_text"]:
    print wd
  
  print "Open Now: ", p_details["opening_hours"]["open_now"]
  
  break



# cell 2
# convert all the data into df, so that it's easier to do Spark sql operation

from datetime import datetime
from pytz import timezone
import pytz
from pyspark.sql import Row


palces_lst = []
local_tz = timezone('US/Pacific')
Merchant = Row("Name", "Open_Now", "Categories", "Address", "Lat_Lng", "Phone", "Website", "Rating", "Comment", "Rating_Time")
merchant_lst = []

for place in query_result.places:
  place.get_details()
  p_details = place.details
  
  p_name = place.name
  p_web = place.website
  p_addr = p_details["formatted_address"]
  p_lat_lng = str(p_details["geometry"]["location"]["lat"]) + ',' + str(p_details["geometry"]["location"]["lng"])
  if "formatted_phone_number" in p_details.keys():
    p_phone = p_details["formatted_phone_number"]
  else:
    p_phone = None
  if "openining_hours" in p_details.keys():
    p_open_now = p_details["opening_hours"]["open_now"]
  else:
    p_open_now = False
  p_categories = ','.join(p_details["types"])
  
  for r in p_details["reviews"]:
    r_rating = r["rating"]
    r_text = r["text"]
    t_time = datetime.fromtimestamp(r["time"]).replace(tzinfo=pytz.utc).astimezone(local_tz).strftime('%Y-%m-%d, %H:%M:%S, PST')
    merchant_lst.append(Merchant(p_name, p_open_now, p_categories, p_addr, p_lat_lng, p_phone, p_web, r_rating, r_text, t_time))

  
  

# cell 3
df = sc.parallelize(merchant_lst).toDF()
df.show()



# cell 4
df.registerTempTable("merchant_table")



# cell 5 - Spark SQL, Now I want to know top recommended bubble tea store near Redmond :)
## It seems that Google only returns at most 5 review coments for each merchant
%sql

select Name, Address, avg(Rating) as Avg_Rating, Count(Rating) as Rating_Count
from merchant_table
group by Name, Address
order by avg(Rating) desc



# cell 6
# calculate user's distance with those stores

from geopy.geocoders import Nominatim
from geopy.distance import vincenty
from decimal import *


def calculate_distance(merchant_loc, user_loc):
  geolocator = Nominatim()

  merchant_lat_lng = [Decimal(l) for l in merchant_loc.split(',')]
  al1 = (merchant_lat_lng[0], merchant_lat_lng[1])

  location2 = geolocator.geocode(user_loc)
  if location2 == None: return None
  al2 = (location2.latitude, location2.longitude)

  distce = vincenty(al1, al2).miles
  return distce
 

user_location = "Redmond Town Center, WA"
distUDF = udf(lambda r: calculate_distance(r, user_location))
df_dist = df.withColumn("Distance", distUDF(df.Lat_Lng))
df_dist.show()



# cell 7 - get merchants based on their average ratings and location close to the user
%sql

select Name, Address, Open_Now, avg(Rating) as Avg_Rating, Count(Rating) as Rating_Count, Distance
from merchant_distance_table
group by Name, Address, Open_Now, Distance
order by avg(Rating) desc, Distance asc
