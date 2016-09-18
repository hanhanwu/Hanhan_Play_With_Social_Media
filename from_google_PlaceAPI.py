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



# cell 2
# convert all the data into df, so that it's easier to do Spark sql operation

from datetime import datetime
from pytz import timezone
import pytz
from pyspark.sql import Row


palces_lst = []
local_tz = timezone('US/Pacific')
Merchant = Row("Name", "Open_Now", "Address", "Phone", "Website", "Rating", "Comment", "Rating_Time")
merchant_lst = []

for place in query_result.places:
  place.get_details()
  p_details = place.details
  
  p_name = place.name
  p_web = place.website
  p_addr = p_details["formatted_address"]
  if "formatted_phone_number" in p_details.keys():
    p_phone = p_details["formatted_phone_number"]
  else:
    p_phone = None
  p_open_now = p_details["opening_hours"]["open_now"]
  
  for r in p_details["reviews"]:
    r_rating = r["rating"]
    r_text = r["text"]
    t_time = datetime.fromtimestamp(r["time"]).replace(tzinfo=pytz.utc).astimezone(local_tz).strftime('%Y-%m-%d, %H:%M:%S, PST')
    merchant_lst.append(Merchant(p_name, p_open_now, p_addr, p_phone, p_web, r_rating, r_text, t_time))
  
  

# cell 3
df = sc.parallelize(merchant_lst).toDF()
df.show()



# cell 4
df.registerTempTable("merchant_table")



# cell 5 - Spark SQL, Now I want to know top recommended bubble tea store near Redmond :)
%sql

select Name, Address, avg(Rating) as Avg_Rating
from merchant_table
group by Name, Address
order by avg(Rating) desc
