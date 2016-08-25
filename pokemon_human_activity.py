## cell 1
# Get Relative Instagram Tags

import flickrapi
from datetime import date
from instagram.client import InstagramAPI
from nltk.stem.porter import *
import operator

Instagram_CLIENT_ID = "[you client ID]"
Instagram_CLIENT_SECRET = "[your client secret]"
Instagram_ACCESS_TOKEN = "[your access token]"

Instagram_api = InstagramAPI(access_token=Instagram_ACCESS_TOKEN,
                      client_id=Instagram_CLIENT_ID,
                      client_secret=Instagram_CLIENT_SECRET)
q = 'pokemongo'
stemmer = PorterStemmer()
count=100

related_tags = [stemmer.stem(t.name) for t in Instagram_api.tag_search(q, count)[0]]
tags_count = dict((i, related_tags.count(i)) for i in related_tags)
sorted_tags = sorted(tags_count.items(), key=operator.itemgetter(1), reverse=True)[:5]
print sorted_tags

tags = ''
for tg in sorted_tags:
  tags += tg[0]+','
  
print tags



## cell 2
import flickrapi
import json
from datetime import date, datetime
from pyspark.sql import Row

def get_location(pid):
  try:
    t = flickr.photos.geo.getLocation(photo_id=pid)
    return t
  except:
    return None

flickr_API_KEY = u'[your flickr API key]'
flickr_API_SECRET = u'[your flickr API secret]'
flickr = flickrapi.FlickrAPI(flickr_API_KEY, flickr_API_SECRET)

today = date.today()
weekday = datetime.now().strftime("%A")
pd = str(today)+" ("+weekday+")"
photo_list= flickr.photos_search(tags=tags, format='json')
photo_list_json = json.loads(photo_list)
pids = [e['id'] for e in photo_list_json['photos']['photo']]

rs = []
for pid in pids:
  t = get_location(pid)
  if t != None:
    lat = t[0][0].get('latitude')
    lng = t[0][0].get('longitude')
    rs.append(Row(id=pid, latitude=lat, longitude=lng, post_date=pd))
    
print len(rs)



## cell 3
flickr_location_df = sc.parallelize(rs).toDF().cache()
flickr_location_df.show()


## cell 4
today_str = str(today.year)+'_'+str(today.month)+'_'+str(today.day)
print today_str
filename = '/FileStore/flickr_photos/all_flickr_photo_'+today_str
flickr_location_df.coalesce(1).write.parquet(filename)



## cell 5, only use this once
%sql -- first time insertion
CREATE TABLE if not exists all_flickr_photo
USING org.apache.spark.sql.parquet
OPTIONS (
  path "/FileStore/flickr_photos/all_flickr_photo_[today's date]"
);



## cell 6
%sql
CREATE TABLE if not exists tmp_daily_table
USING org.apache.spark.sql.parquet
OPTIONS (
  path "/FileStore/flickr_photos/all_flickr_photo_[today's date]"
);



## cell 7
%sql -- Spark tables only support INSERT OVERWRITE for now
drop table if exists all_flickr_photo0;
create table all_flickr_photo0 as 
select * from all_flickr_photo



## cell 8
%sql  -- Don't run this cell in the first time
insert overwrite table all_flickr_photo
select * from all_flickr_photo0
 union all
select * from tmp_daily_table



## cell 9
%sql
select * from all_flickr_photo 
order by post_date desc



## cell 10
%sql
drop table tmp_daily_table;
show tables;



## cell 11 - convert latitude, longitude into city names
from pyspark.sql.functions import udf, concat_ws
from pyspark.sql.types import *
from urllib2 import urlopen
import json

flickr_location_df = spark.sql("select * from all_flickr_photo").cache()

def get_level1_locations(lat_lng):
    elems = lat_lng.split(',')
    url = "http://maps.googleapis.com/maps/api/geocode/json?"
    url += "latlng=%s,%s&sensor=false" % (float(elems[0]), float(elems[1]))
    v = urlopen(url).read()
    j = json.loads(v)
    if len(j['results'])>0:
      components = j['results'][0]['address_components']
      location_list=[]
      for c in components:
          if "locality" in c['types']:
            location_list.append(c['long_name'])
          if "administrative_area_level_1" in c['types']:
            location_list.append(c['long_name'])
          if "country" in c['types']:
            location_list.append(c['long_name'])
      location = ', '.join(filter(None, location_list))
      return location
    return ''
    
get_level1_locations_udf = udf(lambda lat_lng: get_level1_locations(lat_lng), StringType())
new_df1 = flickr_location_df.withColumn("level1_locations", get_level1_locations_udf(concat_ws(',', flickr_location_df.latitude, flickr_location_df.longitude).alias('lat_lng'))).cache()



## cell 12 - add city counts
new_df1.registerTempTable('HotSpots_level1')
new_df2 = sqlContext.sql("""
    SELECT count(id) as count, first(post_date) as post_date, 
    latitude, longitude,
    first(level1_locations) as location
    FROM
    HotSpots_level1
    GROUP BY latitude, longitude
    ORDER BY count desc
    """).cache()
new_df2.show(truncate=False)
