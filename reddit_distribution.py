# Generate reddit posts city, countries distribution
from geotext import GeoText
import operator

cities_dict = {}
countries_dict = {}

for v in posts.values():
  post_title = v['title']
  cities = GeoText(post_title).cities
  for city in cities:
    country = GeoText(city).country_mentions.keys()[0]
    city = city.lower()
    cities_dict.setdefault(city, 0)
    cities_dict[city] += 1
    countries_dict.setdefault(country, 0)
    countries_dict[country] += 1
    
    
sorted_cities_dict = sorted(cities_dict.items(), key=operator.itemgetter(1), reverse=True)
sorted_countries_dict = sorted(countries_dict.items(), key=operator.itemgetter(1), reverse=True) 

for city_count in sorted_cities_dict:
  print city_count
  
print
print '**************************************'
print

for country_count in sorted_countries_dict:
  print country_count
