'''
Created on May 29, 2016
'''
from bs4 import BeautifulSoup
import  urllib2

def main():
    wiki_url = "https://en.wikipedia.org/wiki/Nanjing"
    
    try:
        wiki_page = urllib2.urlopen(wiki_url)
    except:
        print 'cannot open the page'
        
    contents = wiki_page.read()
    soup = BeautifulSoup(contents, 'lxml')
    geoTag = soup.find(True, 'geo')

    
    
    if geoTag and len(geoTag) > 1:
        lat = geoTag.find(True, 'latitude').string
        lon = geoTag.find(True, 'longitude').string 
        print 'location is at latitude:', lat, ' longitude:', lon
    elif len(geoTag) == 1:
        location = geoTag.string.split(';')
        lat = location[0].strip()
        lon = location[1].strip()
        print 'location is at latitude:', lat, ' longitude:', lon
    else:
        print 'no location info'
        
if __name__ == "__main__":
    main()
