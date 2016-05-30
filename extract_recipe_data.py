'''
Created on May 29, 2016
For all the foodies :)
Using About.com food here, the website has manny recipe
Get the ingredients and the cooking instruction
'''

from bs4 import BeautifulSoup
import  urllib2
import unicodedata
import json


def main():
    foodie_url = "http://britishfood.about.com/od/recipeindex/r/applepie.htm"
    
    try:
        foodie_page = urllib2.urlopen(foodie_url)
    except:
        print 'cannot open the page'
        
    contents = foodie_page.read()
    soup = BeautifulSoup(contents, 'lxml')
    food_preparation_tag = soup.find(True, 'preparation')
    food_ingredients_tag = soup.find(True, 'ingredients')
    
    recipe = {}
    preparation = []
    ingredients = []
    
    for s in food_preparation_tag.findAll(text=True):
        if s.strip() != '':
            s = s.lstrip().rstrip()
            if s.lower() == 'preparation' or s.lower() == 'method': continue
            preparation.append(unicodedata.normalize("NFKD", s).encode('ascii','ignore'))
            
    
    
    
    for s in food_ingredients_tag.findAll(text=True):
        if s.strip() != '':
            s = s.lstrip().rstrip()
            if s.lower() == 'ingredients': continue
            if s == 'Add to shopping list': break
            ingredients.append(unicodedata.normalize("NFKD", s).encode('ascii','ignore'))
    
    recipe['Ingredients'] = ingredients
    recipe['Instruction'] = preparation
    
    
    print json.dumps(recipe, indent=4)
    
if __name__ == "__main__":
    main()
