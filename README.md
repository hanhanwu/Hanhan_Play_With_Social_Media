# Hanhan_Play_With_Social_Media
play with social media and data mining


<b>Semantic Web</b>

* Microformats- Many website embed unified formats in the web, like those resume blocks in LinkedIn, small geo info in many websites like Wiki. This type of blocks have unified class name in HTML and they makes your data extraction life easier.
* Microformats Example - Extract geo data form Wiki: https://github.com/hanhanwu/Hanhan_Play_With_Social_Media/blob/master/microformats_extract_geo.py
* It will be good to see the location you are parsing from google map, here, I'm makin your life easier :)
 * Type `jupyter notebook` in your terminal (after you have installed IPython and Jupyter Notebook)
 * In the opened IPython Notebook in your browser, type all the code in this image: https://github.com/hanhanwu/Hanhan_Play_With_Social_Media/blob/master/Geo_Visualization.png
 * Now you will see the map of that location
* Without codind and find all the embeded microformat in a web page: http://microform.at/

* RDF (Resource Description Framework) - the semantic web's model for defining and enabling exchange of triples (subject, predicate and object of the sentence) 
* <b>FuXi</b> - A powerful logic-reasoning system for the semantic web that uses a technique called Forward Chaining to deduce new information form existing info by starting with s aet of facts, deriving new facts from the knows facts by applying a set of logical rules, and repeating this process till a particular can be approved or diapproved, or there is no more fact to derive.  https://code.google.com/archive/p/fuxi/wikis/Installation_Testing.wiki
* N3 - Simple but powerful syntex that expresses facts and ruels in RDF, FuXi is also using it. https://www.w3.org/DesignIssues/Notation3


* Extract Recipe data: https://github.com/hanhanwu/Hanhan_Play_With_Social_Media/blob/master/microformats_recipe_data.py


*********************************************

<b>YouTube Mining</b>

* DEF CON & Linear Optimization
  * I really like the talks in DEF CON, many hackers are there and some of them will do pretty cool demo while allowing you to learn deeper in cyber security. Therefore, every morning, if I don't need to leave very early, I will get up early to watch a DEF CON video. The more video I have watched, the more I have realized, there are some pretty cool video but others are not. Sometimes you will see very ambitious title of the videos, maybe they really did lots of work but you just cannot feel excited nor feel it's a good learning experience, then I would ask myself, why did I get up so early this morning just to watch such a video. So, here, I'm going to apply linear optimization on parsed Youtube DEF CON video, to help me choose videos for my morning
  * Install Google python: https://developers.google.com/api-client-library/python/
  * To get Youtube API Key: https://developers.google.com/youtube/v3/getting-started
  * Youtube Python Samples: https://github.com/hanhanwu/api-samples/tree/master/python
  * BTW, Google API sometimes looks confusing or too complex to me, if you want to see what does returned result look like, better just to print out the response result and check in JSON viewer: http://jsonviewer.stack.hu/
  * Python pulp - for linear optimization
    * General View: https://github.com/coin-or/pulp
    * Documentation: https://pythonhosted.org/PuLP/
  * Linear Optimization review: https://sites.math.washington.edu/~burke/crs/407/notes/section1.pdf
    * To make it simple, that is, you define a goal and set some constraints, finally try to get as close to this goal as possbile, within the constraints. I think, this concept is also very philosophy
  * My code: https://github.com/hanhanwu/Hanhan_Play_With_Social_Media/blob/master/DEF_CON_video_list_linear_optimization.ipynb
    * In this code, I went through the 5 steps in linear optimization:
      * Define the problem: such as minimize or maximize. In my case, I want to minimize views/likes (my objective function)
      * Create Decision Variables
      * Define Onjective Function: This step is very interesting, you are using an interger to multiple the decision variable you created, not string (although they may look like strings)
      * Set Constranits
      * Optimization
  * reference: https://www.analyticsvidhya.com/blog/2017/10/linear-optimization-in-python/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29


*********************************************

<b>Stackoverflow Mining</b>

* This is StackExchange API: https://api.stackexchange.com/docs
 * But something looks wrong in this API, because I didn't find a way to get all the answers of each question through this API.
 * "Posts" can be "Questions" or "Answers", and therefore you cannot search by tag.... Only "Questions" can search by tag
* Py-StackExchange: https://github.com/lucjon/Py-StackExchange
* Get your api key, and other authentication info: http://stackapps.com/apps/oauth/register
* "Answer" data: https://api.stackexchange.com/docs/types/answer
* "Question" data: https://api.stackexchange.com/docs/types/question
* My data extraction code: https://github.com/hanhanwu/Hanhan_Play_With_Social_Media/blob/master/Stackoverflow_data_extraction.py


*********************************************

<b>Twitter Mining</b>

1. Create a twitter app and get CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET: https://apps.twitter.com/
2. A convenient way to get your own twitter data: https://github.com/hanhanwu/Hanhan_Play_With_Social_Media/blob/master/twitter_oauth1.0.py


*********************************************

<b>Reddit Mining</b>

* Topic Modeling
 * About Spark LDA: https://databricks.com/blog/2015/03/25/topic-modeling-with-lda-mllib-meets-graphx.html
 * Spark LDA MLlib description: http://spark.apache.org/docs/latest/mllib-clustering.html#latent-dirichlet-allocation-lda
 * Spark LDA sample Python code: http://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.clustering.LDA
 * Spark DistributedLDAModel (it has `topDocumentsPerTopic()`, `topTopicsPerDocument()`, `topicDistributions`): http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.mllib.clustering.DistributedLDAModel
 * DistributedLDAModel Scala code:
 
 http://stackoverflow.com/questions/33072449/extract-document-topic-matrix-from-pyspark-lda-model

 http://spark.apache.org/docs/latest/mllib-clustering.html#latent-dirichlet-allocation-lda
 
 * Spark word2vec: http://spark.apache.org/docs/latest/mllib-feature-extraction.html#word2vec
 * My code - tf-idf & LDA modeling: https://github.com/hanhanwu/Hanhan_Play_With_Social_Media/blob/master/reddit_tfidf_LDA.py
 * My code - word2vec & kmeans: https://github.com/hanhanwu/Hanhan_Play_With_Social_Media/blob/master/reddit_word2vec_kmeans.py
 
<b>Note:</b> LDA is a type of clustering too. In my code, so far I think word2vec is more convenient to track origional words and see whether the results make sense. So far, I haven't found a way to convert the numbers generated by Spark HashingTF and Spark LDA into the original words....
-- LDA can be used when you have both training and testing data, just want to to prediction
-- Word2Vec can be used to search for silmilar words, with KMeans, they can help group words into clusters. In my code, I am also generated the histogram, showing the cluster distribution for each post. Using the numbers in histogram, we can do prediction like LDA. For example, in the training data, we have "yes"/"no" as label, we can use LDA or (generated histogram here + predictive model) to predict the label of the test data

 * scikit-learn NMF smaple Python code: http://scikit-learn.org/stable/auto_examples/applications/topics_extraction_with_nmf_lda.html
 * My code: https://github.com/hanhanwu/Hanhan_Play_With_Social_Media/blob/master/reddit_scikit_topic_modeling.py
 * In my code, I am using NMF and LDA respectively to extract top topics. And Python Scikit-Learn will simply give you the top topics! So straightforward!! Especially feel great after struggling doing this with Spark topic modeling algorithms.
 
<b>To Sum Up: </b>Obviously, if I just want to extract top topics, using scikit-learn is very convenient. But if I want to do predictive modeling, with training and testing data, Spark algorithms are still good choices
 
* Entity/Interaction Extraction

 * My code - top NN entities: https://github.com/hanhanwu/Hanhan_Play_With_Social_Media/blob/master/reddit_NN_entities.py
 * My code - top interactions: https://github.com/hanhanwu/Hanhan_Play_With_Social_Media/blob/master/reddit_interactions.py
 
<b>As you can see,</b>, sinple NLP techniques still plays great role in text analysis. I just extracted the top 50 NN entities, we could already find trends and topics that popular and make sense. When I go deeper into the semantic layer, we can see NNVBNN interactions are already smary enough to make more sense

* Key Words Search
 * 2 methods for key words search: https://github.com/hanhanwu/Hanhan_Play_With_Social_Media/blob/master/reddit_keywords_search.py
 * Reddit posts destribution on cities and countries: https://github.com/hanhanwu/Hanhan_Play_With_Social_Media/blob/master/reddit_distribution.py
 * I think the distribution can bew extended to more interesting level. After running my code you will find that most cuddling new posts are from London, but US is the country that has top 1 number of reddit new posts at this time, the posts amount is far more than any other countries. This can be ectended to culture mapping based on countries, states and time series analysis.

*********************************************

<b>Pokemon Go</b>

It's so popular now, I don't want to play this game, but it will be so much fun to play with its data

* Potential Data Sources
 * https://pokeapi.co/
 * Google+ page: https://plus.google.com/117587995505124458333/posts
 * Yelp, Instagram, Snapchat, Flickr, GitHub, Twitter: https://www.instagram.com/pokemon_go_/

* Yelp Exploration
 * <b>Code Part 1</b>: https://github.com/hanhanwu/Hanhan_Play_With_Social_Media/blob/master/Pokemon_Yelp_Explore_Part1.py
 * FINDING_1: Yelp search could return you very accurate business category when you type a search term, and it's NOT based on simple key words search, since I have checked those returned busienss results, very few of them contain the key words in the search term. The reason I think Yelp search is accurare, is because when I put 'Pokemon' as search team, it returns toy store as the top category, and check their snippet_text, some have mentioned pokemon card game or pokemon center (a game center). But when I put 'Pokemon Go' as the searth term, most of them are restaurant and later when I checked their snippet_text, many of them are Pokemon station
 * FINDING_2: For the same search term, close locations share very similar trends. For example, in my code, I used cities in Great Seattle and Metro Vancouver, they have very similar results, but when I input Los Angeles/New York, they have different trends. Based on this, I am thinking, would the order of categories help find close location, and therefore define culture circle?
 

*********************************************

<b>Culture Circle Project</b>

-- Regarding the findings through Pokemon Go, it is better to work with multiple social media to detect culture circle through multiple ways

* Geo Data 
 * United Nations code list by country: http://www.unece.org/cefact/locode/service/location.html
 * US UN/LOCODE raw data: https://github.com/hanhanwu/Hanhan_Play_With_Social_Media/blob/master/US_raw.txt
 * clean US_raw into unique city name and State name: https://github.com/hanhanwu/Hanhan_Play_With_Social_Media/blob/master/clean_us_raw_Spark2.0.py
 * Open Geo Data: http://www.datasciencecentral.com/profiles/blogs/6-online-tools-for-open-geo-data

* Geo API
 * Google Places API intro: https://developers.google.com/places/web-service/intro
 * Using Google Places API in python: https://github.com/slimkrazy/python-google-places

* Culture Subcategory Data Source
 * Subcategory Food - Foodology RSS: http://foodology.ca/feed/

* R Map Visualization
 * magrittr, a method replces R nexted functions: https://github.com/hanhanwu/magrittr
 * Leaflet for R: https://rstudio.github.io/leaflet/
 * Leaflet API Reference (check params): http://leafletjs.com/reference.html
 * Get latitude, longitude of a place: http://mondeca.com/index.php/en/any-place-en

* My Code
 * Pokemon catchers geo-location daily data collection, all the data will be stored in flickr_photo each day: https://github.com/hanhanwu/Hanhan_Play_With_Social_Media/blob/master/pokemon_human_activity.py


*********************************************

<b> API </b>

* Google API Python libraries: https://developers.google.com/api-client-library/python/apis/
* Speech to Text (online short words, online large file, offine, open source): https://www.quora.com/What-are-the-top-ten-speech-recognition-APIs

*********************************************

<b>FOR BUSINESS</b>

-- <b>Yelp</b>
* I really like Yelp business API! So well documented and easy to use! Yelp search and business API are really powerful, when the merchant info is very limited, but based on the list generated by Yelp API, can get lots of insights for further data analysis.
* Get Start: https://www.yelp.com/developers/documentation/v3/get_started
* Manage API Authentication: https://www.yelp.com/developers/documentation/v3/authentication
* Yelp Python: https://github.com/Yelp/yelp-python
* My code: https://github.com/hanhanwu/Hanhan_Play_With_Social_Media/blob/master/yelp_search_business.py
 * The code here is very simple, but the output is poswerful. You just set the param as the term you want to search, define the location, then you will get a list of business output, the search results are good


-- <b> Google Place API</b>
* Google Place API doc: https://developers.google.com/places/web-service/intro
* Place API - Nearby Search: https://developers.google.com/places/web-service/search
* Place API - Add Place: https://developers.google.com/places/web-service/add-place
* Place API - Place Auto Complete: https://developers.google.com/places/web-service/autocomplete
* Place API - Query Auto Complete: https://developers.google.com/places/web-service/query
* Google Merchant Category List: https://developers.google.com/places/supported_types
* Python Tutorial: https://github.com/slimkrazy/python-google-places
* My code - Nearby Search and merchant sorting methods: https://github.com/hanhanwu/Hanhan_Play_With_Social_Media/blob/master/Google_Place_Nearby_Search.py
* My code - Add a place to Google API and it will become searchable from Google Nearby Search method, immediately: https://github.com/hanhanwu/Hanhan_Play_With_Social_Media/blob/master/Google_Place_Add_Place.py
* My code - Place Auto Complete (it seems that, fater adding a place, it becomes available in Nearby search, but if you want it to be available in Auto Complete, needs Google approval): https://github.com/hanhanwu/Hanhan_Play_With_Social_Media/blob/master/Google_Place_AutoComplete.py
* My code - Query Auto Complete (it seems that, fater adding a place, it becomes available in Nearby search, but if you want it to be available in Query Auto Complete, needs Google approval): https://github.com/hanhanwu/Hanhan_Play_With_Social_Media/blob/master/Google_Place_QueryAutoComplete.py


-- <b> Foursquare API </b>
* Foursquare API overview: https://developer.foursquare.com/overview/
* Hierarchical venue categories (very specific): https://developer.foursquare.com/categorytree
* How to get access token (redirect_url doesn't require you to really create a website, I'm simply using my GitHub url): https://developer.foursquare.com/overview/auth
* API endpoints (this tells all types of POST/GET requests you could send): https://developer.foursquare.com/docs/
* Foursquare venue add: https://developer.foursquare.com/docs/venues/add
* My code - Search Place: https://github.com/hanhanwu/Hanhan_Play_With_Social_Media/blob/master/Foursquare_search.py


-- <b> OpenCorporate API </b>
* API document: http://api.opencorporates.com/documentation/API-Reference
* ISO Country Code: https://en.wikipedia.org/wiki/ISO_3166-1
* ISO 8601 format: https://en.wikipedia.org/wiki/ISO_8601
* My code - get registered company detailes: https://github.com/hanhanwu/Hanhan_Play_With_Social_Media/blob/master/OpenCorporate_Company_Search.py

*********************************************

<b>GOOD TO READ</b>

* Mobile Network Analysis through mitmproxy: http://www.shubhro.com/2014/12/18/reverse-engineering-kayak-mitmproxy/
* Twitter NLP related Analysis (the detailed descriptions are good!): https://www.analyticsvidhya.com/blog/2016/07/capstone-project/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29


*********************************************

<b>NOTES</b>

-- Flights API   (It seems that they are all used for commercial purpose, difficult to find free data...)

* Wego
 * http://support.wan.travel/hc/en-us/articles/200300495-API-Overview
 * http://support.wan.travel/hc/en-us/articles/200191669

* FlightAware  (The API Homepage looks so good, but needs your credit card info before getting API key...)
 * http://flightaware.com/commercial/flightxml/
 * World Aorport Database (not free): http://flightaware.com/commercial/data/airports

* Flight Data (a lot!): http://www.transtats.bts.gov/DL_SelectFields.asp?Table_ID=236
