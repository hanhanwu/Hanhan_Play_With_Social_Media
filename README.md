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

<b>Twitter Mining</b>

1. Create a twitter app and get CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET: https://apps.twitter.com/
2. A convenient way to get your own twitter data: https://github.com/hanhanwu/Hanhan_Play_With_Social_Media/blob/master/twitter_oauth1.0.py


*********************************************

<b>Reddit Mining</b>

* Topic Modeling
 * About Spark LDA: https://databricks.com/blog/2015/03/25/topic-modeling-with-lda-mllib-meets-graphx.html
 * Spark LDA MLlib sample code: http://spark.apache.org/docs/latest/mllib-clustering.html#latent-dirichlet-allocation-lda
 * scikit-learn NMF smaple Python code: http://scikit-learn.org/stable/auto_examples/applications/topics_extraction_with_nmf_lda.html
 
* Entity/Composition Extraction


* Key Words Search

*********************************************

<b>Good To Read</b>

Mobile Network Analysis through mitmproxy: http://www.shubhro.com/2014/12/18/reverse-engineering-kayak-mitmproxy/


*********************************************

<b>NOTES</b>

-- Flights API   (It seems that they are all used for commercial purpose, difficult to find free data...)

* Wego
 * http://support.wan.travel/hc/en-us/articles/200300495-API-Overview
 * http://support.wan.travel/hc/en-us/articles/200191669

* FlightAware  (The API Homepage looks so good, but needs your credit card info before getting API key...)
 * http://flightaware.com/commercial/flightxml/
 * World Aorport Database (not free): http://flightaware.com/commercial/data/airports
