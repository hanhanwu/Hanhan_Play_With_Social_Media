from yelp.client import Client
from yelp.oauth1_authenticator import Oauth1Authenticator

auth = Oauth1Authenticator(
    consumer_key="[YOUR KEY]",
    consumer_secret="[YOUR CONSUMER SECRET]",
    token="[YOUR TOKEN]",
    token_secret="[YOUR TOKEN SECRET]"
)

client = Client(auth)

params = {
    'term': 'PT COQUITLAM'
}
response = client.search('Vancouver Canada', **params)
