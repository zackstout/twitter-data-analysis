
import tweepy
from textblob import TextBlob

consumer_key = 'BnlEfaOWDLUt9lHFO2CyMFXLZ'
consumer_secret = 'sT0Gx7EGtu1uW7lll3AfhK6gOfzIT0OvWiT9b5JUXbzNn1I8KS'

access_token = '957037152231215104-3ZR1dN46VjYgXi0wpWwuFchN2zcpzTN'
access_token_secret = 'hyPQ0BaijmcGQxHXs90DzD88jlQFzyhGWTSXhNDfVoRkA'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

public_tweets = api.search('Trump')

for tweet in public_tweets:
    print(tweet.text)
    analysis = TextBlob(tweet.text)
    print(analysis.sentiment)
