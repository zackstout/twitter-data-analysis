
#Siraj:
import tweepy
from textblob import TextBlob
#codedex:
from tweepy.streaming import StreamListener
from tweepy import Stream
# import urllib
import json

import sentiment_mod as s

consumer_key = 'BnlEfaOWDLUt9lHFO2CyMFXLZ'
consumer_secret = 'sT0Gx7EGtu1uW7lll3AfhK6gOfzIT0OvWiT9b5JUXbzNn1I8KS'

access_token = '957037152231215104-3ZR1dN46VjYgXi0wpWwuFchN2zcpzTN'
access_token_secret = 'hyPQ0BaijmcGQxHXs90DzD88jlQFzyhGWTSXhNDfVoRkA'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

public_tweets = api.search('Trump')

# for tweet in public_tweets:
#     print(tweet.text)
#     analysis = TextBlob(tweet.text)
#     print(analysis.sentiment)


#codedex:
# class listener(StreamListener):
#     def on_data(self, data):
#         print(data)
#         return True
#     def on_error(self, status):
#         print(status)

class listener(StreamListener):
    def on_data(self, data):
        all_data = json.loads(data)

        tweet = all_data['text']

        sentiment_value, confidence = s.sentiment(tweet)
        print(tweet, sentiment_value, confidence)

        if confidence*100 >= 80:
            output = open("twitter-out.txt", "a")
            output.write(sentiment_value)
            output.write('\n')
            output.close()

        return True

    def on_error(self, status):
        print(status)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=['car'])

# Note: if you run this, it will print a continuous stream to the terminal. Close with ctrl+C.
