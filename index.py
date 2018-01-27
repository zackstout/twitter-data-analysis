
import tweepy
from textblob import TextBlob

import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# plt.switch_backend('new_')

import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb


dates = []
prices = []

def get_data(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        for row in csvFileReader:
            dates.append(int(row[0].split('-')[0]))
            prices.append(float(row[1]))
    return

def predict_prices(dates, prices, x):
    dates = np.reshape(dates, (len(dates), 1))

    svr_lin = SVR(kernel= 'linear', C=1e3)
    svr_poly = SVR(kernel= 'poly', C=1e3, degree = 2)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_lin.fit(dates, prices)
    svr_poly.fit(dates, prices)
    svr_







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



train, test, _ = imdb.load_data(path='imdb.pkl', n_words=10000, valid_portion=0.1)
trainX, trainY = train
testX, testY = test

trainX = pad_sequences(trainX, maxlen=100, value=0.)
testX = pad_sequences(testX, maxlen=100, value=0.)

trainY = to_categorical(trainY, nb_classes=2)
testY = to_categorical(testY, nb_classes=2)

net = tflearn.input_data([None, 100])
net = tflearn.embedding(net, input_dim=10000, output_dim=128)
#will randomly remove a neuron from the circuit on occasion:
net = tflearn.lstm(net, 128, dropout=0.8)
#softmax is doing the work of sigmoid
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.0001, loss='categorical_crossentropy')

#this is training:
model = tflearn.DNN(net, tensorboard_verbose=0)
# oops looks like we need an = after validation_set
model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True, batch_size=32)






#Note, it's gonna take a minute or two or three to run the above code. Ok or way more. But it's doing the thing!! And loss is going down!!!!!!






# print(train) wow that's a lot of numbers
