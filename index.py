
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
    svr_rbf.fit(dates, prices)

    plt.scatter(dates, prices, color='black', label='data')
    plt.plot(dates, svr_rbf.predict(dates), color='red', label='RBF model')


    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()

    return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]

get_data(csv)

predicted_price = predict_price(dates, prices, 29)

print(predicted_price)






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



# First example of Tflearn:
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



# Second example of Tflearn (from their documentation):
from tflearn.datasets import titanic
titanic.download_dataset('titanic_dataset.csv')
from tflearn.data_utils import load_csv
data, labels = load_csv('titanic_dataset.csv', target_column=0, categorical_labels=True, n_classes=2)

def preprocess(data, cols_to_ignore):
    for id in sorted(cols_to_ignore, reverse=True):
        [r.pop(id) for r in data]
    for i in range(len(data)):
        data[i][1] = 1. if data[i][1] == 'female' else 0.
    return np.array(data, dtype=np.float32)

to_ignore = [1, 6]
data = preprocess(data, to_ignore)

# Build the network:
net = tflearn.input_data(shape=[None, 6])
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.fit(data, labels, n_epoch=10, batch_size=16, show_metric=True)

dicaprio = [3, 'Jack Dawson', 'male', 19, 0, 0, 'N/A', 5.0000]
winslet = [1, 'Rose DeWitt Bukater', 'female', 17, 1, 2, 'N/A', 100.0000]
dicaprio, winslet = preprocess([dicaprio, winslet])
print('D rate:', pred[0][1])
print('W rate:', pred[1][1])










# print(train) wow that's a lot of numbers
