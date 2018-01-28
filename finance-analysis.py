
# following codedex:
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web

style.use('ggplot')

# start = dt.datetime(2000, 1, 1)
# end = dt.datetime(2016, 12, 31)
#
# # a dataframe is similar to a spreadsheet:
# df = web.DataReader('TSLA', "yahoo", start, end)
# # generate a csv:
# df.to_csv('tsla.csv')

# # print first 6:
# print(df.head(6))
# # print last 6:
# print(df.tail(6))

# to read information (can also read from a DB, json, etc.):
df = pd.read_csv('tsla.csv', parse_dates = True, index_col = 0)
# print(df.head())

print(df[['Open', 'High']].head())

df['Adj Close'].plot()
plt.show()
