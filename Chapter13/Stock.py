from sklearn.preprocessing import StandardScaler
from textblob import TextBlob
import pandas as pd
import os
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import cross_val_score
import numpy as np
from yahoo_finance import Share

def stringToSentiment(text):
    return TextBlob(text).sentiment.polarity


script_dir = os.path.dirname(__file__)
tweets = pd.read_csv(script_dir.replace("Chapter13","") + "data/so_many_tweets.csv")
print(tweets.head())
print(tweets.shape)

tweets['sentiment'] = tweets['Text'].apply(stringToSentiment) #applica una funziione alla colonna!
print(tweets.head())

tweets['Date'] = pd.to_datetime(tweets.Date)
print(tweets['Date'].head())

tweets.index = pd.RangeIndex(start = 0, stop = 52512, step = 1)
list(tweets.index)[:5]

tweets.index = tweets.Date
print(tweets.head())

daily_tweets = tweets[['sentiment']].resample('D').mean() #ricampiono i tweet calcolando la media
print(daily_tweets.head())

daily_tweets.sentiment.plot(kind='line')
plt.show()

#yahoo = Share("AAPL")
#historical_prices = yahoo.get_historical('2015-05-2', '2015-05-25')
#prices = pd.DataFrame(historical_prices)

prices = pd.read_csv(script_dir.replace("Chapter13","") + "data/AAPL.csv")
# l'indice del nuovo DF deve essere di tipo datetime
prices.index = pd.to_datetime(prices['Date'])
print(prices.info())

not_null_close = prices[prices['Close'].notnull()]

prices.Close = not_null_close.Close.astype('float')
prices.Volume = not_null_close.Volume.astype('float')

s = StandardScaler()

only_prices_and_volumes = prices[["Volume","Close"]]
price_volume_scaled = s.fit_transform(only_prices_and_volumes)
pd.DataFrame(price_volume_scaled, columns=["Volume","Close"]).plot()
plt.show()

merged = pd.concat([prices.Close, daily_tweets.sentiment],axis=1)
print(merged.head())
merged.dropna(inplace=True)
s = StandardScaler()
merged_scaled = s.fit_transform(merged)

pd.DataFrame(merged_scaled, columns=merged.columns).plot()
plt.show()

merged['yesterday_sentiment'] = merged['sentiment'].shift(1)
print(merged.head())

#REGRESSIONE
regression_df = merged[['yesterday_sentiment','Close']]
regression_df.dropna(inplace=True)
print(regression_df.head())

linreg = LinearRegression()
rmse_cv = np.sqrt(abs(cross_val_score(linreg,regression_df[['yesterday_sentiment']],
                                      regression_df['Close'], cv=3, scoring='mean_squared_error').mean()))
print(rmse_cv)

rf = RandomForestRegressor()
rmse_cv = np.sqrt(abs(cross_val_score(rf,regression_df[['yesterday_sentiment']],
                                      regression_df['Close'], cv=3, scoring='mean_squared_error').mean()))
print(rmse_cv)

