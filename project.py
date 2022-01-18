# packages
import datetime
import pandas as pd
import numpy as np
import yfinance as yf
#%%
# import data into Python
sp500 = pd.read_csv('sp500yahoo.csv')

# Start to build up the predictors:
#Daily Spread
sp500['Daily_Spread'] = (sp500['High']-sp500['Low']).shift(1)

#Opening Prices
sp500['Open_lag'] = sp500['Open'].shift(1)

#Lag of Adj close
sp500['1_day_close'] = sp500['Adj Close'].shift(1)

copy_5 = sp500['Adj Close'].copy()
copy_adj_5 = copy_5.rolling(window=5).mean()
copy_30 = sp500['Adj Close'].copy()
copy_adj_30 = copy_30.rolling(window=30).mean()

sp500['5_day_close'], sp500['30_day_close'] = copy_adj_5, copy_adj_30

#volatility in the past days
copy_5 = sp500['Adj Close'].copy()
copy_vol_5 = copy_5.rolling(window=5).std()
copy_30 = sp500['Adj Close'].copy()
copy_vol_30 = copy_30.rolling(window=30).std()

sp500['5_day_vol'], sp500['30_day_vol'] = copy_vol_5, copy_vol_30

#days of week
sp500['Date'] = pd.to_datetime(sp500.Date, format='%Y-%m-%d')
# make monday as 1 and sunday as 7
sp500['day_of_week'] = sp500['Date'].dt.dayofweek+1

#Days of month, year
sp500['month'] = sp500['Date'].dt.month

#Volume
sp500['Volume_lag'] = sp500['Volume'].shift(1)


#%%
sp500 = yf.download('^GSPC', start = '1950-01-03', end = '2021-12-03', progress = True)
