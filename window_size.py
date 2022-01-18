# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 11:52:19 2021

@author: Skye Li
"""

import pandas as pd
import numpy  as np

def cal_VR(window_size):
    sp500 = pd.read_csv('sp500yahoo.csv')
    # This format eliminates the time fields
    sp500.index = pd.to_datetime(sp500.Date).dt.date
    del sp500['Date']
    sp500 = sp500.dropna()
    sp500['lagClose'] = sp500.Close.shift(1)
    sp500 = sp500[1:] 
    sp500['ret']=sp500['Close']/sp500['lagClose']-1.
    #retVec = sp500['ret'].values

    #VaR prob
    p = 0.05

    # roll quantiles and estimate rstar
    Rollq = sp500.rolling(window=window_size).quantile(p)
    sp500['rstar']=Rollq.ret
    # drop start of test window
    sp500 = sp500[window_size:]
    T = len(sp500)
    # exceptions
    sp500['eta']=sp500.ret<=sp500.rstar
    # rolling mean of exceptions for plotting
    #RollExcept = sp500.rolling(window=1000).mean() 

    # total exceptions and formal testing
    v1 = np.sum(sp500.eta)
    # this ratio should be one if all is working
    VR = v1/(p*T)
    
    return VR
    
    
## test window size: 0.5 year, 1 year, 1.5 year, 2 year, 2.5 year
# Assume 1 year have 250 trading days
VR_l = []
for window_size in [125, 250, 375, 500, 750]:
    VR_l.append(cal_VR(window_size))

print(VR_l)
