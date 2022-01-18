# packages

#import datetime
import pandas as pd
import yfinance as yf
import numpy as np

import warnings
warnings.filterwarnings("ignore")

#%%
# import data into Python
sp500_before = pd.read_csv("sp500yahoo.csv")
sp500_later = yf.download('^GSPC', start='2017-08-12', end='2021-12-04',progress=True).reset_index()
sp500 = sp500_before.append(sp500_later)

# Start to build up the predictors:
#Daily Spread
sp500['Daily_Spread'] = (sp500['High']-sp500['Low'])#.shift(1)

#Opening Prices
#sp500['Open_lag'] = sp500['Open'].shift(1)

#Lag of Adj close
sp500['1_day_close'] = sp500['Adj Close'].shift(1)

#Daily return
sp500['ret'] = sp500['Adj Close'] / sp500['1_day_close'] - 1

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
#sp500['Volume_lag'] = sp500['Volume'].shift(1)

#Prepare for y with window size of 2 years (500 days)
copy_500 = sp500['Adj Close'].copy()
copy_vol_500 = copy_500.rolling(window=500).std()
sp500['500_day_vol'] = copy_vol_500

sp500 = sp500.set_index('Date')
#%% Prepare for machine learning
y = sp500['500_day_vol'].values[500:]
X = sp500[["ret", "Daily_Spread", "5_day_close", "30_day_close", "5_day_vol", "30_day_vol", "day_of_week", "month"]].shift(1).iloc[500:]

#Create dummy variables for day of the week and month
days_df = pd.get_dummies(X['day_of_week'], drop_first=True)
days_df.columns = ["Tue", "Wed", "Thu", "Fri"]
months_df = pd.get_dummies(X['month'], drop_first=True)
X = pd.concat([X, days_df], axis=1)
X = pd.concat([X, months_df], axis=1)
del X["day_of_week"]
del X["month"]

#Prepare for training(1950-2012), validation(2013-2017), test sets(2018-2021)
X_train = X[X.index < pd.to_datetime('2013')]
X_val = X[(X.index >= pd.to_datetime('2013')) & (X.index < pd.to_datetime('2018'))]
X_test = X[X.index >= pd.to_datetime('2018')]

y_train = y[:15351]
y_val = y[15351:16610]
y_test = y[16610:]

#%% Machine Learning

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

#Linear Regression
lr = LinearRegression()
# Run on the validation sample
lr.fit(X_train,y_train)
y_pred_lr=lr.predict(X_val)
print("Training set R-squared:")
print(lr.score(X_train,y_train))
print("Validation set R-squared:")
print(lr.score(X_val,y_val))
print("Validation set MSE:")
print(mean_squared_error(y_val, y_pred_lr))


"""Ridge"""

param_l2 = []
train_r2_l2 = []
val_r2_l2 = []
val_mse_l2 = []
for alpha in [0.001, 0.01, 0.1, 0.5, 1.0, 10.0, 100.0, 1000.0]:
    ridge = Lasso(alpha=alpha)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_val)
    param_l2.append(alpha)
    train_r2_l2.append(ridge.score(X_train,y_train))
    val_r2_l2.append(ridge.score(X_val,y_val))
    val_mse_l2.append(mean_squared_error(y_val, y_pred_ridge))
    result_ridge = pd.DataFrame([param_l2, train_r2_l2, val_r2_l2, val_mse_l2]).transpose()
    result_ridge.columns = ["parameters", "train_R2", "val_R2", "val_MSE"]

#best parameter: 0.001

"""Lasso"""

param_l1 = []
train_r2_l1 = []
val_r2_l1 = []
val_mse_l1 = []
for alpha in [0.001, 0.01, 0.1, 0.5, 1.0, 10.0, 100.0, 1000.0]:
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)
    y_pred_lasso = lasso.predict(X_val)
    param_l1.append(alpha)
    train_r2_l1.append(lasso.score(X_train,y_train))
    val_r2_l1.append(lasso.score(X_val,y_val))
    val_mse_l1.append(mean_squared_error(y_val, y_pred_lasso))
    result_lasso = pd.DataFrame([param_l1, train_r2_l1, val_r2_l1, val_mse_l1]).transpose()
    result_lasso.columns = ["parameters", "train_R2", "val_R2", "val_MSE"]

#best parameter: 0.001

"""Regression Tree"""

grid = {
        "ccp_alpha": [0, 0.05, 0.1, 1.0],
        "max_features": [2,5,10,15], 
        "max_depth": [1, 3, 5, 15]}

param_dtr = []
train_r2_dtc = []
val_r2_dtc = []
val_mse_dtr = []

for g in ParameterGrid(grid):
    dtr = DecisionTreeRegressor(random_state=42, ccp_alpha=1.0, max_depth=1, max_features=15)
    dtr.set_params(**g)
    dtr.fit(X_train,y_train)
    y_pred_dtr = dtr.predict(X_val)
    param_dtr.append(g)
    train_r2_dtc.append(dtr.score(X_train,y_train))
    val_r2_dtc.append(dtr.score(X_val,y_val))
    val_mse_dtr.append(mean_squared_error(y_val, y_pred_dtr))
    result_dtr = pd.DataFrame([param_dtr, train_r2_dtc, val_r2_dtc, val_mse_dtr]).transpose()
    result_dtr.columns = ["parameters", "train_R2", "val_R2", "val_MSE"]

#best parameters: {'ccp_alpha': 1.0, 'max_depth': 1, 'max_features': 15}


"""Random Forest"""

grid = {
    "ccp_alpha": [0, 0.05, 0.1, 1.0],
    "max_features": [2,5,10,15], 
    "max_depth": [1, 3, 5, 15],
    "n_estimators": [1, 5, 10, 100]}

param_rfr = []
train_r2_rfr = []
val_r2_rfr = []
val_mse_rfr = []

for g in ParameterGrid(grid):
    rfr = RandomForestRegressor(random_state=42)
    rfr.set_params(**g)
    rfr.fit(X_train,y_train)
    y_pred_rfr = rfr.predict(X_val)
    param_rfr.append(g)
    train_r2_rfr.append(rfr.score(X_train,y_train))
    val_r2_rfr.append(rfr.score(X_val,y_val))
    val_mse_rfr.append(mean_squared_error(y_val, y_pred_rfr))
    result_rfr = pd.DataFrame([param_rfr, train_r2_rfr, val_r2_rfr, val_mse_rfr]).transpose()
    result_rfr.columns = ["parameters", "train_R2", "val_R2", "val_MSE"]
    

#best parameters: {'ccp_alpha': 1.0, 'max_depth': 1, 'max_features': 15, 'n_estimators': 1}


"""XGBoost"""

grid = {
    "alpha": [0, 0.05, 0.1, 1.0],
    "learning_rate": [0.1, 0.3, 0.6, 0.8], 
    "max_depth": [1, 3, 5, 15]}

param_xgb = []
train_r2_xgb = []
val_r2_xgb = []
val_mse_xgb = []

for g in ParameterGrid(grid):
    xgb_model = xgb.XGBRegressor(objective="reg:linear", random_state=42)
    xgb_model.set_params(**g)
    xgb_model.fit(X_train,y_train)
    y_pred_xgb = xgb_model.predict(X_val)
    param_xgb.append(g)
    train_r2_xgb.append(xgb_model.score(X_train,y_train))
    val_r2_xgb.append(xgb_model.score(X_val,y_val))
    val_mse_xgb.append(mean_squared_error(y_val, y_pred_xgb))
    result_xgb = pd.DataFrame([param_xgb, train_r2_xgb, val_r2_xgb, val_mse_xgb]).transpose()
    result_xgb.columns = ["parameters", "train_R2", "val_R2", "val_MSE"]

#best parameters: {'alpha': 1.0, 'learning_rate': 0.3, 'max_depth': 6}

##Compare between models
X_train_val = X_train.append(X_val)
y_train_val = np.append(y_train, y_val, 0)

model_name = []
test_R2 = []
test_MSE = []

#linear regression
lr = LinearRegression()
# Run on the full sample
lr.fit(X_train_val,y_train_val)
y_pred_lr = lr.predict(X_test)
model_name.append("Linear Regression")
test_R2.append(lr.score(X_test,y_test))
test_MSE.append(mean_squared_error(y_test, y_pred_lr))

#Rdige
ridge = Ridge(alpha=0.001)
ridge.fit(X_train_val,y_train_val)
y_pred_ridge = ridge.predict(X_test)
model_name.append("Ridge Regression")
test_R2.append(ridge.score(X_test,y_test))
test_MSE.append(mean_squared_error(y_test, y_pred_ridge))

#Lasso
lasso = Lasso(alpha=0.001)
lasso.fit(X_train_val,y_train_val)
y_pred_lasso = lasso.predict(X_test)
model_name.append("Lasso Regression")
test_R2.append(lasso.score(X_test,y_test))
test_MSE.append(mean_squared_error(y_test, y_pred_lasso))

#Decision Tree
dtr = DecisionTreeRegressor(max_depth=1, max_features=15)
dtr.fit(X_train_val,y_train_val)
y_pred_dtr = dtr.predict(X_test)
model_name.append("Decision Tree Regression")
test_R2.append(dtr.score(X_test,y_test))
test_MSE.append(mean_squared_error(y_test, y_pred_dtr))


#Random Forest
rfr = RandomForestRegressor(random_state=42, ccp_alpha=1.0, max_depth=1, max_features=15, n_estimators=1)
rfr.fit(X_train_val,y_train_val)
y_pred_rfr = rfr.predict(X_test)
model_name.append("Random Forest Regression")
test_R2.append(rfr.score(X_test,y_test))
test_MSE.append(mean_squared_error(y_test, y_pred_rfr))

#XGBoost
xgb_model = xgb.XGBRegressor(objective="reg:linear", random_state=42, alpha=1.0, learning_rate=0.3, max_depth=6)
xgb_model.fit(X_train_val,y_train_val)
y_pred_xgb = xgb_model.predict(X_test)
model_name.append("XGBoost Regression")
test_R2.append(xgb_model.score(X_test,y_test))
test_MSE.append(mean_squared_error(y_test, y_pred_xgb))

## Our best model is Lasso!
result_models = pd.DataFrame([model_name, test_R2, test_MSE]).transpose()
result_models.columns = ["Mode", "test_R2", "test_MSE"]

#%%% Back test
import matplotlib.pyplot as plt
import scipy.stats as stats

retm = np.mean(X_test.ret)
sp500['retv'] = (sp500['ret'] - retm)**2

# Code for a simple moving average: MA
nma = 500
rollma = sp500.rolling(window=nma,win_type="boxcar")
rollmeanmaFull = rollma.mean()

# Code for an exponential moving average: EWMA
# Use riskmetrics lambda = 0.94 parameter
rollewma = sp500.ewm(alpha=(1.-0.94),adjust=False)
rollmeanewmaFull = rollewma.mean()


# Start series after MA startup 
rollmeanma = rollmeanmaFull.iloc[17110:].copy()
rollmeanewma = rollmeanewmaFull.iloc[17110:].copy()
shrt = sp500.iloc[17110:].copy()
p = 0.05

# VaR critical return based on fixed volatility for comparison
mu = np.mean(shrt['ret'])
s  = np.std(shrt['ret'])
rstarfix = np.percentile(shrt['ret'],100.*p)


# set up two VaR critical returns for rolling volatility
# convert variances to standard deviations
normcrit = stats.norm.ppf(p,loc=0,scale=1)
rollmeanma['retsd']=np.sqrt(rollmeanma['retv'])
rollmeanewma['retsd']=np.sqrt(rollmeanewma['retv'])


#Standardized returns based on the rolling std
rollmeanma['stdret']=  (shrt['ret']-mu)/rollmeanma['retsd']
rollmeanewma['stdret']=(shrt['ret']-mu)/rollmeanewma['retsd']
criticalma = np.percentile(rollmeanma['stdret'],100.*p)
criticalewma =np.percentile(rollmeanewma['stdret'],100.*p)

rollmeanma['rstar']=mu + rollmeanma['retsd']*criticalma
rollmeanewma['rstar']=mu + rollmeanewma['retsd']*criticalewma

#ML method
ml_stdret=(shrt['ret']-mu)/y_pred_lasso
ml_critical = np.percentile(ml_stdret,100.*p)
ml_rstar = mu + y_pred_lasso*ml_critical

# Use critical R, rstar to find VaR levels
# Assume usual 100 price portfolio
shrt['varma']=100. - (1+rollmeanma['rstar'])*100
shrt['varewma']=100. - (1+rollmeanewma['rstar'])*100
shrt['varfixed'] =  100. - (1+rstarfix)*100.
ml_var = 100. - (1+ml_rstar)*100.


# build new dataframe of exceptions
# start with series 
# Remember to lag rstar values (applies to next day's return)
etemp = shrt['ret'] < rollmeanma['rstar'].shift(1)
exceptions = pd.DataFrame(etemp,columns=['ma'])
exceptions['ewma'] = shrt['ret'] < rollmeanewma['rstar'].shift(1)
exceptions['fixed'] = shrt['ret'] < rstarfix

exceptions['ml'] = shrt['ret'] < ml_rstar


# Get rolling average of exceptions per week / month / year / 2 years
exceptionsroll = exceptions.rolling(window=20)
exceptionsma = exceptionsroll.mean()

# plot routines
plt.plot(exceptionsma)
plt.legend(['ma','ewma','fixed', "ml"]) # should be close to 5%
plt.xlabel('Year')
plt.xticks(rotation=90)
plt.ylabel('Exceptions')
plt.grid()
    
# total exceptions and formal testing
v1_ma = np.sum(exceptions.ma)
v1_ewma = np.sum(exceptions.ewma)
v1_fixed = np.sum(exceptions.fixed)
v1_ml = np.sum(exceptions.ml)

np.mean(exceptions.ma)
np.mean(exceptions.ewma)
np.mean(exceptions.fixed)

np.mean(exceptions.ml)

T = len(shrt)
# this ratio should be one if all is working
VR_ma = v1_ma/(p*T)
VR_ewma = v1_ewma/(p*T)
VR_fixed = v1_fixed/(p*T)
VR_ml = v1_ml/(p*T)

print("VR of MA: ", VR_ma)
print("VR of EWMA: ", VR_ewma)
print("VR of fixed: ", VR_fixed)
print("VR of Machine Learning: ", VR_ml)