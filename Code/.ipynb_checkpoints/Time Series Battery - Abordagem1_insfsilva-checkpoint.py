#!/usr/bin/env python
# coding: utf-8

# # Time Series Honda LSTM - Strategy 1
# ## Inês Silva, student number 202008362



import pandas as pb
import glob, os
import re
import time
#from dask import dataframe as dd
import numpy
import math

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error

from scipy.optimize import minimize
import statsmodels.tsa.api as smt
import statsmodels.api as sm

from tqdm import tqdm_notebook

from itertools import product

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

import warnings
warnings.filterwarnings('ignore')

#https://towardsdatascience.com/the-complete-guide-to-time-series-analysis-and-forecasting-70d476bfe775

#Reading the data file
data = pd.read_csv('/data/insfsilva/release_2019_07_08/2017_10_06_ITS1/201710060950/general/csv/vel.csv')
print(type(data))
data.index = pd.to_datetime(data.index)
##drop all rows with at least one missing value
#time_series_data = time_series_data.dropna()
##print(time_series_data)

print("Let's check the current data:")
print(data.head())

plt.figure(figsize=(10, 7))
plt.plot(data['speed'], marker='o')
plt.xlabel("Date")
plt.ylabel("Speed")
#plt.title("Pandas Time Series Plot")
plt.grid(False)
plt.show()


# https://traces.readthedocs.io/en/latest/examples.html
#https://mathdatasimplified.com/2021/09/17/traces-a-python-library-for-unevenly-spaced-time-series-analysis/
import datetime
import traces

ts = data[['speed']]
print(type(ts))
#tsts = traces.TimeSeries(ts)

#Transform to evenly-spaced
#That will convert to a regularly-spaced time series using a moving average to avoid aliasing
#regular = tsts.moving_average(300, pandas=True)
#regular.head()



#https://pbpython.com/pandas-grouper-agg.html
data['iso_timestamp'] = pd.to_datetime(data['iso_timestamp'])  # Convert if not already in datetime
data = data.set_index('iso_timestamp')

# Group by hourly inter
#vals and calculate mean
#grouper_data = data.groupby(pd.Grouper(freq='H'))['speed'].mean()
# Inspect output
#print(grouper_data.head())

def tsplot(y, lags=None, figsize=(12, 7), syle='bmh'):
    with plt.style.context(style='bmh'):
        fig = plt.figure(figsize=figsize)
        layout = (1,2)
        acf_ax = plt.subplot2grid(layout, (0,0))
        pacf_ax = plt.subplot2grid(layout, (0,1))
        
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.05, use_vlines=True)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.05, use_vlines=True)
        plt.tight_layout()


# In[8]:
#tsplot(regular)


# In[9]:


plt.figure(figsize=(12, 5))
plt.plot(ts[1:200], marker='o')
plt.xlabel("Date")
plt.ylabel("First Values of Speed")
plt.grid(False)
plt.show()


# Conclusions:
# 
#     ACF and PACF plots show significant values, therefore the time series is not stationary.
#     
#     Problem: Performing differences consecutively does not turn the series into stationary!!!!
#     
#     Problem: other tests suggest stationarity... it looks stationary actually...

# In[10]:


#Deteção de Regimes:

#https://online.stat.psu.edu/stat510/lesson/6/6.1
#Periodogram: A periodogram is used to identify the dominant periods (or frequencies) of a time series. 
#This can be a helpful tool for identifying the dominant cyclical behavior in a series, particularly when
#the cycles are not related to the commonly encountered monthly or quarterly seasonality.

#import scipy
#pgram = scipy.signal.periodogram(ts)

#fig, (ax_t, ax_w) = plt.subplots(2, 1, constrained_layout=True)
#ax_t.plot(regular.index,regular, 'b+')
#ax_t.set_xlabel('Time [s]')
#ax_w.plot(pgram[0], np.sqrt(pgram[1]))
#ax_w.set_xlabel('Speed')
#ax_w.set_ylabel('Normalized amplitude')
#plt.show()


# The periodogram shows a dominant spike at a low frequency == dominant frequency in the signal!!
# 
# O sinal repete-se com uma frequência baixa -> ex: de 3 em 3 medições as medidas repetem-se. -> Não me parece "significativo" em termos de análise

# In[29]:


#fig, ax_w = plt.subplots(1, 1, constrained_layout=True)

#ax_w.plot(pgram[0][1:5000], np.sqrt(pgram[1][1:5000]))
#ax_w.set_xlabel('Speed ')
#ax_w.set_ylabel('Normalized amplitude')
#plt.show()


# In[161]:


#print(pgram[0][1:30])
#print(pgram[1][1:30])


# ### 4. Apply usual methods for time series:

# ### 4.1 SARIMA:

# In[8]:


#https://builtin.com/data-science/time-series-forecasting-python
#plt.plot(regular.index,regular, );
#plt.plot(regular.index,regular, );

#divisao treino/teste


#regular.index = pd.to_datetime(regular.index)

train = ts[0:200] #[regular['iso_timestamp'] < "2017-02-27T11:00:02.699568510-08:00"]
test = ts[201:220]    #[regular['iso_timestamp'] >= "2017-02-27T11:00:02.709563494-08:00"]


#plt.plot(train, color = "black")
#plt.plot(test, color = "red")
#plt.ylabel(data['speed'])
#plt.xlabel('Date')
#plt.xticks(rotation=45)
#plt.title("Train/Test split")
#plt.show()


# In[12]:


#https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html
#NÃO VOLTAR A CORRER! DEMORA!!!!

#1 ano = 525600 minutos
#1 ano = 105120 períodos de 5 minutos num ano
#import pmdarima
#optimal_model = pmdarima.arima.auto_arima(train,
#                                          max_p = 7)
#                                          m=5) #he period for seasonal differencing


#ARIMA(maxiter=50, method='lbfgs', order=(5, 1, 4), out_of_sample_size=0, seasonal_order=(0, 0, 0, 0))
#SARIMAXmodel = SARIMAX(train, order = (3,1,3), seasonal_order=(0,0,0,0))


#https://towardsdatascience.com/time-series-forecasting-with-arima-sarima-and-sarimax-ee61099e78f6
#optimal_model.plot_diagnostics(figsize=(15,12))
#plt.show()


# In[13]:


#optimal_model


# In[14]:


#from statsmodels.tsa.arima.model import ARIMA
#from statsmodels.tsa.statespace.sarimax import SARIMAX

#SARIMAXmodel = SARIMAX(train, order = (5,1,5), seasonal_order=(0,0,0,0))
#SARIMAXmodel = SARIMAXmodel.fit()

#https://www.projectpro.io/article/how-to-build-arima-model-in-python/544
#print(SARIMAXmodel.summary())


# In[15]:


#number of steps to forecast from the end of the sample
#y_pred = SARIMAXmodel.get_forecast(len(test.index))
#y_pred_df = y_pred.conf_int(alpha = 0.05) 
#y_pred_df["Predictions"] = SARIMAXmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
#y_pred_df.index = test.index
#y_pred_out = y_pred_df["Predictions"] 


#forecast
#n_periods = 5000
#index_of_fc = pd.date_range(y_pred_df.index[-1] + pd.DateOffset(minutes=5), periods = n_periods, freq='5T')
#fitted = SARIMAXmodel.predict(start = index_of_fc[0], end = index_of_fc[-1])


# make series for plotting purpose
#fitted_series = pd.Series(fitted, index=index_of_fc)
#lower_series = pd.Series(confint[:, 0], index=index_of_fc)
#upper_series = pd.Series(confint[:, 1], index=index_of_fc)

# Plot
'''
plt.figure(figsize=(10,5))
    
plt.plot(fitted_series, color='blue', label = 'SARIMA Forecast')
#plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.15)
plt.plot(train, color = "black")
plt.plot(test, color = "red")
plt.ylabel(data['speed'])
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.plot(y_pred_out, color='Yellow', label = 'SARIMA Predictions')
plt.legend()
plt.show()
'''
import numpy as np
from sklearn.metrics import mean_squared_error

#arma_rmse = np.sqrt(mean_squared_error(test.values, y_pred_df["Predictions"]))
#print("RMSE: ",arma_rmse)


# Modelo  usado: SARIMAX(train, order = (5,1,5), seasonal_order=(0,0,0,0))
# 
# 

# ### 4..2 LSTM:

# In[9]:


import pandas as pb
import glob, os
#import plotly.graph_objs as go
import re
import datetime
import time
#from dask import dataframe as dd
import numpy
import math
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
from tensorflow.keras.layers import Dropout

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import datetime
#import plotly
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#import pickle

import numpy as np
np.set_printoptions(threshold=np.inf)
pb.set_option('display.max_columns', None)
pb.set_option('display.max_rows', None)
from numpy import array
#from tslearn.clustering import TimeSeriesKMeans
#from tslearn.utils import to_time_series
#from tslearn.metrics import dtw
import gc
import tensorflow as tf


# In[10]:


os.environ["CUDA_VISIBLE_DEVICES"]="0,2"


# In[11]:


#https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
def create_dataset(dataframe, look_back=1):
    dataset = dataframe.values
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)


# In[12]:


def LSTM_forecast(testY, look_back, horizon, model):
    #Main idea: usar o conjunto de teste para criar um conjunto de "forecast"
    #ir criando o conjunto de "forecast" à medida que se fazem previsões uma a uma

    testY_copy = testY.copy()
    for val in range(0, horizon+1):
        a = np.transpose(testY_copy[-(1+look_back):-1])
        #especie de reshape
        a = np.array([a])
        # make predictions
        a_predict = model.predict(a)[0]
        a_predict = np.reshape(a_predict, (1, 1))

        testY_copy = np.concatenate((testY_copy,a_predict), axis=0)

    forecast = testY_copy[len(testY)+1:]
    #print(forecast)
    
    return forecast


# In[13]:


#regular == dataset
def normalize_split_LSTM_train_predict(regular, look_back=1, nepochs=10, 
                                       horizon=10, plot_predictions=False):
    #Normalize the dataset
    import pandas as pd
    from sklearn import preprocessing
    x = regular.values.reshape(-1, 1) #returns a numpy array
    scaler = preprocessing.MinMaxScaler()
    x_scaled = scaler.fit_transform(x)
    regular_scaled = pd.DataFrame(data=x_scaled)
    regular_scaled.index = regular.index
    
    # split into train and test sets
    #divisao treino/teste

    #train = regular_scaled[regular_scaled.index < pd.to_datetime("2017-02-27T11:00:02.699568510-08:00", format='%Y-%m-%dT%H:%M:%S.%f%z')]
    #test = regular_scaled[regular_scaled.index >= pd.to_datetime("2017-02-27T11:00:02.709563494-08:00", format='%Y-%m-%dT%H:%M:%S.%f%z')]
    #print(len(train), len(test))
    #datas anteriores usadas: "2019-12-01"
    
    # reshape into X=t and Y=t+1
    #look_back = 1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    #model.summary()
    model.fit(trainX, trainY, epochs=nepochs, batch_size=1, verbose=1)


    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    # forecast
    forecast = LSTM_forecast(testY, look_back, horizon, model)
    
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform(trainY)
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform(testY)
    forecast = scaler.inverse_transform(forecast)
    
    
    # calculate root mean squared error
    trainScore = np.sqrt(mean_squared_error(trainY, trainPredict))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = np.sqrt(mean_squared_error(testY, testPredict))
    print('Test Score: %.2f RMSE' % (testScore))

    if plot_predictions==True: 
        # shift train predictions for plotting
        trainPredictPlot = np.empty_like(regular_scaled)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
        # shift test predictions for plotting
        testPredictPlot = np.empty_like(regular_scaled)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(trainPredict)+(look_back*2)+1:len(regular_scaled)-1, :] = testPredict
        #shift forecast for plotting
        forecastPlot = np.empty_like(pd.concat([regular_scaled, pd.DataFrame(forecast)]))
        forecastPlot[:, :] = np.nan
        forecastPlot[len(regular_scaled):len(forecastPlot),:] = forecast
        # plot baseline, predictions and forecast
        plt.figure(figsize=(15,7))
        plt.plot(scaler.inverse_transform(regular_scaled))
        plt.plot(trainPredictPlot, label='train set')
        plt.plot(testPredictPlot, label='test set')
        plt.plot(forecastPlot, label='forecast')
        plt.legend()
        plt.show()
    
    return testScore


# In[15]:


#customer 1
#sarima_RMSE = [123.72, 123.72, 123.72, 123.72, 123.72, 123.72, 123.72, 123.72, 123.72, 123.72]
LSTM_nepochs_2 = [126.28, 131.10, 112.32, 130.51, 103.43, 96.81, 91.09, 97.63, 101.93, 143.46]
LSTM_nepochs_10 = [134.59, 126.74, 97.75, 99.83, 96.95, 89.18, 90.75, 116.87, 86.72, 161.03]
look_back = [1,2,3,4,5,6,7,8,9,10]

plt.figure(figsize=(10,7))
#plt.plot(look_back,sarima_RMSE, label='SARIMA order = (4,1,4)')
plt.plot(look_back, LSTM_nepochs_2, label='LSTM nepochs=2')
plt.plot(look_back, LSTM_nepochs_10, label='LSTM nepochs=10')
plt.legend()
plt.show()


# Conclusions (data for customer 1):
# 
#     Improve with more epochs? Yes!!! (2 vs. 10) depends...
#     
#     Improve with higher look back? Test error improves and then worsens... 
#     
#     Chosen: look_back=6, nepochs=10!!!!!!!!!!!
#     
#     
#     
#     LSTM vs. SARIMA: RMSE: test set:
#         SARIMA:                       123.72
#         LSTM, look_back=1, nepochs=2: 126.28                  LSTM, look_back=1, nepochs=10: 134.59 
#         LSTM, look_back=2, nepochs=2: 131.10                  LSTM, look_back=2, nepochs=10: 126.74
#         LSTM, look_back=3, nepochs=2: 112.32                  LSTM, look_back=3, nepochs=10: 97.75
#         LSTM, look_back=4, nepochs=2: 130.51                  LSTM, look_back=4, nepochs=10: 99.83
#         LSTM, look_back=5, nepochs=2: 103.43                  LSTM, look_back=5, nepochs=10: 96.95 
#         LSTM, look_back=6, nepochs=2: 96.81                   LSTM, look_back=6, nepochs=10: 89.18
#         LSTM, look_back=7, nepochs=2: 91.09                   LSTM, look_back=7, nepochs=10: 90.75
#         LSTM, look_back=8, nepochs=2: 97.63                   LSTM, look_back=8, nepochs=10: 116.87
#         LSTM, look_back=9, nepochs=2: 101.93                  LSTM, look_back=9, nepochs=10: 86.72
#         LSTM, look_back=10, nepochs=2:143.46                  LSTM, look_back=1, nepochs=2: 161.03
#          
#    

'''for look_back_value in look_back[7:]:
    print(look_back_value)
    for val in [2,10]:
        print(val)
        if val==2:
            LSTM_nepochs_2 += [normalize_split_LSTM_train_predict(trainX, look_back=look_back_value, nepochs=val,horizon=10)]
        else:
            LSTM_nepochs_10 += [normalize_split_LSTM_train_predict(trainX, look_back=look_back_value, nepochs=val,horizon=10)]
                
    

plt.figure(figsize=(10,7))
#plt.plot(look_back,sarima_RMSE, label='SARIMA order = (5,1,5)')
plt.plot(look_back, LSTM_nepochs_2, label='LSTM nepochs=2')
plt.plot(look_back, LSTM_nepochs_10, label='LSTM nepochs=10')
plt.legend()
plt.show()
'''

look_back_value = 2
val = 2

results = normalize_split_LSTM_train_predict(ts, look_back=look_back_value, nepochs=val,horizon=10)
