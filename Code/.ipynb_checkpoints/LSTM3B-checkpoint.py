#@title Packages

import pandas as pd
import numpy as np
import random
import math
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, GRU, Bidirectional, Input, Attention, Concatenate, GlobalAveragePooling1D, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adagrad, Adadelta
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError, MeanAbsolutePercentageError, Huber
from tensorflow.keras.backend import sqrt, mean, square
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

#@tile Read and Prepare Data

def read_prepare_data(symbol):
    #read
    data = pd.read_csv('/Users/pedroalexleite/Desktop/Tese/Dados/dataset4.csv')
    train =  pd.read_csv('/Users/pedroalexleite/Desktop/Tese/Dados/train.csv')
    test =  pd.read_csv('/Users/pedroalexleite/Desktop/Tese/Dados/test.csv')

    #we're going to use only one symbol
    data = data[data['Symbol'] == symbol].copy()
    train = train[train['Symbol'] == symbol].copy()
    test = test[test['Symbol'] == symbol].copy()

    #we're going to use only the price variable
    data = data[['Date', '10Y Treasury Yield', 'Crude Oil Prices', 'Gold Prices', 'Consumer Confidence Proxy', 'Retail Sales Proxy',
                 'Fed Funds Rate', 'Consumer Price Index', 'Unemployment Rate', 'New Home Sales', 'M2 Money Supply', 'Close']].copy()
    train = train[['Date', '10Y Treasury Yield', 'Crude Oil Prices', 'Gold Prices', 'Consumer Confidence Proxy', 'Retail Sales Proxy',
                  'Fed Funds Rate', 'Consumer Price Index', 'Unemployment Rate', 'New Home Sales', 'M2 Money Supply', 'Close']].copy()
    test = test[['Date', '10Y Treasury Yield', 'Crude Oil Prices', 'Gold Prices', 'Consumer Confidence Proxy', 'Retail Sales Proxy',
                 'Fed Funds Rate', 'Consumer Price Index', 'Unemployment Rate', 'New Home Sales', 'M2 Money Supply', 'Close']].copy()

    #set date as index
    data.set_index('Date', inplace=True)
    train.set_index('Date', inplace=True)
    test.set_index('Date', inplace=True)

    #normalize
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train = pd.DataFrame(scaler.fit_transform(train), columns=train.columns, index=train.index)
    test = pd.DataFrame(scaler.transform(test), columns=test.columns, index=test.index)
    data = pd.DataFrame(scaler.transform(data), columns=data.columns, index=data.index)

    return scaler, data, train, test

scaler, data, train, test = read_prepare_data('AAPL')

#verify
#print(data.head())
#print(data.index)
#print(data.columns)

#@title Create Dataset

def create_dataset(dataframe, look_back):
    dataset = dataframe.values
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back, -1])

    return np.array(dataX), np.array(dataY)

#@title Reshape

def reshape(train, test, look_back):
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], trainX.shape[2]))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], testX.shape[2]))

    return trainX, trainY, testX, testY

#@title Forecast

def forecast_values(testY, look_back, horizon, model, last_sequence):
    testY_copy = testY.copy()
    last_sequence = last_sequence.copy()

    for val in range(0, horizon+1):
        a = last_sequence[-look_back:]
        a = np.reshape(a, (1, look_back, last_sequence.shape[-1]))
        a_predict = model.predict(a, verbose=0)[0]
        new_row = last_sequence[-1:].copy()
        new_row[0, -1] = a_predict
        last_sequence = np.vstack([last_sequence, new_row])
        testY_copy = np.append(testY_copy, a_predict)

    forecast = testY_copy[len(testY)+1:]
    return forecast

#@title Auxiliary Function

def predict_forecast_plot(data, train, test, trainX, trainY, testX, testY, nepochs, look_back, horizon, plot_predictions, model):
    #make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    #forecast (get the last sequence from testX for forecasting)
    last_sequence = testX[-1]
    forecast = forecast_values(testY, look_back, horizon, model, last_sequence)

    #invert predictions - need to handle multivariate data
    dummy = np.zeros((len(trainPredict), train.shape[1]))
    dummy[:, -1] = trainPredict.flatten()
    trainPredict = scaler.inverse_transform(dummy)[:, -1]

    dummy = np.zeros((len(trainY), train.shape[1]))
    dummy[:, -1] = trainY.flatten()
    trainY = scaler.inverse_transform(dummy)[:, -1]

    dummy = np.zeros((len(testPredict), train.shape[1]))
    dummy[:, -1] = testPredict.flatten()
    testPredict = scaler.inverse_transform(dummy)[:, -1]

    dummy = np.zeros((len(testY), train.shape[1]))
    dummy[:, -1] = testY.flatten()
    testY = scaler.inverse_transform(dummy)[:, -1]

    dummy = np.zeros((len(forecast), train.shape[1]))
    dummy[:, -1] = forecast.flatten()
    forecast = scaler.inverse_transform(dummy)[:, -1]

    #calculate root mean squared error
    trainScore = np.sqrt(mean_squared_error(trainY, trainPredict))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = np.sqrt(mean_squared_error(testY, testPredict))
    print('Test Score: %.2f RMSE' % (testScore))

    #plot predictions
    if plot_predictions==True:
        # Get the original Close prices
        original_data = scaler.inverse_transform(data)[:, -1]

        #shift train predictions for plotting
        trainPredictPlot = np.empty_like(original_data)
        trainPredictPlot[:] = np.nan
        trainPredictPlot[look_back:len(trainPredict)+look_back] = trainPredict

        #shift test predictions for plotting
        testPredictPlot = np.empty_like(original_data)
        testPredictPlot[:] = np.nan
        testPredictPlot[len(trainPredict)+(look_back*2)+1:len(original_data)-1] = testPredict

        #shift forecast for plotting
        forecastPlot = np.empty((len(original_data) + len(forecast),))
        forecastPlot[:] = np.nan
        forecastPlot[len(original_data):] = forecast

        #plot baseline, predictions and forecast
        plt.figure(figsize=(15,7))
        plt.plot(original_data, label='actual')
        plt.plot(trainPredictPlot, label='train set')
        plt.plot(testPredictPlot, label='test set')
        plt.plot(forecastPlot, label='forecast')
        plt.legend()
        plt.show()

    return testScore

#@title Train and Predict

def model(data, train, test, look_back=1, nepochs=10, horizon=10, plot_predictions=False):
    #reshape
    trainX, trainY, testX, testY = reshape(train, test, look_back)

    #create the network
    model = Sequential()
    model.add(LSTM(16, input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(Dropout(0.1))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    #model.summary()

    #fit
    model.fit(trainX, trainY, epochs=nepochs, batch_size=1, verbose=1)

    #predict, forecast and plot
    testScore = predict_forecast_plot(data, train, test, trainX, trainY, testX, testY, nepochs, look_back, horizon, plot_predictions, model)

    return testScore

#model(data, train, test, look_back=60, nepochs=50, horizon=10, plot_predictions=True)

#@title Run the Model Several Times

n_runs = 50
rmse_results = []
for i in range(n_runs):
    print(f"Running iteration {i+1}/{n_runs}...")
    test_rmse = model(data, train, test, look_back60, nepochs=50, horizon=10, plot_predictions=False)
    rmse_results.append(test_rmse)

rmse_results = np.array(rmse_results)
print("All RMSE results:", rmse_results)
print(f"Mean RMSE: {np.mean(rmse_results):.2f}")
print(f"Standard Deviation: {np.std(rmse_results):.2f}")
