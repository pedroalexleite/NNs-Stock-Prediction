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
import time
import os

start_time = time.time()

#@tile Read and Prepare Data

def read_prepare_data(symbol):
    #read
    data = pd.read_csv('/Users/pedroalexleite/Desktop/Tese/Dados/dataset4.csv')
    train = pd.read_csv('/Users/pedroalexleite/Desktop/Tese/Dados/train.csv')
    test = pd.read_csv('/Users/pedroalexleite/Desktop/Tese/Dados/test.csv')

    #we're going to use only one symbol
    data = data[data['Symbol'] == symbol].copy()
    train = train[train['Symbol'] == symbol].copy()
    test = test[test['Symbol'] == symbol].copy()

    #we're going to use the price variable
    data = data[['Date', 'Close']].copy()
    train = train[['Date', 'Close']].copy()
    test = test[['Date', 'Close']].copy()

    #set date as index
    data.set_index('Date', inplace=True)
    train.set_index('Date', inplace=True)
    test.set_index('Date', inplace=True)

    #normalize
    scaler = MinMaxScaler(feature_range=(0, 1))
    train = pd.DataFrame(scaler.fit_transform(train), columns=train.columns, index=train.index)
    test = pd.DataFrame(scaler.transform(test), columns=test.columns, index=test.index)
    data = pd.DataFrame(scaler.transform(data), columns=data.columns, index=data.index)

    return scaler, data, train, test

scaler, data, train, test = read_prepare_data('AAPL')

#@title Create Dataset

def create_dataset(dataframe, look_back):
    dataset = dataframe.values
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])

    return np.array(dataX), np.array(dataY)

#@title Reshape

def reshape(train, test, look_back):
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], trainX.shape[2]))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], testX.shape[2]))

    return trainX, trainY, testX, testY

#@title Forecast

def forecast_values(testY, look_back, horizon, model):
    testY_copy = testY.copy()
    for val in range(0, horizon+1):
        a = testY_copy[-(1+look_back):-1]
        a = np.reshape(a, (1, look_back, 1))
        a_predict = model.predict(a, verbose=0)[0]
        a_predict = np.reshape(a_predict, (1, 1))
        testY_copy = np.concatenate((testY_copy, a_predict), axis=0)

    forecast = testY_copy[len(testY):]
    return forecast

#@title Auxiliary Function

def predict_forecast_plot(data, train, test, trainX, trainY, testX, testY, nepochs, look_back, horizon, plot_predictions, model):
    #make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    #forecast
    forecast = forecast_values(testY, look_back, horizon, model)

    #invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform(trainY)
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform(testY)
    forecast = scaler.inverse_transform(forecast)

    #calculate root mean squared error
    trainScore = np.sqrt(mean_squared_error(trainY, trainPredict))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = np.sqrt(mean_squared_error(testY, testPredict))
    print('Test Score: %.2f RMSE' % (testScore))

    #plot predictions
    if plot_predictions==True:
        #shift train predictions for plotting
        trainPredictPlot = np.empty_like(data)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

        #shift test predictions for plotting
        testPredictPlot = np.empty_like(data)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(trainPredict)+(look_back*2)+1:len(data)-1, :] = testPredict

        #shift forecast for plotting
        forecastPlot = np.empty_like(pd.concat([data, pd.DataFrame(forecast)]))
        forecastPlot[:, :] = np.nan
        forecastPlot[len(data):len(forecastPlot),:] = forecast

        #plot baseline, predictions and forecast
        plt.figure(figsize=(15,7))
        plt.plot(scaler.inverse_transform(data), label='real')
        plt.plot(trainPredictPlot, label='train set prediction')
        plt.plot(testPredictPlot, label='test set prediction')
        plt.plot(forecastPlot, label='forecast')
        plt.legend()
        plt.show()

    return testScore

#@title Train and Predict

def model(data, train, test, look_back=30, nepochs=50, horizon=7, plot_predictions=False):
    #reshape
    trainX, trainY, testX, testY = reshape(train, test, look_back)

    #create the network
    input_layer = Input(shape=(trainX.shape[1], trainX.shape[2]))
    x = LSTM(16, activation='relu')(input_layer)
    output = Dense(1, activation='linear')(x)
    model_instance = Model(inputs=input_layer, outputs=output)
    model_instance.compile(loss='mean_squared_error', optimizer='adam')
    #model.summary()

    #fit
    model_instance.fit(trainX, trainY, epochs=nepochs, batch_size=1, verbose=1)

    #predict, forecast and plot
    testScore = predict_forecast_plot(data, train, test, trainX, trainY, testX, testY, nepochs, look_back, horizon, plot_predictions, model_instance)

    return testScore

#@title Run the Model Several Times

n_runs = 30
rmse_results = []
for i in range(n_runs):
    print(f"Running iteration {i+1}/{n_runs}...")
    test_rmse = model(data, train, test, look_back=30, nepochs=50, horizon=7, plot_predictions=False)
    rmse_results.append(test_rmse)

rmse_results = np.array(rmse_results)
print("All RMSE results:", rmse_results)
print(f"Mean RMSE: {np.mean(rmse_results):.2f}")
print(f"Standard Deviation: {np.std(rmse_results):.2f}")

end_time = time.time()
minutes = int((end_time-start_time)/60)
hours = minutes//60
remaining_minutes = minutes%60
current_file = os.path.basename(__file__)
print(f"{current_file} took {hours}h{remaining_minutes}")
