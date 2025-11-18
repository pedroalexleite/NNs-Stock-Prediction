import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, GRU, Dense
import tensorflow as tf
import random
import os
import time

start_time = time.time()

#@title Seeds for reproducibility

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

SEED = 42
print(f"Using seed: {SEED}")

#@title Read and Prepare Data

def read_prepare_data(symbol):
    #read
    data = pd.read_csv('/Users/pedroalexleite/Desktop/NNs-Stock-Prediction/Data/dataset4.csv')
    train = pd.read_csv('/Users/pedroalexleite/Desktop/NNs-Stock-Prediction/Data/train.csv')
    test = pd.read_csv('/Users/pedroalexleite/Desktop/NNs-Stock-Prediction/Data/test.csv')

    #we're going to use only one symbol
    data = data[data['Symbol'] == symbol].copy()
    train = train[train['Symbol'] == symbol].copy()
    test = test[test['Symbol'] == symbol].copy()

    #we're going to use the market variables
    data = data[['Date', 'Open', 'High', 'Low', 'Volume', 'Close']].copy()
    train = train[['Date', 'Open', 'High', 'Low', 'Volume', 'Close']].copy()
    test = test[['Date', 'Open', 'High', 'Low', 'Volume', 'Close']].copy()

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
        #get the original Close prices
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


def model(data, train, test, look_back=30, nepochs=50, horizon=7, plot_predictions=False, seed=SEED):
    #set seeds for reproducibility
    set_seeds(seed)

    #reshape
    trainX, trainY, testX, testY = reshape(train, test, look_back)

    #create the network
    input_layer = Input(shape=(trainX.shape[1], trainX.shape[2]))
    x = GRU(8, activation='relu', return_sequences=True)(input_layer)
    x = GRU(8, activation='relu', return_sequences=True)(x)
    x = GRU(8, activation='relu')(x)
    output = Dense(1, activation='linear')(x)
    model_instance = Model(inputs=input_layer, outputs=output)
    model_instance.compile(loss='mean_squared_error', optimizer='adam')

    #fit
    model_instance.fit(trainX, trainY, epochs=nepochs, batch_size=1, verbose=1)

    #predict, forecast and plot
    results = predict_forecast_plot(data, train, test, trainX, trainY, testX, testY, nepochs, look_back, horizon, plot_predictions, model_instance)

    return results

model(data, train, test, look_back=30, nepochs=50, horizon=7, plot_predictions=True, seed=SEED)
