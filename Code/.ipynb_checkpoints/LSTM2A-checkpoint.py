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

#@title Models

def list_models(look_back, trainX):
    layers = [1, 2, 3]
    neurons = [8, 16, 32, 64, 128]
    models = []
    configurations = []

    for num_layers in layers:
        for num_neurons in neurons:
            input_layer = Input(shape=(trainX.shape[1], trainX.shape[2]))
            x = LSTM(num_neurons, activation='relu', return_sequences=(num_layers > 1))(input_layer)

            for layer_idx in range(1, num_layers):
                return_seq = layer_idx < (num_layers - 1)
                x = LSTM(num_neurons, activation='relu', return_sequences=return_seq)(x)

            output = Dense(1, activation='linear')(x)
            model = Model(inputs=input_layer, outputs=output)
            model.compile(loss='mean_squared_error', optimizer='adam')

            models.append(model)
            configurations.append({
                "layers": num_layers,
                "neurons": num_neurons,
                "activation": "relu"
            })

    return models, configurations

look_back = 30
trainX, trainY, testX, testY = reshape(train, test, look_back)
models, configurations = list_models(look_back, trainX)

#@title Train and Predict (Find the Optimal Model)

def optimal_model(models, configurations, data, train, test, look_back=30, nepochs=50, horizon=7, plot_predictions=False):
    results = []

    for idx, (model, config) in enumerate(zip(models, configurations), start=1):
        print(f"Training Model {idx}: Layers={config['layers']}, Neurons={config['neurons']}")

        #reshape
        trainX, trainY, testX, testY = reshape(train, test, look_back)

        #fit
        model.fit(trainX, trainY, epochs=nepochs, batch_size=1, verbose=1)

        #predict, foecast and plot
        testScore =  predict_forecast_plot(data, train, test, trainX, trainY, testX, testY, nepochs, look_back, horizon, plot_predictions, model)

        #append results
        results.append({
            "model_index": idx,
            "configuration": config,
            "test_score": testScore
        })

    return results

def calculate_average_results(models, configurations, data, train, test, look_back=30, nepochs=50, horizon=7, runs=5):
    all_results = []

    for run in range(runs):
        print(f"Run {run + 1}/{runs}")
        results = optimal_model(models, configurations, data, train, test, look_back, nepochs, horizon)
        all_results.extend(results)

    #aggregate results by model
    average_results = {}
    for result in all_results:
        model_idx = result["model_index"]
        if model_idx not in average_results:
            average_results[model_idx] = {
                "configuration": result["configuration"],
                "test_scores": []
            }
        average_results[model_idx]["test_scores"].append(result["test_score"])

    #calculate average train and test scores
    for model_idx, scores in average_results.items():
        test_avg = np.mean(scores["test_scores"])
        average_results[model_idx]["test_score_avg"] = test_avg

    return average_results

models, configurations= list_models(30, trainX)
average_results = calculate_average_results(models, configurations, data, train, test, look_back=30, nepochs=50, horizon=7, runs=5)

for model_idx, scores in average_results.items():
    print(f"Model {model_idx}:")
    print(f"  Configuration: {scores['configuration']}")
    print(f"  Average Test Score: {scores['test_score_avg']:.2f}")

end_time = time.time()
minutes = int((end_time-start_time)/60)
hours = minutes//60
remaining_minutes = minutes%60
current_file = os.path.basename(__file__)
print(f"{current_file} took {hours}h{remaining_minutes}")
