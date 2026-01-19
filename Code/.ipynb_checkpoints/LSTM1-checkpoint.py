#@title Packages

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

#@title Read and Prepare Data

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
    data = data[['Date', 'Close']].copy()
    train = train[['Date', 'Close']].copy()
    test = test[['Date', 'Close']].copy()

    #set date as index
    data.set_index('Date', inplace=True)
    train.set_index('Date', inplace=True)
    test.set_index('Date', inplace=True)

    #verify
    print(data.head())
    print(data.index)
    print(data.columns)

    #normalize
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)
    train = pd.DataFrame(scaler.fit_transform(train), columns=train.columns, index=train.index)
    test = pd.DataFrame(scaler.fit_transform(test), columns=test.columns, index=test.index)

    return scaler, data, train, test

scaler, data, train, test = read_prepare_data('AAPL')

#@title Create Dataset

def create_dataset(dataframe, look_back=1):
    dataset = dataframe.values
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])

    return np.array(dataX), np.array(dataY)

#@title LSTM Forecast

def LSTM_forecast(testY, look_back, horizon, model):
    testY_copy = testY.copy()
    for val in range(0, horizon+1):
        a = np.transpose(testY_copy[-(1+look_back):-1])
        a = np.array([a])
        a_predict = model.predict(a)[0]
        a_predict = np.reshape(a_predict, (1, 1))
        testY_copy = np.concatenate((testY_copy,a_predict), axis=0)

    forecast = testY_copy[len(testY)+1:]

    return forecast

#@title LSTM Models

def LSTM_models(trainX, look_back):
    layers = [1, 2, 3]
    neurons = [2, 4, 8, 16]
    dropouts = [0.1, 0.2]
    models = []
    configurations = []

    for num_layers in layers:
        for num_neurons in neurons:
            for dropout_rate in dropouts:
                model = Sequential()

                model.add(LSTM(num_neurons, input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=(num_layers > 1)))
                model.add(Dropout(dropout_rate))

                for layer_idx in range(1, num_layers):
                    model.add(LSTM(num_neurons, return_sequences=(layer_idx < num_layers - 1)))
                    model.add(Dropout(dropout_rate))

                model.add(Dense(1))

                model.compile(loss='mean_squared_error', optimizer='adam')

                models.append(model)
                configurations.append({
                    "layers": num_layers,
                    "neurons": num_neurons,
                    "dropout": dropout_rate
                })

    return models, configurations


look_back = 10
models, configurations = LSTM_models(trainX, look_back)

print("Configurations:")
for idx, config in enumerate(configurations, start=1):
    print(f"Model {idx}: Layers={config['layers']}, Neurons={config['neurons']}, Dropout={config['dropout']}")

#@title Train and Predict (Find the Optimal LSTM Model)

def LSTM_optimal_model(models, configurations, train, test, look_back=1, nepochs=10, horizon=10, plot_predictions=False):
    results = []

    for idx, (model, config) in enumerate(zip(models, configurations), start=1):
        print(f"Training Model {idx}: Layers={config['layers']}, Neurons={config['neurons']}, Dropout={config['dropout']}")
        try:
            #reshape
            trainX, trainY = create_dataset(train, look_back)
            testX, testY = create_dataset(test, look_back)
            trainX = np.reshape(trainX, (trainX.shape[0], data.shape[1, trainX.shape[1]))
            testX = np.reshape(testX, (testX.shape[0], data.shape[1, testX.shape[1]))

            #fit
            model.fit(trainX, trainY, epochs=nepochs, batch_size=1, verbose=1)

            #make predictions
            trainPredict = model.predict(trainX)
            testPredict = model.predict(testX)

            #forecast
            forecast = LSTM_forecast(testY, look_back, horizon, model)

            #invert predictions
            trainPredict = scaler.inverse_transform(trainPredict)
            trainY = scaler.inverse_transform(trainY)
            testPredict = scaler.inverse_transform(testPredict)
            testY = scaler.inverse_transform(testY)
            forecast = scaler.inverse_transform(forecast)

            #calculate root mean squared error
            trainScore = np.sqrt(mean_squared_error(trainY, trainPredict))
            testScore = np.sqrt(mean_squared_error(testY, testPredict))

            #append results
            results.append({
                "model_index": idx,
                "configuration": config,
                "train_score": trainScore,
                "test_score": testScore
            })

            #plot predictions
            if plot_predictions==True:
                #shift train predictions for plotting
                trainPredictPlot = np.empty_like(data)
                trainPredictPlot[:, :] = np.nan
                trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict

                #shift test predictions for plotting
                testPredictPlot = np.empty_like(data)
                testPredictPlot[:, :] = np.nan
                testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(data) - 1, :] = testPredict

                #shift forecast for plotting
                forecastPlot = np.empty_like(pd.concat([data, pd.DataFrame(forecast)]))
                forecastPlot[:, :] = np.nan
                forecastPlot[len(data):len(forecastPlot), :] = forecast

                #plot baseline, predictions and forecast
                plt.figure(figsize=(15, 7))
                plt.plot(scaler.inverse_transform(data))
                plt.plot(trainPredictPlot, label='train set')
                plt.plot(testPredictPlot, label='test set')
                plt.plot(forecastPlot, label='forecast')
                plt.legend()
                plt.title(f"Model {idx}: Layers={config['layers']}, Neurons={config['neurons']}, Dropout={config['dropout']}")
                plt.show()

        except Exception as e:
            print(f"Error with Model {idx + 1}: {e}")
            continue

    return results

def calculate_average_results(models, configurations, train, test, look_back=1, nepochs=10, horizon=10, runs=10):
    all_results = []

    for run in range(runs):
        print(f"Run {run + 1}/{runs}")
        results = LSTM_optimal_model(models, configurations, train, test, look_back, nepochs, horizon)
        all_results.extend(results)

    #aggregate results by model
    average_results = {}
    for result in all_results:
        model_idx = result["model_index"]
        if model_idx not in average_results:
            average_results[model_idx] = {
                "configuration": result["configuration"],
                "train_scores": [],
                "test_scores": []
            }
        average_results[model_idx]["train_scores"].append(result["train_score"])
        average_results[model_idx]["test_scores"].append(result["test_score"])

    #calculate average train and test scores
    for model_idx, scores in average_results.items():
        train_avg = np.mean(scores["train_scores"])
        test_avg = np.mean(scores["test_scores"])
        average_results[model_idx]["train_score_avg"] = train_avg
        average_results[model_idx]["test_score_avg"] = test_avg

    return average_results

models, configurations= LSTM_models(train, 10)
average_results = calculate_average_results(models, configurations, train, test, look_back=10, nepochs=100, horizon=10, runs=10)

for model_idx, scores in average_results.items():
    print(f"Model {model_idx}:")
    print(f"  Configuration: {scores['configuration']}")
    print(f"  Average Train Score: {scores['train_score_avg']:.2f}")
    print(f"  Average Test Score: {scores['test_score_avg']:.2f}")
