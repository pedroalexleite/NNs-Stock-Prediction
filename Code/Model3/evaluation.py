#@title Packages

import random
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, mean_absolute_error, r2_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, LSTM, GRU, Dense, LeakyReLU, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError, MeanAbsolutePercentageError
from tensorflow.keras.backend import sqrt, mean, square
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
import time
import os

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

    #check if symbol exists in data
    if len(data) == 0 or len(train) == 0 or len(test) == 0:
        return None, None, None, None

    #we're going to use the market variables
    data = data[['Date', 'Open', 'High', 'Low', 'Volume', 'Close']].copy()
    train = train[['Date', 'Open', 'High', 'Low', 'Volume', 'Close']].copy()
    test = test[['Date', 'Open', 'High', 'Low', 'Volume', 'Close']].copy()

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

#@title Calculate All Metrics

def calculate_all_metrics(actual, predicted):
    #regression
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / np.where(actual != 0, actual, 1))) * 100
    r2 = r2_score(actual, predicted)

    #classification (auxiliary)
    actual_direction = np.sign(np.diff(actual.flatten()))
    predicted_direction = np.sign(np.diff(predicted.flatten()))
    actual_binary = (actual_direction >= 0).astype(int)
    predicted_binary = (predicted_direction >= 0).astype(int)

    #classification
    accuracy = accuracy_score(actual_binary, predicted_binary) * 100
    precision = precision_score(actual_binary, predicted_binary, zero_division=0) * 100
    recall = recall_score(actual_binary, predicted_binary, zero_division=0) * 100
    f1 = f1_score(actual_binary, predicted_binary, zero_division=0) * 100

    return rmse, mse, mae, mape, r2, accuracy, precision, recall, f1

#@title Auxiliary Function

def predict_forecast_plot(data, train, test, trainX, trainY, testX, testY, nepochs, look_back, horizon, plot_predictions, model, scaler):
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

    #calculate all metrics for test set
    test_rmse, test_mse, test_mae, test_mape, test_r2, test_accuracy, test_precision, test_recall, test_f1 = calculate_all_metrics(testY, testPredict)

    print('Evaluation:')
    print('- Regression:')
    print(f'RMSE: {test_rmse:.2f}')
    print(f'MSE: {test_mse:.2f}')
    print(f'MAE: {test_mae:.2f}')
    print(f'MAPE: {test_mape:.2f}%')
    print(f'R²: {test_r2:.2f}')
    print('- Classification:')
    print(f'Accuracy: {test_accuracy:.2f}%')
    print(f'Precision: {test_precision:.2f}%')
    print(f'Recall: {test_recall:.2f}%')
    print(f'F1-Score: {test_f1:.2f}%')

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
        plt.plot(original_data, label='real')
        plt.plot(trainPredictPlot, label='train set prediction')
        plt.plot(testPredictPlot, label='test set prediction')
        plt.plot(forecastPlot, label='forecast')
        plt.legend()
        plt.title(f'Price Predictions and Forecast - {symbol}')
        plt.show()

    return (test_rmse, test_mse, test_mae, test_mape, test_r2, test_accuracy, test_precision, test_recall, test_f1,
            testPredict.flatten(), testY.flatten(), forecast.flatten())

#@title Train and Predict

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
    model_instance.compile(loss='mean_absolute_percentage_error', optimizer='adam')

    #fit
    model_instance.fit(trainX, trainY, epochs=nepochs, batch_size=1, verbose=1)

    #predict, forecast and plot
    results = predict_forecast_plot(data, train, test, trainX, trainY, testX, testY, nepochs, look_back, horizon, plot_predictions, model_instance, scaler)

    return results

#isk-return calculation function
def calculate_risk_return_tradeoff(predicted_prices, actual_prices=None):
    predicted_returns = np.diff(predicted_prices) / predicted_prices[:-1]
    expected_return = np.mean(predicted_returns)
    risk = np.std(predicted_returns)
    return expected_return, risk


print("Starting multi-symbol analysis...")
data_full = pd.read_csv('/Users/pedroalexleite/Desktop/NNs-Stock-Prediction/Data/dataset4.csv')
unique_symbols = data_full['Symbol'].unique()
print(f"Found {len(unique_symbols)} unique symbols")

look_back = 30
nepochs = 50
horizon = 7
plot_predictions = False
all_metrics = []
all_predictions = []

for i, symbol in enumerate(unique_symbols):
    print(f"\nProcessing symbol {i+1}/{len(unique_symbols)}: {symbol}")

    try:
        scaler, data, train, test = read_prepare_data(symbol)

        if scaler is None:
            print(f"  Skipping {symbol} - insufficient data")
            continue

        results = model(
            data=data,
            train=train,
            test=test,
            look_back=look_back,
            nepochs=nepochs,
            horizon=horizon,
            plot_predictions=plot_predictions
        )

        (test_rmse, test_mse, test_mae, test_mape, test_r2,
         test_accuracy, test_precision, test_recall, test_f1,
         test_predictions, test_actual, forecast) = results

        metrics_row = {
            'Symbol': symbol,
            'RMSE': test_rmse,
            'MSE': test_mse,
            'MAE': test_mae,
            'MAPE': test_mape,
            'R2': test_r2,
            'Accuracy': test_accuracy,
            'Precision': test_precision,
            'Recall': test_recall,
            'F1_Score': test_f1
        }
        all_metrics.append(metrics_row)

        predictions_row = {
            'Symbol': symbol,
            'Test_Predictions': test_predictions.tolist(),
            'Test_Actual': test_actual.tolist(),
            'Forecast_Values': forecast.tolist()
        }
        all_predictions.append(predictions_row)

    except Exception as e:
        print(f"  Error processing {symbol}: {e}")
        continue

print(f"\nSaving results for {len(all_metrics)} symbols...")
metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv('/Users/pedroalexleite/Desktop/NNs-Stock-Prediction/Data/final_results.csv', index=False)
print("Metrics saved to final_results.csv")

predictions_df = pd.DataFrame(all_predictions)

print("Calculating risk-return tradeoff...")
daily_returns = []
daily_risks = []

for idx, row in predictions_df.iterrows():
    try:
        test_predictions = row['Test_Predictions'] if isinstance(row['Test_Predictions'], list) else eval(row['Test_Predictions'])
        expected_return, risk = calculate_risk_return_tradeoff(test_predictions)
        daily_returns.append(expected_return)
        daily_risks.append(risk)
    except Exception as e:
        print(f"Error processing {row['Symbol']}: {e}")
        daily_returns.append(np.nan)
        daily_risks.append(np.nan)

risk_return_df = pd.DataFrame({
    'Symbol': predictions_df['Symbol'],
    'Daily_Return': daily_returns,
    'Daily_Risk': daily_risks
})

try:
    sp500 = pd.read_csv("/Users/pedroalexleite/Desktop/NNs-Stock-Prediction/Data/sp500.csv")
    merged_df = pd.merge(risk_return_df, sp500[['Symbol', 'Sector']], on='Symbol', how='left')
    merged_df.to_csv('/Users/pedroalexleite/Desktop/NNs-Stock-Prediction/Data/risk_return.csv', index=False)
    print("Risk-return analysis saved to risk_return.csv")
except Exception as e:
    print(f"Warning: Could not merge with sector data: {e}")
    risk_return_df.to_csv('/Users/pedroalexleite/Desktop/NNs-Stock-Prediction/Data/risk_return.csv', index=False)
    print("Risk-return analysis saved to risk_return.csv (without sector info)")

end_time = time.time()
minutes = int((end_time-start_time)/60)
hours = minutes//60
remaining_minutes = minutes%60
current_file = os.path.basename(__file__)
print(f"{current_file} took {hours}h{remaining_minutes}")
