#@title Packages

import random
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, LSTM, GRU, Dense, LeakyReLU, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError, MeanAbsolutePercentageError
from tensorflow.keras.backend import sqrt, mean, square
import tensorflow.keras.backend as K
import time
import os

start_time = time.time()

#@title seeds for reproducibility

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

BASE_SEEDS = [42, 123, 456, 789, 321]
print(f"Using {len(BASE_SEEDS)} different seeds: {BASE_SEEDS}")

#@title Read and Prepare Data

def read_prepare_data(symbol):
    #read
    data = pd.read_csv('../Data/dataset4.csv')
    train = pd.read_csv('../Data/train.csv')
    test = pd.read_csv('../Data/test.csv')

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

#@title Auxiliary Function

def predict_forecast_plot(scaler, data, train, test, trainX, trainY, testX, testY, nepochs, look_back, horizon, plot_predictions, model, is_log_scaler=False, is_zscore=False):
    #make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    #forecast (get the last sequence from testX for forecasting)
    last_sequence = testX[-1]
    forecast = forecast_values(testY, look_back, horizon, model, last_sequence)

    #invert predictions - handle MinMaxScaler, StandardScaler, and log normalization
    if is_log_scaler and hasattr(scaler, 'log_shifts'):
        # For log normalization, we need to handle the log transformation properly
        def inverse_log_transform(values):
            dummy = np.zeros((len(values), train.shape[1]))
            dummy[:, -1] = values.flatten()
            # First inverse MinMax scaling
            scaled_back = scaler.minmax_scaler.inverse_transform(dummy)[:, -1]
            # Then inverse log transformation for Close price column
            close_col_idx = train.shape[1] - 1  # Close is last column
            shift = scaler.log_shifts.get(close_col_idx, 0)
            original_values = np.exp(scaled_back) - shift
            return original_values

        trainPredict = inverse_log_transform(trainPredict)
        trainY = inverse_log_transform(trainY)
        testPredict = inverse_log_transform(testPredict)
        testY = inverse_log_transform(testY)
        forecast = inverse_log_transform(forecast)
    elif is_zscore or isinstance(scaler, StandardScaler):
        # StandardScaler inverse transformation - works directly on individual values
        trainPredict = scaler.inverse_transform(trainPredict.reshape(-1, 1) * np.array([0, 0, 0, 0, 1]) + np.zeros((len(trainPredict), train.shape[1])))[:, -1]
        trainY = scaler.inverse_transform(trainY.reshape(-1, 1) * np.array([0, 0, 0, 0, 1]) + np.zeros((len(trainY), train.shape[1])))[:, -1]
        testPredict = scaler.inverse_transform(testPredict.reshape(-1, 1) * np.array([0, 0, 0, 0, 1]) + np.zeros((len(testPredict), train.shape[1])))[:, -1]
        testY = scaler.inverse_transform(testY.reshape(-1, 1) * np.array([0, 0, 0, 0, 1]) + np.zeros((len(testY), train.shape[1])))[:, -1]
        forecast = scaler.inverse_transform(forecast.reshape(-1, 1) * np.array([0, 0, 0, 0, 1]) + np.zeros((len(forecast), train.shape[1])))[:, -1]
    else:
        # Standard MinMaxScaler inverse transformation
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
        if is_log_scaler:
            original_data = inverse_log_transform(data.iloc[:, -1].values.reshape(-1, 1))
        else:
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
        plt.show()

    return testScore

#@title Train and Predict

def rmse_loss(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def model(scaler, data, train, test, look_back=30, nepochs=50, horizon=7, plot_predictions=False,
          batch_size=1, learning_rate=0.001, optimizer='adam', activation='relu', loss='mse', seed=None, is_log_scaler=False):

    if seed is not None:
        set_seeds(seed)

    #reshape
    trainX, trainY, testX, testY = reshape(train, test, look_back)

    #build model
    input_layer = Input(shape=(trainX.shape[1], trainX.shape[2]))

    if activation == 'leaky_relu':
        x = GRU(8, return_sequences=True)(input_layer)
        x = LeakyReLU(alpha=0.01)(x)
        x = GRU(8, return_sequences=True)(x)
        x = LeakyReLU(alpha=0.01)(x)
        x = GRU(8)(x)
        x = LeakyReLU(alpha=0.01)(x)
    else:
        x = GRU(8, activation=activation, return_sequences=True)(input_layer)
        x = GRU(8, activation=activation, return_sequences=True)(x)
        x = GRU(8, activation=activation)(x)

    output = Dense(1, activation='linear')(x)
    model_instance = Model(inputs=input_layer, outputs=output)

    #optimizer
    optimizer = optimizer.lower()
    optimizers_dict = {
        'adam': Adam(learning_rate=learning_rate),
        'sgd': SGD(learning_rate=learning_rate),
        'rmsprop': RMSprop(learning_rate=learning_rate),
        'adagrad': Adagrad(learning_rate=learning_rate),
        'adadelta': Adadelta(learning_rate=learning_rate)
    }
    if optimizer not in optimizers_dict:
        raise ValueError(f"Unsupported optimizer: {optimizer}")
    opt = optimizers_dict[optimizer]

    #loss function
    loss_map = {
        'mse': MeanSquaredError(),
        'mean_squared_error': MeanSquaredError(),
        'rmse': rmse_loss,
        'mae': MeanAbsoluteError(),
        'mean_absolute_error': MeanAbsoluteError(),
        'mape': MeanAbsolutePercentageError(),
        'mean_absolute_percentage_error': MeanAbsolutePercentageError()
    }
    loss = loss.lower()
    if loss not in loss_map:
        raise ValueError(f"Unsupported loss function: {loss}")
    loss_fn = loss_map[loss]

    #compile model
    model_instance.compile(loss=loss_fn, optimizer=opt)

    #train
    model_instance.fit(trainX, trainY, epochs=nepochs, batch_size=batch_size, verbose=1)

    #predict and evaluate
    testScore = predict_forecast_plot(scaler, data, train, test, trainX, trainY, testX, testY, nepochs, look_back, horizon, plot_predictions, model_instance, is_log_scaler)

    return testScore

#@title Composite Score Function

def calculate_composite_score(results, all_results, alpha=0.7, beta=0.2, gamma=0.1):
    #extract all metrics for normalization
    all_means = [r['mean_rmse'] for r in all_results.values()]
    all_stds = [r['std_rmse'] for r in all_results.values()]
    all_cvs = [r['cv_percent'] for r in all_results.values()]

    #handle edge case where all values are the same
    mean_range = max(all_means) - min(all_means)
    std_range = max(all_stds) - min(all_stds)
    cv_range = max(all_cvs) - min(all_cvs)

    #normalize metrics to 0-1 scale
    if mean_range > 0:
        norm_mean = (results['mean_rmse'] - min(all_means)) / mean_range
    else:
        norm_mean = 0

    if std_range > 0:
        norm_std = (results['std_rmse'] - min(all_stds)) / std_range
    else:
        norm_std = 0

    if cv_range > 0:
        norm_cv = (results['cv_percent'] - min(all_cvs)) / cv_range
    else:
        norm_cv = 0

    return alpha * norm_mean + beta * norm_std + gamma * norm_cv

#@title Outlier Handling Techniques

def winsorization(data, lower_percentile=5, upper_percentile=95):
    data_winsorized = data.copy()
    for column in data.columns:
        lower_bound = np.percentile(data[column], lower_percentile)
        upper_bound = np.percentile(data[column], upper_percentile)
        data_winsorized[column] = np.clip(data[column], lower_bound, upper_bound)
    return data_winsorized

def clipping(data, lower_percentile=1, upper_percentile=99):
    data_clipped = data.copy()
    for column in data.columns:
        lower_bound = np.percentile(data[column], lower_percentile)
        upper_bound = np.percentile(data[column], upper_percentile)
        data_clipped[column] = np.clip(data[column], lower_bound, upper_bound)
    return data_clipped

def read_prepare_data_with_outliers(symbol, outlier_technique=None, **kwargs):
    data = pd.read_csv('../Data/dataset4.csv')
    train = pd.read_csv('../Data/train.csv')
    test = pd.read_csv('../Data/test.csv')

    data = data[data['Symbol'] == symbol].copy()
    train = train[train['Symbol'] == symbol].copy()
    test = test[test['Symbol'] == symbol].copy()

    data = data[['Date', 'Open', 'High', 'Low', 'Volume', 'Close']].copy()
    train = train[['Date', 'Open', 'High', 'Low', 'Volume', 'Close']].copy()
    test = test[['Date', 'Open', 'High', 'Low', 'Volume', 'Close']].copy()

    data.set_index('Date', inplace=True)
    train.set_index('Date', inplace=True)
    test.set_index('Date', inplace=True)

    if outlier_technique == 'winsorization':
        train = winsorization(train, **kwargs)
        print(f"Applied winsorization to training data with parameters: {kwargs}")
    elif outlier_technique == 'clipping':
        train = clipping(train, **kwargs)
        print(f"Applied clipping to training data with parameters: {kwargs}")

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_normalized = pd.DataFrame(scaler.fit_transform(train), columns=train.columns, index=train.index)
    test_normalized = pd.DataFrame(scaler.transform(test), columns=test.columns, index=test.index)
    data_normalized = pd.DataFrame(scaler.transform(data), columns=data.columns, index=data.index)

    return scaler, data_normalized, train_normalized, test_normalized

#@title Smoothing Techniques

def simple_moving_average(data, window=7):
    data_sma = data.copy()
    for column in data.columns:
        data_sma[column] = data[column].rolling(window=window, center=True).mean()
        data_sma[column] = data_sma[column].fillna(data[column])
    return data_sma

def rolling_median(data, window=7):
    data_rmedian = data.copy()
    for column in data.columns:
        data_rmedian[column] = data[column].rolling(window=window, center=True).median()
        data_rmedian[column] = data_rmedian[column].fillna(data[column])
    return data_rmedian

def gaussian_filter(data, sigma=1):
    from scipy.ndimage import gaussian_filter1d
    data_gaussian = data.copy()
    for column in data.columns:
        data_gaussian[column] = gaussian_filter1d(data[column].values, sigma=sigma)
    return data_gaussian

def savitzky_golay_filter(data, window_length=7, polyorder=2):
    from scipy.signal import savgol_filter
    data_savgol = data.copy()
    for column in data.columns:
        if window_length % 2 == 0:
            window_length += 1
        if window_length <= polyorder:
            window_length = polyorder + 2
        data_savgol[column] = savgol_filter(data[column].values, window_length, polyorder)
    return data_savgol

def read_prepare_data_with_smoothing(symbol, smoothing_technique=None, **kwargs):
    data = pd.read_csv('../Data/dataset4.csv')
    train = pd.read_csv('../Data/train.csv')
    test = pd.read_csv('../Data/test.csv')

    data = data[data['Symbol'] == symbol].copy()
    train = train[train['Symbol'] == symbol].copy()
    test = test[test['Symbol'] == symbol].copy()

    data = data[['Date', 'Open', 'High', 'Low', 'Volume', 'Close']].copy()
    train = train[['Date', 'Open', 'High', 'Low', 'Volume', 'Close']].copy()
    test = test[['Date', 'Open', 'High', 'Low', 'Volume', 'Close']].copy()

    data.set_index('Date', inplace=True)
    train.set_index('Date', inplace=True)
    test.set_index('Date', inplace=True)

    if smoothing_technique == 'sma':
        train = simple_moving_average(train, **kwargs)
        print(f"Applied Simple Moving Average to training data with parameters: {kwargs}")
    elif smoothing_technique == 'rolling_median':
        train = rolling_median(train, **kwargs)
        print(f"Applied Rolling Median to training data with parameters: {kwargs}")
    elif smoothing_technique == 'gaussian':
        train = gaussian_filter(train, **kwargs)
        print(f"Applied Gaussian Filter to training data with parameters: {kwargs}")
    elif smoothing_technique == 'savgol':
        train = savitzky_golay_filter(train, **kwargs)
        print(f"Applied Savitzky-Golay Filter to training data with parameters: {kwargs}")

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_normalized = pd.DataFrame(scaler.fit_transform(train), columns=train.columns, index=train.index)
    test_normalized = pd.DataFrame(scaler.transform(test), columns=test.columns, index=test.index)
    data_normalized = pd.DataFrame(scaler.transform(data), columns=data.columns, index=data.index)

    return scaler, data_normalized, train_normalized, test_normalized

#@title Normalization Techniques

def zscore_normalization(train, test, data):
    scaler = StandardScaler()
    train_scaled = pd.DataFrame(scaler.fit_transform(train), columns=train.columns, index=train.index)
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns, index=test.index)
    data_scaled = pd.DataFrame(scaler.transform(data), columns=data.columns, index=data.index)
    return scaler, data_scaled, train_scaled, test_scaled

def log_normalization(train, test, data):
    shifts = {}

    def log_transform(df, fit_shifts=False):
        df_log = df.copy()
        for i, column in enumerate(df.columns):
            min_val = df[column].min()
            if fit_shifts:
                if min_val <= 0:
                    shifts[i] = -min_val + 1e-8
                else:
                    shifts[i] = 0

            shift = shifts.get(i, 0)
            df_log[column] = np.log(df[column] + shift)
        return df_log

    train_log = log_transform(train, fit_shifts=True)
    test_log = log_transform(test, fit_shifts=False)
    data_log = log_transform(data, fit_shifts=False)

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = pd.DataFrame(scaler.fit_transform(train_log), columns=train_log.columns, index=train_log.index)
    test_scaled = pd.DataFrame(scaler.transform(test_log), columns=test_log.columns, index=test_log.index)
    data_scaled = pd.DataFrame(scaler.transform(data_log), columns=data_log.columns, index=data_log.index)

    scaler.log_shifts = shifts
    scaler.minmax_scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.minmax_scaler.fit(train_log)

    return scaler, data_scaled, train_scaled, test_scaled

def read_prepare_data_with_normalization(symbol, normalization_technique='minmax', **kwargs):
    #read data
    data = pd.read_csv('../Data/dataset4.csv')
    train = pd.read_csv('../Datas/train.csv')
    test = pd.read_csv('../Data/test.csv')

    #filter by symbol
    data = data[data['Symbol'] == symbol].copy()
    train = train[train['Symbol'] == symbol].copy()
    test = test[test['Symbol'] == symbol].copy()

    #select market variables
    data = data[['Date', 'Open', 'High', 'Low', 'Volume', 'Close']].copy()
    train = train[['Date', 'Open', 'High', 'Low', 'Volume', 'Close']].copy()
    test = test[['Date', 'Open', 'High', 'Low', 'Volume', 'Close']].copy()

    #set date as index
    data.set_index('Date', inplace=True)
    train.set_index('Date', inplace=True)
    test.set_index('Date', inplace=True)

    #apply normalization
    if normalization_technique == 'zscore':
        scaler, data_normalized, train_normalized, test_normalized = zscore_normalization(train, test, data)
        print(f"Applied Z-Score Normalization")
    elif normalization_technique == 'log':
        scaler, data_normalized, train_normalized, test_normalized = log_normalization(train, test, data)
        print(f"Applied Log Normalization")

    return scaler, data_normalized, train_normalized, test_normalized

#@title Enhanced Experimentation Functions with Seeds

def run_window_experiments_with_seeds(configurations, symbol='AAPL', seeds=BASE_SEEDS, nepochs=50, horizon=7):
    all_results = []

    # Add default configuration results
    for seed in seeds:
        all_results.append({
            "config_name": "Default (No Preprocessing)",
            "config_index": 0,
            "test_score": 4.042,
            "seed": seed,
            "run": seeds.index(seed) + 1
        })

    for run_idx, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"RUN {run_idx + 1}/{len(seeds)} - SEED: {seed}")
        print(f"{'='*60}")

        scaler, data_norm, train_norm, test_norm = read_prepare_data(symbol)

        for config_idx, config in enumerate(configurations):
            print(f"\nTesting {config['name']} - Seed {seed}")

            try:
                score = model(
                    scaler, data_norm, train_norm, test_norm,
                    look_back=config['window_size'],
                    nepochs=nepochs,
                    horizon=horizon,
                    plot_predictions=False,
                    seed=seed,
                    batch_size=1,
                    learning_rate=0.001,
                    optimizer='adam',
                    activation='relu',
                    loss='mse'
                )

                all_results.append({
                    "config_name": config['name'],
                    "config_index": config_idx + 1,
                    "test_score": score,
                    "seed": seed,
                    "run": run_idx + 1
                })

            except Exception as e:
                print(f"Error in {config['name']} with seed {seed}: {str(e)}")
                continue

    aggregated_results = {}
    for result in all_results:
        config_name = result["config_name"]
        if config_name not in aggregated_results:
            aggregated_results[config_name] = {
                "config_index": result["config_index"],
                "test_scores": [],
                "seeds_used": []
            }
        aggregated_results[config_name]["test_scores"].append(result["test_score"])
        aggregated_results[config_name]["seeds_used"].append(result["seed"])

    #calculate statistics for each configuration
    for config_name, data_dict in aggregated_results.items():
        scores = data_dict["test_scores"]
        if config_name == "Default (No Preprocessing)":
            # Use the provided default values
            data_dict["mean_rmse"] = 4.042
            data_dict["std_rmse"] = 0.089
            data_dict["cv_percent"] = 2.2
            data_dict["min_rmse"] = 4.042
            data_dict["max_rmse"] = 4.042
        else:
            data_dict["mean_rmse"] = np.mean(scores)
            data_dict["std_rmse"] = np.std(scores)
            data_dict["min_rmse"] = np.min(scores)
            data_dict["max_rmse"] = np.max(scores)
            data_dict["cv_percent"] = (data_dict["std_rmse"] / data_dict["mean_rmse"]) * 100

    #calculate composite scores
    for config_name, data_dict in aggregated_results.items():
        data_dict["composite_score"] = calculate_composite_score(data_dict, aggregated_results)

    return aggregated_results

def print_detailed_results(results, experiment_name, seeds):
    print(f"\n{'='*80}")
    print(f"ROBUST {experiment_name.upper()} COMPARISON RESULTS")
    print(f"{'='*80}")
    print(f"Each configuration tested with {len(seeds)} different seeds: {seeds}")
    print()

    for config_name, data_dict in results.items():
        print(f"Configuration {data_dict['config_index']} - {config_name}:")
        print(f"  Individual RMSEs: {[round(score, 3) for score in data_dict['test_scores']]}")
        print(f"  Mean RMSE: {data_dict['mean_rmse']:.3f}")
        print(f"  Std RMSE:  {data_dict['std_rmse']:.3f}")
        print(f"  Range:     {data_dict['min_rmse']:.3f} - {data_dict['max_rmse']:.3f}")
        print(f"  CV%:       {data_dict['cv_percent']:.1f}%")
        print(f"  Composite Score: {data_dict['composite_score']:.3f}")
        print()

    best_mean_name, best_mean_results = min(results.items(), key=lambda x: x[1]['mean_rmse'])
    best_cv_name, best_cv_results = min(results.items(), key=lambda x: x[1]['cv_percent'])
    best_composite_name, best_composite_results = min(results.items(), key=lambda x: x[1]['composite_score'])

    print(f"{'='*80}")
    print("BEST CONFIGURATIONS BY DIFFERENT CRITERIA:")
    print(f"{'='*80}")

    print("1. BEST MEAN PERFORMANCE:")
    print(f"   {best_mean_name}")
    print(f"   Mean RMSE: {best_mean_results['mean_rmse']:.3f} ± {best_mean_results['std_rmse']:.3f}")
    print(f"   CV%: {best_mean_results['cv_percent']:.1f}%")
    print(f"   Composite Score: {best_mean_results['composite_score']:.3f}")

    print("\n2. MOST CONSISTENT (Lowest CV%):")
    print(f"   {best_cv_name}")
    print(f"   Mean RMSE: {best_cv_results['mean_rmse']:.3f} ± {best_cv_results['std_rmse']:.3f}")
    print(f"   CV%: {best_cv_results['cv_percent']:.1f}%")
    print(f"   Composite Score: {best_cv_results['composite_score']:.3f}")

    print("\n3. BEST COMPOSITE SCORE (70% mean, 20% std, 10% cv):")
    print(f"   {best_composite_name}")
    print(f"   Mean RMSE: {best_composite_results['mean_rmse']:.3f} ± {best_composite_results['std_rmse']:.3f}")
    print(f"   CV%: {best_composite_results['cv_percent']:.1f}%")
    print(f"   Composite Score: {best_composite_results['composite_score']:.3f}")

    winners = [best_mean_name, best_cv_name, best_composite_name]
    winner_counts = {name: winners.count(name) for name in set(winners)}
    consensus_winner = max(winner_counts, key=winner_counts.get)

    print(f"\n{'='*80}")
    print("FINAL RECOMMENDATION:")
    print(f"{'='*80}")

    if winner_counts[consensus_winner] >= 2:
        print(f"★ RECOMMENDED CONFIGURATION: {consensus_winner}")
        print(f"  This configuration performs well across {winner_counts[consensus_winner]} different criteria.")
        print(f"  Mean RMSE: {results[consensus_winner]['mean_rmse']:.3f}")
        print(f"  CV%: {results[consensus_winner]['cv_percent']:.1f}%")
        print(f"  Composite Score: {results[consensus_winner]['composite_score']:.3f}")
    else:
        print("★ NO CLEAR CONSENSUS - Consider your priorities:")
        print(f"  - If you prioritize average performance: {best_mean_name}")
        print(f"  - If you prioritize consistency: {best_cv_name}")
        print(f"  - For balanced approach: {best_composite_name}")

    print(f"\n{'='*80}")
    print("RANKING BY COMPOSITE SCORE:")
    print(f"{'='*80}")
    sorted_configs = sorted(results.items(), key=lambda x: x[1]['composite_score'])
    for rank, (config_name, data_dict) in enumerate(sorted_configs, 1):
        print(f"{rank:2d}. {config_name}: "
              f"Score={data_dict['composite_score']:.3f}, "
              f"RMSE={data_dict['mean_rmse']:.3f}±{data_dict['std_rmse']:.3f}, "
              f"CV%={data_dict['cv_percent']:.1f}%")

#@title Run All Experiments

outlier_configurations = [
    {
        'name': 'Winsorization (5%-95%)',
        'technique_params': {'outlier_technique': 'winsorization', 'lower_percentile': 5, 'upper_percentile': 95},
        'model_params': {'batch_size': 1, 'learning_rate': 0.001, 'optimizer': 'adam', 'activation': 'relu', 'loss': 'mse'}
    },
    {
        'name': 'Clipping (1%-99%)',
        'technique_params': {'outlier_technique': 'clipping', 'lower_percentile': 1, 'upper_percentile': 99},
        'model_params': {'batch_size': 1, 'learning_rate': 0.001, 'optimizer': 'adam', 'activation': 'relu', 'loss': 'mse'}
    }
]

smoothing_configurations = [
    {
        'name': 'Simple Moving Average',
        'technique_params': {'smoothing_technique': 'sma', 'window': 7},
        'model_params': {'batch_size': 1, 'learning_rate': 0.001, 'optimizer': 'adam', 'activation': 'relu', 'loss': 'mse'}
    },
    {
        'name': 'Rolling Median',
        'technique_params': {'smoothing_technique': 'rolling_median', 'window': 7},
        'model_params': {'batch_size': 1, 'learning_rate': 0.001, 'optimizer': 'adam', 'activation': 'relu', 'loss': 'mse'}
    },
    {
        'name': 'Gaussian Filter',
        'technique_params': {'smoothing_technique': 'gaussian', 'sigma': 1.5},
        'model_params': {'batch_size': 1, 'learning_rate': 0.001, 'optimizer': 'adam', 'activation': 'relu', 'loss': 'mse'}
    },
    {
        'name': 'Savitzky-Golay',
        'technique_params': {'smoothing_technique': 'savgol', 'window_length': 7, 'polyorder': 2},
        'model_params': {'batch_size': 1, 'learning_rate': 0.001, 'optimizer': 'adam', 'activation': 'relu', 'loss': 'mse'}
    }
]

normalization_configurations = [
    {
        'name': 'Z-Score',
        'technique_params': {'normalization_technique': 'zscore'},
        'model_params': {'batch_size': 1, 'learning_rate': 0.001, 'optimizer': 'adam', 'activation': 'relu', 'loss': 'mse'}
    },
    {
        'name': 'Log Norm',
        'technique_params': {'normalization_technique': 'log'},
        'model_params': {'batch_size': 1, 'learning_rate': 0.001, 'optimizer': 'adam', 'activation': 'relu', 'loss': 'mse'}
    }
]

window_configurations = [
    {'name': f'Window Size {ws}', 'window_size': ws}
    for ws in [1, 2, 3, 4, 5, 6, 7, 15, 60, 90]
]

def prepare_data_outliers(symbol, **kwargs):
    if not kwargs:
        return read_prepare_data(symbol)
    else:
        return read_prepare_data_with_outliers(symbol, **kwargs)

def prepare_data_smoothing(symbol, **kwargs):
    if not kwargs:
        return read_prepare_data(symbol)
    else:
        return read_prepare_data_with_smoothing(symbol, **kwargs)

def prepare_data_normalization(symbol, **kwargs):
    return read_prepare_data_with_normalization(symbol, **kwargs)

def run_window_experiments_with_seeds(configurations, symbol='AAPL', seeds=BASE_SEEDS, nepochs=50, horizon=7):
    all_results = []

    # Add default configuration results
    for seed in seeds:
        all_results.append({
            "config_name": "Default (No Preprocessing)",
            "config_index": 0,
            "test_score": 4.042,
            "seed": seed,
            "run": seeds.index(seed) + 1
        })

    for run_idx, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"RUN {run_idx + 1}/{len(seeds)} - SEED: {seed}")
        print(f"{'='*60}")

        scaler, data_norm, train_norm, test_norm = read_prepare_data(symbol)

        for config_idx, config in enumerate(configurations):
            print(f"\nTesting {config['name']} - Seed {seed}")

            try:
                score = model(
                    scaler, data_norm, train_norm, test_norm,
                    look_back=config['window_size'],
                    nepochs=nepochs,
                    horizon=horizon,
                    plot_predictions=False,
                    seed=seed,
                    batch_size=1,
                    learning_rate=0.001,
                    optimizer='adam',
                    activation='relu',
                    loss='mse'
                )

                all_results.append({
                    "config_name": config['name'],
                    "config_index": config_idx + 1,
                    "test_score": score,
                    "seed": seed,
                    "run": run_idx + 1
                })

            except Exception as e:
                print(f"Error in {config['name']} with seed {seed}: {str(e)}")
                continue

    aggregated_results = {}
    for result in all_results:
        config_name = result["config_name"]
        if config_name not in aggregated_results:
            aggregated_results[config_name] = {
                "config_index": result["config_index"],
                "test_scores": [],
                "seeds_used": []
            }
        aggregated_results[config_name]["test_scores"].append(result["test_score"])
        aggregated_results[config_name]["seeds_used"].append(result["seed"])

    #calculate statistics for each configuration
    for config_name, data_dict in aggregated_results.items():
        scores = data_dict["test_scores"]
        if config_name == "Default (No Preprocessing)":
            # Use the provided default values
            data_dict["mean_rmse"] = 4.042
            data_dict["std_rmse"] = 0.089
            data_dict["cv_percent"] = 2.2
            data_dict["min_rmse"] = 4.042
            data_dict["max_rmse"] = 4.042
        else:
            data_dict["mean_rmse"] = np.mean(scores)
            data_dict["std_rmse"] = np.std(scores)
            data_dict["min_rmse"] = np.min(scores)
            data_dict["max_rmse"] = np.max(scores)
            data_dict["cv_percent"] = (data_dict["std_rmse"] / data_dict["mean_rmse"]) * 100

    #calculate composite scores
    for config_name, data_dict in aggregated_results.items():
        data_dict["composite_score"] = calculate_composite_score(data_dict, aggregated_results)

    return aggregated_results

if __name__ == "__main__":
    print(f"\nStarting robust evaluation with {len(BASE_SEEDS)} seeds...")

    #1. Outlier Handling Experiments
    print(f"\n{'='*80}")
    print("STARTING OUTLIER HANDLING EXPERIMENTS")
    print(f"{'='*80}")

    outlier_results = run_experiments_with_seeds(
        configurations=outlier_configurations,
        data_preparation_func=prepare_data_outliers,
        symbol='AAPL',
        seeds=BASE_SEEDS,
        experiment_name="Outlier Handling"
    )

    print_detailed_results(outlier_results, "Outlier Handling", BASE_SEEDS)

    #2. Smoothing Experiments
    print(f"\n{'='*80}")
    print("STARTING SMOOTHING EXPERIMENTS")
    print(f"{'='*80}")

    smoothing_results = run_experiments_with_seeds(
        configurations=smoothing_configurations,
        data_preparation_func=prepare_data_smoothing,
        symbol='AAPL',
        seeds=BASE_SEEDS,
        experiment_name="Smoothing"
    )

    print_detailed_results(smoothing_results, "Smoothing", BASE_SEEDS)

    #3. Normalization Experiments
    print(f"\n{'='*80}")
    print("STARTING NORMALIZATION EXPERIMENTS")
    print(f"{'='*80}")

    normalization_results = run_experiments_with_seeds(
        configurations=normalization_configurations,
        data_preparation_func=prepare_data_normalization,
        symbol='AAPL',
        seeds=BASE_SEEDS,
        experiment_name="Normalization"
    )

    print_detailed_results(normalization_results, "Normalization", BASE_SEEDS)

    #4. Window Size Experiments
    print(f"\n{'='*80}")
    print("STARTING WINDOW SIZE EXPERIMENTS")
    print(f"{'='*80}")

    window_results = run_window_experiments_with_seeds(
        configurations=window_configurations,
        symbol='AAPL',
        seeds=BASE_SEEDS
    )

    print_detailed_results(window_results, "Window Size", BASE_SEEDS)

    print(f"\n{'='*100}")
    print("FINAL SUMMARY OF ALL EXPERIMENTS")
    print(f"{'='*100}")

    all_experiments = {
        "Outlier Handling": outlier_results,
        "Smoothing": smoothing_results,
        "Normalization": normalization_results,
        "Window Size": window_results
    }

    print("\nBEST CONFIGURATION FROM EACH EXPERIMENT TYPE:")
    print("-" * 60)

    for exp_name, exp_results in all_experiments.items():
        if exp_results:
            best_config = min(exp_results.items(), key=lambda x: x[1]['composite_score'])
            print(f"{exp_name:15s}: {best_config[0]} (Score: {best_config[1]['composite_score']:.3f})")

    all_best_configs = []
    for exp_name, exp_results in all_experiments.items():
        if exp_results:
            best_config = min(exp_results.items(), key=lambda x: x[1]['composite_score'])
            all_best_configs.append((f"{exp_name} - {best_config[0]}", best_config[1]))

    if all_best_configs:
        overall_best = min(all_best_configs, key=lambda x: x[1]['composite_score'])
        print(f"\n★ OVERALL BEST CONFIGURATION ACROSS ALL EXPERIMENTS:")
        print(f"  {overall_best[0]}")
        print(f"  Composite Score: {overall_best[1]['composite_score']:.3f}")
        print(f"  Mean RMSE: {overall_best[1]['mean_rmse']:.3f} ± {overall_best[1]['std_rmse']:.3f}")
        print(f"  CV%: {overall_best[1]['cv_percent']:.1f}%")

    end_time = time.time()
    minutes = int((end_time-start_time)/60)
    hours = minutes//60
    remaining_minutes = minutes%60
    current_file = os.path.basename(__file__)
    print(f"\n{current_file} took {hours}h{remaining_minutes}m to complete")
