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
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError, MeanAbsolutePercentageError, Huber
from tensorflow.keras.backend import sqrt, mean, square
from tensorflow.keras import backend as K
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

def read_prepare_data_with_regularization(symbol):
    data = pd.read_csv('/Users/pedroalexleite/Desktop/Tese/Dados/dataset4.csv')
    train = pd.read_csv('/Users/pedroalexleite/Desktop/Tese/Dados/train.csv')
    test = pd.read_csv('/Users/pedroalexleite/Desktop/Tese/Dados/test.csv')

    data = data[data['Symbol'] == symbol].copy()
    train = train[train['Symbol'] == symbol].copy()
    test = test[test['Symbol'] == symbol].copy()

    data = data[['Date', 'Close']].copy()
    train = train[['Date', 'Close']].copy()
    test = test[['Date', 'Close']].copy()

    data.set_index('Date', inplace=True)
    train.set_index('Date', inplace=True)
    test.set_index('Date', inplace=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_normalized = pd.DataFrame(scaler.fit_transform(train), columns=train.columns, index=train.index)
    test_normalized = pd.DataFrame(scaler.transform(test), columns=test.columns, index=test.index)
    data_normalized = pd.DataFrame(scaler.transform(data), columns=data.columns, index=data.index)

    return scaler, data_normalized, train_normalized, test_normalized

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

def predict_forecast_plot(scaler, data, train, test, trainX, trainY, testX, testY, nepochs, look_back, horizon, plot_predictions, model):
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

def rmse_loss(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def apply_data_augmentation(trainX, trainY, noise_factor=0.01):
    noise = np.random.normal(0, noise_factor, trainX.shape)
    trainX_aug = trainX + noise
    return trainX_aug, trainY

def model_with_regularization(scaler, data, train, test, look_back=30, nepochs=50, horizon=7, plot_predictions=False,
                              batch_size=1, learning_rate=0.001, optimizer='adam', activation='relu', loss='mse',
                              regularization_technique=None, seed=None, **reg_params):

    if seed is not None:
        set_seeds(seed)

    #reshape
    trainX, trainY, testX, testY = reshape(train, test, look_back)

    #data augmentation
    if regularization_technique == 'data_augmentation':
        trainX, trainY = apply_data_augmentation(trainX, trainY, **reg_params)
        print(f"Applied Data Augmentation with parameters: {reg_params}")

    #build model
    input_layer = Input(shape=(trainX.shape[1], trainX.shape[2]))

    #regularization
    if regularization_technique == 'l2_regularization':
        if activation == 'leaky_relu':
            x = LSTM(16, kernel_regularizer=l2(reg_params.get('l2_lambda', 0.01)))(input_layer)
            x = LeakyReLU(alpha=0.01)(x)
        else:
            x = LSTM(16, activation=activation, kernel_regularizer=l2(reg_params.get('l2_lambda', 0.01)))(input_layer)
        print(f"Applied L2 Regularization with lambda: {reg_params.get('l2_lambda', 0.01)}")

    elif regularization_technique == 'recurrent_dropout':
        if activation == 'leaky_relu':
            x = LSTM(16, recurrent_dropout=reg_params.get('recurrent_dropout_rate', 0.2))(input_layer)
            x = LeakyReLU(alpha=0.01)(x)
        else:
            x = LSTM(16, activation=activation, recurrent_dropout=reg_params.get('recurrent_dropout_rate', 0.2))(input_layer)
        print(f"Applied Recurrent Dropout with rate: {reg_params.get('recurrent_dropout_rate', 0.2)}")

    else:
        if activation == 'leaky_relu':
            x = LSTM(16)(input_layer)
            x = LeakyReLU(alpha=0.01)(x)
        else:
            x = LSTM(16, activation=activation)(input_layer)

    if regularization_technique == 'dropout':
        x = Dropout(0.1)(x)
        print("Applied Dropout with rate: 0.1 after LSTM")

    elif regularization_technique == 'batch_normalization':
        x = BatchNormalization()(x)
        print("Applied Batch Normalization after LSTM")

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

    if regularization_technique == 'gradient_clipping':
        if optimizer == 'adam':
            opt = Adam(learning_rate=learning_rate, clipnorm=reg_params.get('clip_norm', 1.0))
        elif optimizer == 'sgd':
            opt = SGD(learning_rate=learning_rate, clipnorm=reg_params.get('clip_norm', 1.0))
        elif optimizer == 'rmsprop':
            opt = RMSprop(learning_rate=learning_rate, clipnorm=reg_params.get('clip_norm', 1.0))
        print(f"Applied Gradient Clipping with norm: {reg_params.get('clip_norm', 1.0)}")

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

    model_instance.compile(loss=loss_fn, optimizer=opt)

    #callbacks
    callbacks = []

    if regularization_technique == 'early_stopping':
        early_stop = EarlyStopping(
            monitor='loss',
            patience=reg_params.get('patience', 10),
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stop)
        print(f"Applied Early Stopping with patience: {reg_params.get('patience', 10)}")

    if regularization_technique is None:
        print("No regularization applied")

    #fit model
    model_instance.fit(
        trainX, trainY,
        epochs=nepochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=callbacks if callbacks else None
    )

    #predict, forecast and plot
    testScore = predict_forecast_plot(
        scaler, data, train, test, trainX, trainY, testX, testY,
        nepochs, look_back, horizon, plot_predictions, model_instance
    )

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

#@title Enhanced Experimentation Functions with Seeds

def run_regularization_experiments_with_seeds(configurations, symbol='AAPL', seeds=BASE_SEEDS, 
                                            look_back=30, horizon=7, experiment_name="Regularization"):
    all_results = []
    
    for run_idx, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"RUN {run_idx + 1}/{len(seeds)} - SEED: {seed}")
        print(f"{'='*60}")
        
        for config_idx, config in enumerate(configurations):
            print(f"\nTesting {config['name']} - Seed {seed}")
            
            try:
                # Get data for each configuration
                scaler, data, train, test = read_prepare_data_with_regularization(symbol)
                
                score = model_with_regularization(
                    scaler, data, train, test,
                    look_back=look_back,
                    nepochs=config['nepochs'],
                    horizon=horizon,
                    plot_predictions=False,
                    seed=seed,
                    batch_size=1,
                    learning_rate=0.001,
                    optimizer='adam',
                    activation='relu',
                    loss='mse',
                    regularization_technique=config['technique'],
                    **config['params']
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

    #aggregate results by configuration
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

    #find best configurations
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

    #check for consensus
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

    #rank all configurations by composite score
    print(f"\n{'='*80}")
    print("RANKING BY COMPOSITE SCORE:")
    print(f"{'='*80}")
    sorted_configs = sorted(results.items(), key=lambda x: x[1]['composite_score'])
    for rank, (config_name, data_dict) in enumerate(sorted_configs, 1):
        print(f"{rank:2d}. {config_name}: "
              f"Score={data_dict['composite_score']:.3f}, "
              f"RMSE={data_dict['mean_rmse']:.3f}±{data_dict['std_rmse']:.3f}, "
              f"CV%={data_dict['cv_percent']:.1f}%")

#@title Configuration Definitions

regularization_configurations = [
    {
        'name': 'Dropout (0.1)',
        'technique': 'dropout',
        'params': {},
        'nepochs': 50
    },
    {
        'name': 'Early Stopping',
        'technique': 'early_stopping',
        'params': {'patience': 10},
        'nepochs': 500
    },
    {
        'name': 'L2 Regularization (0.001)',
        'technique': 'l2_regularization',
        'params': {'l2_lambda': 0.001},
        'nepochs': 50
    },
    {
        'name': 'Batch Normalization',
        'technique': 'batch_normalization',
        'params': {},
        'nepochs': 50
    },
    {
        'name': 'Recurrent Dropout (0.1)',
        'technique': 'recurrent_dropout',
        'params': {'recurrent_dropout_rate': 0.1},
        'nepochs': 50
    },
    {
        'name': 'Gradient Clipping',
        'technique': 'gradient_clipping',
        'params': {'clip_norm': 1.0},
        'nepochs': 50
    },
    {
        'name': 'Data Augmentation (Gaussian Noise 0.01)',
        'technique': 'data_augmentation',
        'params': {'noise_factor': 0.01},
        'nepochs': 50
    }
]

#@title Run All Experiments

if __name__ == "__main__":
    print(f"\nStarting robust regularization technique evaluation with {len(BASE_SEEDS)} seeds...")
    
    #Regularization Technique Experiments
    print(f"\n{'='*80}")
    print("STARTING REGULARIZATION TECHNIQUE EXPERIMENTS")
    print(f"{'='*80}")
    
    regularization_results = run_regularization_experiments_with_seeds(
        configurations=regularization_configurations,
        symbol='AAPL',
        seeds=BASE_SEEDS,
        experiment_name="Regularization Techniques"
    )
    
    print_detailed_results(regularization_results, "Regularization Techniques", BASE_SEEDS)
    
    #Final Summary
    print(f"\n{'='*100}")
    print("FINAL SUMMARY OF REGULARIZATION TECHNIQUE EXPERIMENTS")
    print(f"{'='*100}")
    
    print("\nBEST REGULARIZATION TECHNIQUE:")
    print("-" * 60)
    
    if regularization_results:
        best_config = min(regularization_results.items(), key=lambda x: x[1]['composite_score'])
        print(f"★ RECOMMENDED REGULARIZATION TECHNIQUE: {best_config[0]}")
        print(f"  Composite Score: {best_config[1]['composite_score']:.3f}")
        print(f"  Mean RMSE: {best_config[1]['mean_rmse']:.3f} ± {best_config[1]['std_rmse']:.3f}")
        print(f"  CV%: {best_config[1]['cv_percent']:.1f}%")

    end_time = time.time()
    minutes = int((end_time-start_time)/60)
    hours = minutes//60
    remaining_minutes = minutes%60
    current_file = os.path.basename(__file__)
    print(f"\n{current_file} took {hours}h{remaining_minutes}m to complete")