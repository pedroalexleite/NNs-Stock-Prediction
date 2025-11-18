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
    data = pd.read_csv('/Users/pedroalexleite/Desktop/NNs-Stock-Prediction/Data/dataset4.csv')
    train = pd.read_csv('/Users/pedroalexleite/Desktop/NNs-Stock-Prediction/Data/train.csv')
    test = pd.read_csv('/Users/pedroalexleite/Desktop/NNs-Stock-Prediction/Data/test.csv')
    
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

#@title Models

def list_models(look_back, trainX, seed=None):
    if seed is not None:
        set_seeds(seed)
    
    layers = [1, 2, 3]
    neurons = [8, 16, 32, 64, 128]
    models = []
    configurations = []
    
    for num_layers in layers:
        for num_neurons in neurons:
            if seed is not None:
                set_seeds(seed)
            
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
                "neurons": num_neurons
            })
    
    return models, configurations

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

look_back = 30
trainX, trainY, testX, testY = reshape(train, test, look_back)

initial_models, configurations = list_models(look_back, trainX, seed=BASE_SEEDS[0])
print("Configurations:")
for idx, config in enumerate(configurations, start=1):
    print(f"Model {idx}: Layers={config['layers']}, Neurons={config['neurons']}")

#@title Train and Predict (Find the Optimal Model)

def optimal_model(models, configurations, data, train, test, look_back=30, nepochs=50, horizon=7, plot_predictions=False, seed=None):
    results = []
    for idx, (model, config) in enumerate(zip(models, configurations), start=1):
        print(f"Training Model {idx}: Layers={config['layers']}, Neurons={config['neurons']}")
        
        #use the provided seed for this model training
        if seed is not None:
            set_seeds(seed)
        
        #reshape
        trainX, trainY, testX, testY = reshape(train, test, look_back)
        
        #fit
        history = model.fit(trainX, trainY, epochs=nepochs, batch_size=1, verbose=1)
        
        #predict, foecast and plot
        testScore = predict_forecast_plot(data, train, test, trainX, trainY, testX, testY, nepochs, look_back, horizon, plot_predictions, model)
        
        #append results
        results.append({
            "model_index": idx,
            "configuration": config,
            "test_score": testScore,
            "seed": seed
        })

    return results

def calculate_average_results(configurations, data, train, test, look_back=30, nepochs=50, horizon=7, seeds=BASE_SEEDS):
    all_results = []
    
    for run_idx, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"RUN {run_idx + 1}/{len(seeds)} - SEED: {seed}")
        print(f"{'='*60}")
        
        #create fresh models for this seed
        run_models, run_configurations = list_models(look_back, 
                                                    reshape(train, test, look_back)[0], 
                                                    seed=seed)
        
        #train all models with this seed
        results = optimal_model(run_models, run_configurations, data, train, test, 
                              look_back, nepochs, horizon, plot_predictions=False, seed=seed)
        all_results.extend(results)

    #aggregate results by model configuration
    aggregated_results = {}
    for result in all_results:
        model_idx = result["model_index"]
        if model_idx not in aggregated_results:
            aggregated_results[model_idx] = {
                "configuration": result["configuration"],
                "test_scores": [],
                "seeds_used": []
            }
        aggregated_results[model_idx]["test_scores"].append(result["test_score"])
        aggregated_results[model_idx]["seeds_used"].append(result["seed"])
    
    #calculate statistics for each model
    for model_idx, data_dict in aggregated_results.items():
        scores = data_dict["test_scores"]
        data_dict["mean_rmse"] = np.mean(scores)
        data_dict["std_rmse"] = np.std(scores)
        data_dict["min_rmse"] = np.min(scores)
        data_dict["max_rmse"] = np.max(scores)
        data_dict["cv_percent"] = (data_dict["std_rmse"] / data_dict["mean_rmse"]) * 100

    #calculate composite scores for each model
    for model_idx, data_dict in aggregated_results.items():
        data_dict["composite_score"] = calculate_composite_score(data_dict, aggregated_results)

    return aggregated_results

#execute with multiple seeds for robust comparison
print(f"\nStarting robust evaluation with {len(BASE_SEEDS)} seeds...")
average_results = calculate_average_results(configurations, data, train, test, 
                                          look_back=30, nepochs=50, horizon=7, 
                                          seeds=BASE_SEEDS)

#display detailed results
print(f"\n{'='*80}")
print("ROBUST MODEL COMPARISON RESULTS")
print(f"{'='*80}")
print(f"Each model tested with {len(BASE_SEEDS)} different seeds: {BASE_SEEDS}")
print()

for model_idx, results in average_results.items():
    config = results["configuration"]
    print(f"Model {model_idx} - {config['layers']}L, {config['neurons']}N:")
    print(f"  Individual RMSEs: {[round(score, 3) for score in results['test_scores']]}")
    print(f"  Mean RMSE: {results['mean_rmse']:.3f}")
    print(f"  Std RMSE:  {results['std_rmse']:.3f}")
    print(f"  Range:     {results['min_rmse']:.3f} - {results['max_rmse']:.3f}")
    print(f"  CV%:       {results['cv_percent']:.1f}%")
    print(f"  Composite Score: {results['composite_score']:.3f}")
    print()

#find best model based on different criteria
best_mean_idx, best_mean_results = min(average_results.items(), key=lambda x: x[1]['mean_rmse'])
best_cv_idx, best_cv_results = min(average_results.items(), key=lambda x: x[1]['cv_percent'])
best_composite_idx, best_composite_results = min(average_results.items(), key=lambda x: x[1]['composite_score'])

print(f"{'='*80}")
print("BEST MODELS BY DIFFERENT CRITERIA:")
print(f"{'='*80}")

print("1. BEST MEAN PERFORMANCE:")
config = best_mean_results['configuration']
print(f"   Model {best_mean_idx}: {config['layers']}L, {config['neurons']}N")
print(f"   Mean RMSE: {best_mean_results['mean_rmse']:.3f} ± {best_mean_results['std_rmse']:.3f}")
print(f"   CV%: {best_mean_results['cv_percent']:.1f}%")
print(f"   Composite Score: {best_mean_results['composite_score']:.3f}")

print("\n2. MOST CONSISTENT (Lowest CV%):")
config = best_cv_results['configuration']
print(f"   Model {best_cv_idx}: {config['layers']}L, {config['neurons']}N")
print(f"   Mean RMSE: {best_cv_results['mean_rmse']:.3f} ± {best_cv_results['std_rmse']:.3f}")
print(f"   CV%: {best_cv_results['cv_percent']:.1f}%")
print(f"   Composite Score: {best_cv_results['composite_score']:.3f}")

print("\n3. BEST COMPOSITE SCORE (70% mean, 20% std, 10% cv):")
config = best_composite_results['configuration']
print(f"   Model {best_composite_idx}: {config['layers']}L, {config['neurons']}N")
print(f"   Mean RMSE: {best_composite_results['mean_rmse']:.3f} ± {best_composite_results['std_rmse']:.3f}")
print(f"   CV%: {best_composite_results['cv_percent']:.1f}%")
print(f"   Composite Score: {best_composite_results['composite_score']:.3f}")

#check for consensus
winners = [best_mean_idx, best_cv_idx, best_composite_idx]
winner_counts = {idx: winners.count(idx) for idx in set(winners)}
consensus_winner = max(winner_counts, key=winner_counts.get)

print(f"\n{'='*80}")
print("FINAL RECOMMENDATION:")
print(f"{'='*80}")

if winner_counts[consensus_winner] >= 2:
    print(f"★ RECOMMENDED MODEL: Model {consensus_winner}")
    print(f"  This model performs well across {winner_counts[consensus_winner]} different criteria.")
    config = average_results[consensus_winner]['configuration']
    print(f"  Configuration: {config['layers']} layers, {config['neurons']} neurons")
    print(f"  Mean RMSE: {average_results[consensus_winner]['mean_rmse']:.3f}")
    print(f"  CV%: {average_results[consensus_winner]['cv_percent']:.1f}%")
    print(f"  Composite Score: {average_results[consensus_winner]['composite_score']:.3f}")
else:
    print("★ NO CLEAR CONSENSUS - Consider your priorities:")
    print(f"  - If you prioritize average performance: Model {best_mean_idx}")
    print(f"  - If you prioritize consistency: Model {best_cv_idx}")
    print(f"  - For balanced approach: Model {best_composite_idx}")

#rank all models by composite score
print(f"\n{'='*80}")
print("RANKING BY COMPOSITE SCORE:")
print(f"{'='*80}")
sorted_models = sorted(average_results.items(), key=lambda x: x[1]['composite_score'])
for rank, (model_idx, results) in enumerate(sorted_models, 1):
    config = results['configuration']
    print(f"{rank:2d}. Model {model_idx} ({config['layers']}L, {config['neurons']}N): "
          f"Score={results['composite_score']:.3f}, "
          f"RMSE={results['mean_rmse']:.3f}±{results['std_rmse']:.3f}, "
          f"CV%={results['cv_percent']:.1f}%")

end_time = time.time()
minutes = int((end_time-start_time)/60)
hours = minutes//60
remaining_minutes = minutes%60
current_file = os.path.basename(__file__)
print(f"{current_file} took {hours}h{remaining_minutes}")