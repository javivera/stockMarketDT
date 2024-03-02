import pandas as pd
# import numpy as np
import talib
import ta
#---------------------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
# import cuml
# from cuml.ensemble import RandomForestClassifier
# from cuml.metrics import accuracy_score
# --------------------------------------
import os
import itertools
import concurrent.futures
from tqdm import tqdm
# import subprocess
import joblib
import csv
from graphLib import returns_graph,confusion
import random


def permutator(predictors:list[str]):
    predictors_permu = []
    for j in range(1,len(predictors)):
        for comb in itertools.combinations(predictors, j):
            predictors_permu.append(list(comb))
    # for pred in predictors:
    predictors_permu.append(predictors)
    rand_pred_permu = random.sample(predictors_permu, 75)
    print(f'Trying {len(rand_pred_permu)} indicators')
    return rand_pred_permu

def addFeauturesToDf(df,drop=0):
    # df.astype({col: int for col in df.columns[1:]})
   
    df['Tomorrow'] = df['Close'].shift(-1)
    df['price_diff'] = (df['Tomorrow'] - df['Close'])
    # df['Bins'] = pd.qcut(df['price_diff'], q=4, labels=[0, 1, 2,3])
    df['Target'] = (df['Tomorrow'] > df['Close']).astype(int)
    slowk, _ = talib.STOCH(
            df['High'].values,
            df['Low'].values,
            df['Close'].values,
            fastk_period=14,
            slowk_period=3,
            slowk_matype=0,  # Simple moving average for %K
            slowd_period=3,
            slowd_matype=0  # Simple moving average for %D
        )

    df['STOCH_K'] = slowk
    
    williams_r = talib.WILLR(
            df['High'].values,
            df['Low'].values,
            df['Close'].values,
            timeperiod=14
        )

    df['WILL_R'] = williams_r
    df['RSI'] = talib.RSI(df['Close'].values, timeperiod=14)
    df['SMA'] = talib.SMA(df['Close'].values, timeperiod=14)

    df['EMA'] = talib.EMA(df['Close'], timeperiod=14)
    df['upper_band'], df['middle_band'], df['lower_band'] = talib.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2)


    df['SAR'] = talib.SAR(df['High'], df['Low'], acceleration=0.02, maximum=0.2)

    macd, signal, _ = talib.MACD(
            df['Close'].values,
            fastperiod=12,
            slowperiod=26,
            signalperiod=3
        )

    df['MACD'] = macd
    df['MACD_Signal'] = signal

    df['OBV'] = talib.OBV(df['Close'].astype(float).values, df['Volume'].astype(float).values)

    df['PROC'] = talib.ROCP(df['Close'].values, timeperiod=7)

    df['AOI'] = ta.momentum.AwesomeOscillatorIndicator(df['High'], df['Low']).awesome_oscillator()
    df['CCI'] = ta.trend.CCIIndicator(df['High'], df['Low'], df['Close']).cci()
    df['ROC'] = ta.momentum.ROCIndicator(df['Close'], 10).roc()
    df['TRIX'] = ta.trend.TRIXIndicator(df['Close'], 10).trix()
    del df['Tomorrow']
    if drop == 0:
        df.dropna(inplace=True)

    return df

def model_train(df,predictors,step):
    df2 = df.copy()
    df2 = df2.iloc[-step:]
    y_df2 = df2['Target']
    del df2['Target']

    df = df[:-step]
    X_train, X_test, y_train, y_test = train_test_split(df, df["Target"], test_size=0.7, random_state=42)
    # X_train = df.iloc[-1200:-step]
    # y_train = df["Target"].iloc[-1200:-step]
    # X_test = df.iloc[-step:]
    # y_test = df["Target"].iloc[-step:]
    # del X_train['Target']
    # del X_test['Target']
    params = {
        'n_estimators': [10, 20, 50, 100,200],
        'min_samples_split': [2, 5, 10],
    }

    grid_search = GridSearchCV(RandomForestClassifier(random_state=41), params, cv=5)
    grid_search.fit(X_train[predictors], y_train)
    best_params = grid_search.best_params_

    model = RandomForestClassifier(**best_params,random_state=41)
    # model = RandomForestClassifier(**best_params,n_streams=1)

    model.fit(X_train[predictors], y_train)

    _,f1 = model_test(df2,y_df2,model,predictors)
    print(f'F1 on test suite was {f1}')
    return df2, y_df2, model

def model_test(X_test,y_test,model,predictors):
    preds = model.predict(X_test[predictors])
    preds = pd.Series(preds, index=X_test.index)
    f1 = f1_score(y_test, preds)
    # print(precision_score(y_test, preds))
    # print(preds.value_counts())
    # y_test.value_counts() / len(y_test)
    return preds,f1

def model_data(X_test,y_test,preds):
    df_preds = X_test.copy()

    df_preds['target'] = y_test
    df_preds['Pred'] = preds
    df_preds['correct'] = df_preds['target'] == df_preds['Pred']
    df_preds['Pred'] = df_preds['Pred'].apply(lambda x: 1 if x == 1 else -1)
    df_preds['next_pred'] = df_preds['Pred'].shift(-1)
    df_preds['sum_of_preds'] = ((df_preds['Pred'] + df_preds['next_pred']).shift(1)).apply(lambda x: 1 if x == 0 else 0)

    # df_preds['return'] = (df_preds['Tomorrow']) - (df_preds['Close'])

    # df_preds['fees'] = (0.01) * df_preds['sum_of_preds']
    df_preds['next_close'] = df_preds['Close'].shift(-1)
    df_preds['fees'] = 0.00012 * (df_preds['Close'] + df_preds['next_close']) * df_preds['sum_of_preds']

    df_preds['return'] = df_preds['next_close'] - df_preds['Close']
    

    # df_preds['return'] = np.log(df_preds['Close'].shift(-1) / df_preds['Close'])
    df_preds['return_strat_without_fees'] = (df_preds['return'] * df_preds['Pred'])
    df_preds['return_strat'] =  (df_preds['return'] * df_preds['Pred']) - df_preds['fees']


    df_preds['Baseline_Strat'] = df_preds['return'].cumsum()
    df_preds['Model_Strat'] = df_preds['return_strat'].cumsum()
    # del df_preds['fees']
    del df_preds['return_strat_without_fees']
    # df_preds['Model_Strat_fees'] = df_preds['return_strat_fees'].cumsum()
    # df_preds[['Close','return', 'Baseline_Strat','Log_return_strat' , 'Model_Strat']]
    # print('Earningsg with fees')
    # strat_earnings = df_preds['Model_strat'].iloc[-2]
    risk_free_rate = 0.04
    
    df_preds['ExcessReturn'] = df_preds['return_strat'] - risk_free_rate
    # rolling_window = 3
    # df_preds['sharpe_ratio'] = df_preds['ExcessReturn'].rolling(window=rolling_window).mean() / df_preds['ExcessReturn'].rolling(window=rolling_window).std()
    df_preds['Variance'] = df_preds['return_strat'].rolling(2).var()
    del df_preds['ExcessReturn']

    return df_preds

def trainAndTestModels(df,predictor,step):
    # Recieves a df and predictors and returns a
    # info_df a df with metrics
    # df_preds the predictions made
    # predictor the predictor used
    # model fitted to that df based on the predictors and a df with information about accuray
    data = {}
    accuracy = []
    strat_earn = []
    strat_used = []
    #raining the model with given predictors
    X_test, y_test, model = model_train(df,predictor,step)
    #Using predict and saving accuracy
    # preds,acc = model_test(X_test,y_test, model, predictor)
    preds = pd.Series(model.predict(X_test[predictor]), index=X_test.index)
    acc = precision_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    # recall = classification_report(y_test, model.predict(X_test[predictor]))

    #Calculating returns of strategy
    df_preds = model_data(X_test,y_test, preds)
    # Creating a datafram with accuracy and earnings
    # strat_earn.append(model_earnings)
    accuracy.append(acc)
    strat_used.append("+".join(predictor))
    data['acc'] = accuracy
    data['strat_earn'] = strat_earn
    
    # subprocess.call("clear")
    # print(f'{predictor} earnings: {data["strat_earn"]}')

    # info_df = pd.DataFrame(data,index=strat_used).sort_values('strat_earn',ascending=False)
    
    # print(pd.DataFrame(data,index=start_used).sort_values('start_earn',ascending=False))
    # return model_earnings,df_preds,y_test,model,predictor
    return f1,df_preds,y_test,model,predictor,acc

def multi_processor(df,predictors,step):
    print('Starting multiprocessing')
    # start_time = time.time()
    # print('Trying {} strategies'.format(len(predictors)))    
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #
    #     futures = [executor.submit(trainAndTestModels,df,input_val) for input_val in predictors]
    #
    #     concurrent.futures.wait(futures)
    #
    #     results = [f.result() for f in futures]
    # 
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    #
    # print(f"Total time taken for the multiprocessor: {elapsed_time} seconds")


    with tqdm(total=len(predictors)) as pbar:
        def update_progress_bar(future):
        # Update the progress bar when a task is done
            pbar.update(1)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(trainAndTestModels, df,input_val,step) for input_val in predictors]

        # Attach the update_progress_bar function to each future
            for future in concurrent.futures.as_completed(futures):
                future.add_done_callback(update_progress_bar)

        # Wait for all tasks to complete
            concurrent.futures.wait(futures)

    # Retrieve results
    results = [f.result() for f in futures]


    return results

def modelGenerator(df:pd.DataFrame,symbol:str,predictors:list[str],step=30):
    # Generates a the best model given an amount of permutations of the predictors 

    # for symbol in symbols:
    # df = data_retrieve_binance(symbol,interval)
        # df = data_retrieve(symbol,interval)
    print(f'{symbol} Df retrieved')
    print('Adding features to Df')
    df = addFeauturesToDf(df)
    print('Features added')
    results = multi_processor(df,permutator(predictors),step)
    
    return results,df

    # This selects between all the predictors which ones worked best
    # result_df = pd.concat([r[0] for r in results]).sort_values('strat_earn',ascending=False)
    # main_indices = [result_df.index[0],result_df.index[1]]
    # Now using that best predictor we generate graphics output the dataframe with the results and save the model

def bestResultPicker(results,symbol,interval):  
    print('Generating graphics')
    results = sorted(results, key=lambda x: x[0],reverse=True)
    best_df = results[0][1]
    # for j in results:
        # print(j[0])
    best_y_test = results[0][2]
    best_model = results[0][3]
    print(f'Best F1 was {results[0][0]}')
    
    print('Generating file structure')
    print("-------------------------------------------------")

    free_index = fileStructureCreator(symbol,interval)
    model = f'model-{free_index}'

    confusion(best_y_test,best_df['Pred'],symbol,interval,model)
    returns_graph(results[:4],symbol,interval,model)
    print('Best strategy graph')
    print('Outputting best predictor data')
    best_df.to_csv(f"offline/{symbol}/{interval}/{model}/data/{model}.csv")

    print("Dumping Model and indicators")
    joblib.dump(best_model, f"offline/{symbol}/{interval}/{model}/{model}-random_forest.joblib")

    if os.path.exists(f"offline/{symbol}/{interval}/{model}/{model}-indicators.csv"):
        os.remove(f"offline/{symbol}/{interval}/{model}/{model}-indicators.csv")

    for j in range(len(results)):
        with open(f"offline/{symbol}/{interval}/{model}/{model}-indicators.csv", 'a',newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(results[j][4])

def fileStructureCreator(symbol,interval):
    if not os.path.exists("offline"):
        os.mkdir("offline")
    if not os.path.exists(f"offline/{symbol}"):
        os.mkdir(f"offline/{symbol}")
    if not os.path.exists(f"offline/{symbol}/{interval}"):
        os.mkdir(f"offline/{symbol}/{interval}")

    file_list = os.listdir(f'offline/{symbol}/{interval}/')
    if not file_list:
        free_index = 0

    else:
        free_index = 0
        while free_index < 100:
            if free_index not in [int(file.split('-')[1]) for file in file_list]:
                break
            else:
                free_index += 1

    model = f'model-{free_index}'
    if not os.path.exists(f"offline/{symbol}/{interval}/{model}"):
        os.mkdir(f"offline/{symbol}/{interval}/{model}")
    if not os.path.exists(f"offline/{symbol}/{interval}/{model}/data"):
        os.mkdir(f"offline/{symbol}/{interval}/{model}/data")

    return free_index


