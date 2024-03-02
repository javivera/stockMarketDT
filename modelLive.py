import joblib
from mainLib import addFeauturesToDf
# from graphLib import strat_graph
from dataRetrieveLib import get_binance_futures_klines
import pandas as pd
import time
import csv
import os
from datetime import datetime
import subprocess
import sys
# import shutil
import plotly.graph_objects as go
# from tastyAPI import get_data
from datetime import datetime
from binanceAPI import buy_order, sell_order,cancel_orders,get_active_orders
from mainLib import addFeauturesToDf
# from modelGenerator import generateModels
from plotly.subplots import make_subplots

def modelLive(symbol:str,interval:str,live=0,model_number=0):
    current_date = datetime.now().date().strftime("%Y-%m-%d")
    model_number = f'model-{model_number}'
    print('Creating folders')
    treeCreator(symbol,interval)
    csv_file_path = f'live/crypto/{symbol}/{interval}/{current_date}.csv'
    # if os.path.exists(csv_file_path):
        # os.remove(csv_file_path)

    if interval == '1m':
        minutes = [j for j in range(0,60)] # todos los minutos posibles , se puede mejorar 
    elif interval == '2m':
        minutes = [j for j in range(0,60,2)]
    elif interval == '5m':
        minutes = [4,9,14,19,24,29,34,39,44,49,54,59]
    elif interval == '15m':
        minutes = [14,29,44,59]
    else:
        minutes = [59]
        
    trade_status = 0

    print('Model is live')
    trade_status = 0
    while True:
        current_time = datetime.now().time()
        time.sleep(5)
        # if True:
        if current_time.minute in minutes and (current_time.second < 59 and current_time.second > 45):
            if True:
                # print(current_time.second)
                subprocess.call("clear")
                if live == 1:
                    print(f'Model is ONLINE for {interval}')
                else:
                    print(f'Model is OFFLINE for {interval}')

                print('Retrieving data for {}...'.format(symbol))
                df = get_binance_futures_klines(symbol,interval,100)
                print(df.iloc[-1])
                # df = data_retrieve_yahoo(symbol)
                # start_time = datetime.now() - timedelta(seconds=120)  # 1 month ago
                # df = get_data(symbol,interval,start_time)

                print('Reading indicators and preparing model')
                with open(f'offline/{symbol}/{interval}/{model_number}/{model_number}-indicators.csv', 'r') as file:
                    csv_reader = csv.reader(file)
                    predictors = next(csv_reader, None)

                df_prep = df.copy()
                df_prep = addFeauturesToDf(df_prep,1)
                df_prep = df_prep[predictors]
                print(f'offline/{symbol}/{interval}/{model_number}/{model_number}-random_forest.joblib')
                model = joblib.load(f"offline/{symbol}/{interval}/{model_number}/{model_number}-random_forest.joblib")
                df_prep['Pred'] = pd.Series(model.predict(df_prep[-1:]),index=df_prep[-1:].index)
                df_prep['Pred'] = df_prep['Pred'].apply(lambda x: -1 if x == 0 else x)
                # df_prep = df_prep.iloc[-1:]
                # strat_graph(df,df_prep,symbol,interval,modelName,dir)
                df['Predicted'] = df_prep['Pred'].iloc[-1:]
                price = round(float(df['Close'].iloc[-1]),2)
                last_predict = df['Predicted'].iloc[-1]

                df['Time of operation'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(trade_status)
                if int(live) == 1:
                    if last_predict == 1 and trade_status == 0:
                        cancel_orders(symbol)
                        print(f'Placing Order at {price}')
                        buy_order(symbol,price,0.003)
                        trade_status = 1
                        df['Time of operation'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print(trade_status)
                    elif last_predict == 1 and trade_status == -1:
                        cancel_orders(symbol)
                        print(f'Placing Order at {price}')
                        buy_order(symbol,price,0.006)
                        trade_status = 1
                        df['Time of operation'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    elif last_predict == -1 and trade_status == 0:
                        cancel_orders(symbol)
                        print(f'Placing Order at {price}')
                        sell_order(symbol,price,0.003)
                        trade_status = -1
                        df['Time of operation'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    elif last_predict == -1 and trade_status == 1:
                        cancel_orders(symbol)
                        print(f'Placing Order at {price}')
                        sell_order(symbol,price,0.006)
                        trade_status = -1
                        df['Time of operation'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                df['return_model'] = 0
                df['earnings_model'] = 0
                append_df_to_csv(df[['Time of operation','Close','Predicted','return_model','earnings_model']].iloc[-1:], csv_file_path)
                df = earningsCalculator(symbol,interval,csv_file_path)

                print('--------------------------')
                print(df[['Time of operation','Close','Predicted','return_model','earnings_model']].reset_index(drop=True).tail(3))
                print('Datapoint added')
                print('--------------------------')
                print('Current Balance: {}'.format(df['earnings_model'].iloc[-2:-1]))
                print('--------------------------')
                print('Model is waiting')
                time.sleep(240)

def treeCreator(symbol,interval):

    if not os.path.exists("live"):
        os.mkdir("live")
    if not os.path.exists("live/crypto"):
        os.mkdir("live/crypto")

    if not os.path.exists(f"live/crypto/{symbol}"):
        os.mkdir(f"live/crypto/{symbol}")

    if not os.path.exists(f"live/crypto/{symbol}/{interval}"):
        os.mkdir(f"live/crypto/{symbol}/{interval}")

    if not os.path.exists(f"live/crypto/{symbol}/{interval}/"):
        os.mkdir(f"live/crypto/{symbol}/{interval}/")

    # if not os.path.exists(f"live/crypto/{symbol}/{interval}/{model}/graphs"):
        # os.mkdir(f"live/crypto/{symbol}/{interval}/{model}/graphs")

    if not os.path.exists(f"live/crypto/{symbol}/{interval}/data"):
        os.mkdir(f"live/crypto/{symbol}/{interval}/data")

def append_df_to_csv(new_df, csv_file_path):
    # 1. Load the existing CSV file into a DataFrame
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        # If the file doesn't exist, create a new file with the new DataFrame
        new_df.to_csv(csv_file_path,index=False)
        return
    # 2. Append the new DataFrame to the existing one
    combined_df = pd.concat([df,new_df])
    # 3. Write the combined DataFrame back to the CSV file
    combined_df.to_csv(csv_file_path, index=False)

def earningsCalculator(symbol,interval,csv_file_path):
    df = pd.read_csv(csv_file_path)
    current_date = datetime.now().date().strftime("%Y-%m-%d")
    # print(df)
    # df.set_index('Time of operation', inplace=True)
    df['next_close'] = df['Close'].shift(-1)
    df['return_baseline'] = df['next_close'] - df['Close'] 
    df['return_model'] = df['return_baseline'] * df['Predicted'] 
    df['earnings_baseline'] = df['return_baseline'].cumsum()
    df['earnings_model'] = df['return_model'].cumsum()
    # df.set_index('Datetime', inplace=True)
    df['next_pred'] = df['Predicted'].shift(-1)
    df['sum_of_preds'] = ((df['Predicted'] + df['next_pred']).shift(1)).apply(lambda x: 1 if x == 0 else 0)


    df['fees'] = 0.00012 * (df['Close'] + df['next_close']) * df['sum_of_preds']
    df['adjusted_earnings'] = df['earnings_model'] - df['fees']

    df[['Time of operation','Close','Predicted','return_model','earnings_model']].to_csv(csv_file_path,index=False)
    fig = make_subplots(rows=2, cols=1, subplot_titles=[
    'Returns',
    'Histogram',
    ],vertical_spacing=0.1)   

    fig.add_trace(go.Histogram(x=df['return_model'], nbinsx=100, marker_color='#008080',name='Histogram'),row=2, col=1)

    fig.add_trace(go.Scatter(x=df['Time of operation'], y=df['earnings_baseline'], mode='lines', name='Baseline Returns', yaxis='y1'),row=1,col=1)
    # fig.add_trace(go.Scatter(x=df.index, y=df['earnings_model'], mode='lines', name='Model Returns', yaxis='y1'))
    fig.add_trace(go.Scatter(x=df['Time of operation'], y=df['earnings_model'], mode='lines', name='Model Returns', yaxis='y1'),row=1,col=1)

    # fig.update_layout(title=f'{symbol} - {interval} - Earnings')

    fig.update_layout(title=f'{symbol} - {interval} data')
    fig.write_html(f'live/crypto/{symbol}/{interval}/{current_date}-earnings_chart.html', auto_open=False)
    try:
        return df
    except:
        raise Exception

if __name__ == "__main__":
    arguments = sys.argv[1:]  # List of command-line arguments
    modelLive(arguments[0],arguments[1],int(arguments[2]))    





