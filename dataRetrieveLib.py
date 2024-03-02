from datetime import datetime
import yfinance as yf
from binance.client import Client
import pandas as pd
import requests
import json
import pandas as pd
import hashlib
import hmac
import time

def data_retrieve_binance(ticker,interval,end_date=0):
    
    client = Client('kLIAV7kiJBeWgPCpDHtiqTkBsBMhtC6J8vCac18AS4aq88l5GCqt1ujx7jW508MW', '27sUNaPVVfoXVSRRMLqvqutHQXUXn3lAFERTppVzfvGgjceyP3k2lKZ6VbPxKdnN')
    if end_date == 1:
        if interval == '5m':
            klines = client.get_historical_klines(ticker, Client.KLINE_INTERVAL_5MINUTE, "8 day ago UTC")
        if interval == '15m':
            klines = client.get_historical_klines(ticker, Client.KLINE_INTERVAL_15MINUTE, "12 day ago UTC")
        if interval == '1h':
            klines = client.get_historical_klines(ticker, Client.KLINE_INTERVAL_1HOUR, "50 day ago UTC")
        if interval == '1m':
            klines = client.get_historical_klines(ticker, Client.KLINE_INTERVAL_1MINUTE, "6 day ago UTC")
        if interval == '1d':
            klines = client.get_historical_klines(ticker, Client.KLINE_INTERVAL_1DAY, "90 day ago UTC")
    
    else:
        if interval == '5m':
            klines = client.get_historical_klines(ticker, Client.KLINE_INTERVAL_5MINUTE, "10 day ago UTC")
        if interval == '15m':
            klines = client.get_historical_klines(ticker, Client.KLINE_INTERVAL_15MINUTE, "10 day ago UTC")
        if interval == '1h':
            klines = client.get_historical_klines(ticker, Client.KLINE_INTERVAL_1HOUR, "10 day ago UTC")
        if interval == '1m':
            klines = client.get_historical_klines(ticker, Client.KLINE_INTERVAL_1MINUTE, "10 day ago UTC")
        if interval == '1d':
            klines = client.get_historical_klines(ticker, Client.KLINE_INTERVAL_1DAY, "10 day ago UTC")

    df = pd.DataFrame(klines,columns=['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']).astype(float)
    df = df.iloc[:,:6]
    df['Datetime'] = df['Datetime'].apply(lambda x: convertTimestamps(x))
    df = df.set_index('Datetime')
    return df

def convertTimestamps(timestamp):
 
    dt_object = datetime.fromtimestamp(timestamp / 1000)
    formatted_date = dt_object.strftime('%Y-%m-%d %H:%M:%S')
    return formatted_date

def data_retrieve_yahoo(ticker="AAPL",interval="5m"):
    
    df = yf.Ticker(ticker).history(period="1y",interval=interval)
    if "Dividends" in df.columns:
        df = df.drop("Dividends", axis=1)
    if "Stock Splits" in df.columns:
        df = df.drop("Stock Splits", axis=1)
    # print(df)
    df['time'] = df.index.time
    df_filtered = df[df['time'] != pd.to_datetime('15:30:00').time()]

    df_filtered = df_filtered.drop(columns=['time'])
    return df_filtered

def get_binance_futures_klines(symbol, interval, limit):
    api_key = 'kLIAV7kiJBeWgPCpDHtiqTkBsBMhtC6J8vCac18AS4aq88l5GCqt1ujx7jW508MW'
    api_secret = '27sUNaPVVfoXVSRRMLqvqutHQXUXn3lAFERTppVzfvGgjceyP3k2lKZ6VbPxKdnN'

    base_url = "https://fapi.binance.com"
    endpoint = f"{base_url}/fapi/v1/markPriceKlines"

    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
        "timestamp": int(time.time() * 1000),
    }

    query_string = "&".join(f"{key}={value}" for key, value in params.items())
    signature = hmac.new(api_secret.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()
    params['signature'] = signature

    headers = {
        "X-MBX-APIKEY": api_key,
    }

    response = requests.get(endpoint, params=params, headers=headers)
    data = json.loads(response.text)

    columns = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades',
               'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']
    df = pd.DataFrame(data, columns=columns)
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='ms')
    df['Close time'] = pd.to_datetime(df['Close time'], unit='ms')
    df = df[['Datetime', 'Close', 'High', 'Low', 'Open', 'Volume']]

    df = df.astype({col: float for col in df.columns[1:]})

    df = df.set_index('Datetime')


    return df


# data_retrieve_yahoo("AAPL","1h")




