from mainLib import modelGenerator, bestResultPicker
from dataRetrieveLib import get_binance_futures_klines,data_retrieve_yahoo
import sys
from tastyAPI import get_data
from datetime import datetime
from datetime import timedelta

def generateModels(tickers:list[str],interval:str,predictors:list[str]):
    print(interval)
    for sym in tickers: 
        if interval == '1m':
            start_time = datetime.now() - timedelta(days=5)  # 1 month ago
            step = 240
        elif interval == '5m':
            start_time = datetime.now() - timedelta(days=53)  # 1 month ago
            # step = 72 # 6 horas
            step = 78 # 1 dia de mercado abierto
        elif interval == '15m':
            start_time = datetime.now() - timedelta(days=100)  # 1 month ago
            step = 35
        elif interval == '1h':
            start_time = datetime.now() - timedelta(days=200)  # 1 month ago
            step = 25
        elif interval == '1d':
            start_time = datetime.now() - timedelta(days=200)  # 1 month ago
            step = 20
        else:
            print('no interval selected')
            return False
        # For Crypto
        # df = get_binance_futures_klines(sym,interval,1500)
        # -------------------------------------------------------

        end_time = (datetime.now() - timedelta(hours=4)).strftime("%Y-%m-%d %H:%M:%S")
        end_time = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
        df = get_data(sym,interval,start_time,end_time)
        results,df = modelGenerator(df,sym,predictors,step)
        bestResultPicker(results,sym,interval)

if __name__ == '__main__':
    arguments = sys.argv[1:]  # List of command-line arguments
    predictors = ['High','Low','Close','RSI', 'MACD', 'MACD_Signal', 'PROC', 'STOCH_K', 'WILL_R','SMA','EMA','upper_band','lower_band','SAR','AOI','OBV','Volume','CCI','ROC','TRIX']
    generateModels(arguments[:-1],arguments[-1],predictors)
