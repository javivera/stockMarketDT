from datetime import datetime, timedelta
from tastytrade import ProductionSession
import pandas as pd
from datetime import datetime

from tastytrade.dxfeed import EventType
from tastytrade import DXLinkStreamer
import time

from decimal import Decimal
from tastytrade import Account
from tastytrade.instruments import Equity
from tastytrade.order import *

def get_data(sym,interval,start_time,end_time=None):
    password = "Lasheras:1441"
    username = "verave.javu@gmail.com"
    session = ProductionSession(username, password)
    print(session)
    print('Retrieving data from Tasty')
    print(interval,start_time,end_time)
    print(session.get_candle([sym], interval=interval, start_time=start_time, end_time=end_time))
    print('Tasty data retrieved sucessful')
    data = {'time' : [c.time for c in candles],
            'High': [c.high for c in candles], 
            'Close': [c.close for c in candles], 
            'Open': [c.open for c in candles], 
            'Low': [c.low for c in candles], 
            'Volume': [c.volume for c in candles],
            'vwap' : [c.vwap for c in candles],}


    df = pd.DataFrame(data)
    df['time'] = df.time.apply(lambda x: datetime.fromtimestamp(x/1000).strftime('%Y-%m-%d %H:%M:%S'))
    df['timestamp'] = pd.to_datetime(df['time'])

    start_time = pd.to_datetime('22:00:00').time()
    end_time = pd.to_datetime('11:30:00').time()
    mask = (df['timestamp'].dt.time > start_time) | (df['timestamp'].dt.time < end_time)
    del df['timestamp']

    # Filter out rows within the specified time range
    df = df[~mask]
    df.set_index('time', inplace=True)

    print(df)
    return df

def get_bid_ask(sym):
    password = "Alsinas:2440"
    username = "verave.javu@gmail.com"
    session = ProductionSession(username, password)
    quotes = session.get_event(EventType.QUOTE, sym)
    data = {'bid' : [quotes[0].bidPrice],
            'ask' : [quotes[0].askPrice],
            'time' : [quotes[0].bidTime]}
    df = pd.DataFrame(data)
    df['time'] = df.time.apply(lambda x: datetime.utcfromtimestamp(x/1000).strftime('%Y-%m-%d %H:%M:%S'))
    df.set_index('time', inplace=True)

async def get_current_price(sym):
    password = "Lasheras:1441"
    username = "verave.javu@gmail.com"
    session = ProductionSession(username, password)

    async with DXLinkStreamer(session) as streamer:
        await streamer.subscribe(EventType.QUOTE, [sym])
        quotes = {}
        async for quote in streamer.listen(EventType.QUOTE):
            quotes[quote.eventSymbol] = quote
            break

        return quotes[sym].askPrice

def open_order(sym):
    password = "Alsinas:2440"
    username = "verave.javu@gmail.com"
    session = ProductionSession(username, password)

    account = Account.get_account(session, '5WX64928')
    symbol = Equity.get_equity(session, sym)
    leg = symbol.build_leg(Decimal('1'), OrderAction.BUY_TO_OPEN)  # buy to open 5 shares

    order = NewOrder(
        time_in_force=OrderTimeInForce.DAY,
        order_type=OrderType.LIMIT,
        legs=[leg],  # you can have multiple legs in an order
        price=Decimal('5'),  # limit price, here $50 for 5 shares = $10/share
        price_effect=PriceEffect.DEBIT
    )
    response = account.place_order(session, order, dry_run=False)  # a test order
    print(response)

# open_order('YPF')
# get_bid_ask(['AAPL'])

# start_time = datetime.now() - timedelta(days=200)  # 1 month ago
# df = get_data('TSLA','1d',start_time,None)

# if __name__ == '__main__':

    # start_time = datetime.now() - timedelta(days=1)  # 1 month ago
    # df = get_data('TSLA','1m',start_time,None)
    # df = get_data2('TSLA','1m',start_time,None)
    # open_order('YPF')
    # print(df)
