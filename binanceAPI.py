from binance.cm_futures import CMFutures
from binance.client import Client
from binance.enums import *
from binance.exceptions import BinanceAPIException, BinanceOrderException
from decimal import Decimal

client = Client('sOuZwdaABrxLKI73H0KOQPwycFUmiQUcjvTuwOkMXEGiPE6cqBA5cNOlmDBD6Yaf', 
                'Fc9qOY4NWTOPnSq5fW0OCDHBFNVOMEPp4bUht6wnz1roGxq6ldAsS5fsksPqt4YS')

def buy_order(sym,price,quantity):
    client = Client('sOuZwdaABrxLKI73H0KOQPwycFUmiQUcjvTuwOkMXEGiPE6cqBA5cNOlmDBD6Yaf', 
                'Fc9qOY4NWTOPnSq5fW0OCDHBFNVOMEPp4bUht6wnz1roGxq6ldAsS5fsksPqt4YS')

    # client.futures_create_order(symbol=sym, side='BUY', type='MARKET', quantity=quantity)
    print(client.futures_create_order(symbol=sym, side='BUY',type='LIMIT', timeInForce='GTC',price = price, quantity=quantity ))

def sell_order(sym,price,quantity):
    client = Client('sOuZwdaABrxLKI73H0KOQPwycFUmiQUcjvTuwOkMXEGiPE6cqBA5cNOlmDBD6Yaf', 
                'Fc9qOY4NWTOPnSq5fW0OCDHBFNVOMEPp4bUht6wnz1roGxq6ldAsS5fsksPqt4YS')

    a = client.futures_create_order(symbol=sym, side='SELL', type = 'LIMIT', timeInForce='GTC', price = price, quantity=quantity)

def cancel_orders(sym):
    client = Client('sOuZwdaABrxLKI73H0KOQPwycFUmiQUcjvTuwOkMXEGiPE6cqBA5cNOlmDBD6Yaf','Fc9qOY4NWTOPnSq5fW0OCDHBFNVOMEPp4bUht6wnz1roGxq6ldAsS5fsksPqt4YS')

    client.futures_cancel_all_open_orders(symbol = sym)

def get_active_orders():    
    client = Client('sOuZwdaABrxLKI73H0KOQPwycFUmiQUcjvTuwOkMXEGiPE6cqBA5cNOlmDBD6Yaf','Fc9qOY4NWTOPnSq5fW0OCDHBFNVOMEPp4bUht6wnz1roGxq6ldAsS5fsksPqt4YS')
    print(client.futures_get_open_orders())

cancel_orders('ETHBUSD')
