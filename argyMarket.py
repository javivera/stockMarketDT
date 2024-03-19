import plotly.graph_objects as go
import plotly.subplots as sp
from dataRetrieveLib import *
from tastyAPI import get_current_price

ticker_list = ['YPF','EDN','BBAR','MELI','LOMA','GGAL','BMA','CAAP','CEPU','CRESY','DESP','GLOB','IRS','PAM','SUPV','TEO','TGS','TS','TX']
ticker_list = ['ETH-USD','SOL-USD','BTC-USD','BNB-USD','XRP-USD','DOGE-USD','ADA-USD']


df_list = []
def pct_change(df):
    df['change'] = df.Close - df.Close.shift(1)
    df['pct_change'] = df.change / df.Close.shift(1) * 100
    df['pct_sum'] = df['pct_change'].cumsum()
    # print(df['difference'].value_counts()[0]/df['difference'].count())
    return df
    
for j in ticker_list:
    # df_list.append(np.log(data_retrieve_yahoo(ticker=j,interval="1d").Close).diff())
    # df_list.append(data_retrieve_yahoo(ticker=j,interval="1d"))
    # price = await get_current_price(j)

    df = data_retrieve_yahoo(ticker=j,interval="1d")
    print(df)
    # df.loc[-1,'Close'] = price
    df_list.append(pct_change(df))

pairs = [(ticker_list[i], ticker_list[j]) for i in range(len(ticker_list)) for j in range(i+1, len(ticker_list))]
pairs_df = [(df_list[i], df_list[j]) for i in range(len(df_list)) for j in range(i+1, len(df_list))]


counter = 0
for j in pairs_df:
    corr = (j[0]['pct_change']).corr(j[1]['pct_change'])
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=j[0].index, y=j[0]['pct_change'], mode='lines', name='PCT_Change1'))
    fig2.add_trace(go.Scatter(x=j[1].index, y=j[1]['pct_change'], mode='lines', name='PCT_Change2'))

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=j[0].index, y=j[0].Close, mode='lines', name='Close_1'))
    fig4.add_trace(go.Scatter(x=j[1].index, y=j[1].Close, mode='lines', name='Close_2'))

    if True:

        df_diff = (j[0] - j[1])[['pct_change']].dropna()
        df_diff['stock_1'] = j[0]['pct_change']
        df_diff['stock_2'] = j[1]['pct_change']

        df_diff['next_pct_change'] = df_diff['pct_change'].shift(-1)
        df_diff['prev_pct_change'] = df_diff['pct_change'].shift(1)
        df_diff['difference'] = df_diff.apply(lambda x: 1 if (x['next_pct_change'] > 0 and x['pct_change'] < 0) or (x['next_pct_change'] < 0 and x['pct_change'] >0 ) else -1, axis=1)
        win_rate = df_diff['difference'].value_counts()[1] / df_diff['difference'].count()

        def cases(x):
            if (x['prev_pct_change'] < 0 and x['next_pct_change'] > 0 and x['pct_change'] < 0) or (x['next_pct_change'] < 0 and x['pct_change'] > 0 and x['prev_pct_change'] > 0 ):
                return 1
            elif (x['pct_change'] < 0 and x['prev_pct_change'] > 0) or (x['pct_change'] > 0 and x['prev_pct_change'] < 0):
                return 0
            else:
                return -1

        df_diff['difference_2'] = df_diff.apply(cases,axis=1) 
        df_diff['strat_test'] = abs(df_diff['next_pct_change']) * df_diff['difference']
        df_diff['strat_test'] = df_diff['strat_test'].cumsum().shift(1)
        df_diff['strat_test2'] = abs(df_diff['next_pct_change']) * df_diff['difference_2']
        df_diff['strat_test2'] = df_diff['strat_test2'].cumsum().shift(1)

        print(pairs[counter])
        # if df_diff['strat_test'][-1] > 0 or df_diff['strat_test2'][-1] > 0:
        if True or df_diff['strat_test'][-1] > 0:
        # plt.plot(df.index,df)
        # plt.plot(df.index,df.rolling(5).mean())
        # plt.title(pairs[counter])
        # plt.show()
            fig1 = go.Figure(data=go.Scatter(x=df_diff.index, y=df_diff['pct_change'], mode='lines', name='Change'))
        # fig2 = go.Figure(data=go.Scatter(x=df.index, y=df['pct_sum'], mode='lines', name='Sum'))
            fig3 = go.Figure(data=go.Scatter(x=df_diff.index, y=df_diff['strat_test'], mode='lines', name='Strat'))
            fig5 = go.Figure(data=go.Scatter(x=df_diff.index, y=df_diff['strat_test2'], mode='lines', name='Strat2'))


            fig = sp.make_subplots(rows=5, cols=1, shared_xaxes=False, shared_yaxes=False, vertical_spacing=0.05)
            for t in fig2.data:
                fig.add_trace(t, row=3, col=1)
            for t in fig4.data:
                fig.add_trace(t, row=2, col=1)
            fig.add_trace(fig1.data[0], row=1, col=1)
            fig.add_trace(fig3.data[0], row=4, col=1)
            fig.add_trace(fig5.data[0], row=5, col=1)
            # fig.add_trace(fig4.data[0], row=4, col=1)

            fig.update_layout(
                title_text=f'{pairs[counter]} -- {corr} -- {win_rate}',
                showlegend=True,
                legend=dict(x=0.5, y=1.0),
            )


            fig.write_html(f'graphs/{pairs[counter]}.html')
            print(f'figure wrote {pairs[counter]}')
        # fig  = go.Figure(data = [go.Histogram(x = df,nbinsx = 15)])
            fig.show()
    counter += 1
