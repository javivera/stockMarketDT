import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

def confusion(y_test:pd.Series, preds:pd.Series, ticker:str, interval:str, modelName:str):

    matrix = confusion_matrix(y_test, preds.apply(lambda x: 1 if x == 1 else 0))
    # Build the plot
    plt.figure(figsize=(16,7))
    sns.set(font_scale=1.4)
    sns.heatmap(matrix, annot=True, annot_kws={'size':10},
                cmap=plt.cm.Greens, linewidths=0.2, fmt='g')

    # Add labels to the plot
    plt.savefig(f'offline/{ticker}/{interval}/{modelName}/confusion.png')

def returns_graph(dfs,ticker,interval,model):
    fig = make_subplots(rows=4, cols=1, subplot_titles=[
    'Returns',
    'Histogram',
    'Variance'
    ],vertical_spacing=0.1)

    # fig = go.Figure()
    acc = dfs[0][0]
    f1 = dfs[0][5]
    start_date = dfs[0][1].index[0]
    end_date = dfs[0][1].index[-1]
    for j in dfs:
        # print(j[0]) 
        # j[1].reset_index(inplace=True)
        fig.add_trace(go.Scatter(x=j[1].index, y=j[1]['Model_Strat'], mode='lines', name=f"{j[4]} Model returns", yaxis='y1'),row=1, col=1)

    # dfs[0][1].reset_index(inplace=True)
    fig.add_trace(go.Scatter(x=dfs[0][1].index, y=dfs[0][1]['Baseline_Strat'], mode='lines', name='Baseline Returns', yaxis='y1'),row=1, col=1)
    # fig.update_layout(title_text=f"Returns Chart for {ticker} with time {interval}, accuracy {acc} and f1 {f1}, start time {start_date}, end time {end_date}")
    # fig.write_html(f'offline/{ticker}/{interval}/{modelName}/graphs/{ticker}-{interval}_chart.html', auto_open=False)
        
    # -------------------------------------------------------------------------------------------------------------------
    # fig = go.Figure()
    # fig.update_layout(title_text=f"Sharpe Ratio for {ticker} with time {interval}, accuracy {acc} and f1 {f1}")
    # fig.add_trace(go.Scatter(x=dfs[0][1].index, y=dfs[0][1]['sharpe_ratio'], mode='lines', name=f"{dfs[4]} Sharpe Ratio", yaxis='y1'))
    # fig.write_html(f'offline/{ticker}/{interval}/{modelName}/graphs/{ticker}-{interval}_sharpe.html', auto_open=False)
    # -------------------------------------------------------------------------------------------------------------------

    fig.add_trace(go.Histogram(x=dfs[0][1]['return_strat'], nbinsx=50, marker_color='#008080',name='Histogram'),row=2, col=1)
    # -------------------------------------------------------------------------------------------------------------------
    fig.add_trace(go.Scatter(x=dfs[0][1].index, y=dfs[0][1]['Variance'], mode='lines', name=f"{dfs[0][4]} Variance", yaxis='y1'),row=3, col=1)

       # fig.write_html(f'offline/{ticker}/{interval}/{modelName}/graphs/{ticker}-{interval}_variance.html', auto_open=False)

    print('Returns graphs Generated')

    df = dfs[0][1]
    fig.add_trace(go.Candlestick(x=df.index,open=df['Open'],high=df['High'],low=df['Low'],close=df['Close']),row=4 ,col =1)
    df = df.loc[df['sum_of_preds'] == 1]
    highlight_dates_pred_down = df.loc[df['Pred'] == -1].index
    highlight_dates_pred_up = df.loc[df['Pred'] == 1].index
    highlight_dates_correct = df.loc[df['correct'] == True].index
    highlight_dates_incorrect = df.loc[df['correct'] == False].index

    for j in range(len(highlight_dates_pred_down)):
        # Find the index corresponding to the highlight date
        highlight_index = df.index.get_loc(highlight_dates_pred_down[j])
        # Add an arrow annotation to highlight the specific data point
        fig.add_annotation(go.layout.Annotation(
            text='Highlighted',
            x=df.index[highlight_index],
            y=df['Close'].iloc[highlight_index],
            arrowhead=2,  # Arrow style
            arrowsize=1.5,  # Size of the arrow
            arrowwidth=2,  # Width of the arrow
            arrowcolor='red',  # Color of the arrow
            ax=0,  # Arrow x-axis component
            ay=-40,
            yshift=60,  # Arrow y-axis component
        ),row=4, col=1)

    for j in range(len(highlight_dates_pred_up)):
        # Find the index corresponding to the highlight date
        highlight_index = df.index.get_loc(highlight_dates_pred_up[j])
        # Add an arrow annotation to highlight the specific data point
        fig.add_annotation(go.layout.Annotation(
            
        
            text='Highlighted',
            x=df.index[highlight_index],
            y=df['Close'].iloc[highlight_index],
            arrowhead=2,  # Arrow style
            arrowsize=1.5,  # Size of the arrow
            arrowwidth=2,  # Width of the arrow
            arrowcolor='blue',  # Color of the arrow
            ax=0,  # Arrow x-axis component
            ay=40,
            yshift=-60,  # Arrow y-axis component
        ),row=4, col=1)

    for j in range(len(highlight_dates_correct)):
        # Find the index corresponding to the highlight date
        highlight_index = df.index.get_loc(highlight_dates_correct[j])
        # Add an arrow annotation to highlight the specific data point
        fig.add_annotation(go.layout.Annotation(
            
            text='Highlighted',
            x=df.index[highlight_index],
            y=df['Close'].iloc[highlight_index],
            arrowhead=2,  # Arrow style
            arrowsize=1.5,  # Size of the arrow
            arrowwidth=2,  # Width of the arrow
            arrowcolor='green',  # Color of the arrow
            ax=0,  # Arrow x-axis component
            ay=-80,
            yshift=80,  # Arrow y-axis component
        ),row=4, col=1)


    for j in range(len(highlight_dates_incorrect)):
        # Find the index corresponding to the highlight date
        highlight_index = df.index.get_loc(highlight_dates_incorrect[j])
        # Add an arrow annotation to highlight the specific data point
        fig.add_annotation(go.layout.Annotation(
            text='Highlighted',
            x=df.index[highlight_index],
            y=df['Close'].iloc[highlight_index],
            arrowhead=2,  # Arrow style
            arrowsize=1.5,  # Size of the arrow
            arrowwidth=2,  # Width of the arrow
            arrowcolor='yellow',  # Color of the arrow
            ax=0,  # Arrow x-axis component
            ay=80,
            yshift=-80,  # Arrow y-axis component
        ),row=4, col=1)

    fig.update_layout(
        title_text=f"Graphs for {ticker} in {interval}, accuracy {acc} and f1 {f1}, start time {start_date}, end time {end_date}.{model}",
        height=2200, # Adjust the height as needed
        width=1400,
        xaxis_rangeslider_visible=False,
    )


    print(f'Creating graph for {ticker}')
    fig.write_html(f'offline/{ticker}/{interval}/{model}/{ticker}-{interval}_strat.html', auto_open=False)
    # fig.write_html(dir, auto_open=False)

# def returns_distr(df:pd.DataFrame,ticker:str,interval:str,modelName:str):
#
#     df['return_strat'].hist()
#     if os.path.exists(f'offline/{ticker}/{interval}/{modelName}/graphs/earnings_distr.png'):
#     # Delete the file
#         os.remove(f'offline/{ticker}/{interval}/{modelName}/graphs/earnings_distr.png')
#
#     plt.savefig(f'offline/{ticker}/{interval}/{modelName}/graphs/earnings_distr.png')



