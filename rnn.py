#Importing the Libraries
import pandas as pd
import numpy as np
import matplotlib. pyplot as plt
import matplotlib
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib. dates as mandates
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.models import load_model
from keras.layers import LSTM
from keras.utils.vis_utils import plot_model
from datetime import datetime
from datetime import timedelta
from mainLib import addFeauturesToDf
import matplotlib.pyplot as plt

from tastyAPI import get_data
end_time = "2023-12-05 6:30:00"
start_time = datetime.now() - timedelta(days=300)  # 1 month ago
date_format = "%Y-%m-%d %H:%M:%S"
end_time = datetime.strptime(end_time, date_format)

df = get_data('YPF','5m',start_time,end_time)
df = addFeauturesToDf(df)
print(df)
features = ['RSI', 'MACD', 'MACD_Signal', 'PROC', 'STOCH_K', 'WILL_R','SMA','EMA','upper_band','lower_band','SAR','AOI','OBV','vwap','Volume','CCI','ROC','TRIX','Open','High','Low','Volume']
scaler = MinMaxScaler()

output_var = pd.DataFrame(df['Target'])

feature_transform = scaler.fit_transform(df[features])
feature_transform= pd.DataFrame(columns=features, data=feature_transform, index=df.index)
feature_transform.head()

timesplit= TimeSeriesSplit(n_splits=10)
for train_index, test_index in timesplit.split(feature_transform):
        X_train, X_test = feature_transform[:len(train_index)], feature_transform[len(train_index): (len(train_index)+len(test_index))]
        y_train, y_test = output_var[:len(train_index)].values.ravel(), output_var[len(train_index): (len(train_index)+len(test_index))].values.ravel()


trainX =np.array(X_train)
testX =np.array(X_test)
X_train = trainX.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = testX.reshape(X_test.shape[0], 1, X_test.shape[1])


lstm = Sequential()
lstm.add(LSTM(50, input_shape=(1, trainX.shape[1]), activation='relu', return_sequences=False))
lstm.add(Dense(units=1,activation='sigmoid'))
lstm.compile(loss='mean_squared_error', optimizer='adam')
plot_model(lstm, show_shapes=True, show_layer_names=True)

history=lstm.fit(X_train, y_train, epochs=100, batch_size=15, verbose=1, shuffle=False)

y_pred= lstm.predict(X_test)
plt.plot(y_test, label='True Value')
plt.plot(y_pred, label='LSTM Value')
plt.title('Prediction by LSTM')
plt.xlabel('Time Scale')
plt.ylabel('Scaled USD')
plt.legend()
plt.show()

print(y_pred)

binary_preds = (y_pred > 0.5).astype(int)
print(binary_preds)
print(testX)
X_test['Predictions'] = binary_preds
print(X_test)
