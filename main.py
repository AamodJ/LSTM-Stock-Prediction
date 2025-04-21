import pandas as pd 
import numpy as np


import matplotlib.pyplot as plt 


# Need to set KERAS_BACKEND = torch before importing keras
import os 
os.environ['KERAS_BACKEND'] = 'torch'


# keras imports
import keras 
from keras import Input
import keras_tuner as kt 
from keras import layers 

# sklearn imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# misc
import yfinance as yf 
import math 
from datetime import datetime, timedelta 

keras.utils.set_random_seed(42)

interval = '5m'
ticker = '^IXIC'
start_date = datetime.today() - timedelta(days=60)
end_date = datetime.today()

df: pd.DataFrame
df = yf.download(ticker, interval=interval, start=start_date, end=end_date)
df.columns = df.columns.droplevel(level=1)
df.rename(columns={key: key.lower() for key in df.columns}, inplace=True)
print(f'Dataframe shape: {df.shape}')


X_train, X_test, y_train, y_test = train_test_split(df, pd.DataFrame(df.close), test_size=0.3, shuffle=False)


# Define MinMaxScalers 
## Note these need to be separate for test and train datasets so that no information from test data leaks into training
X_train_scaler = MinMaxScaler()
y_train_scaler = MinMaxScaler()

# Fit scalers on train data 
X_train_scaled = X_train_scaler.fit_transform(X_train)
y_train_scaled = y_train_scaler.fit_transform(y_train)


def create_datasets(dataX, dataY, lookback=60):
  X = []
  Y = []

  for i in range(0, len(dataX) - lookback):
    X.append(dataX[i: i + lookback, ])
    Y.append(dataY[i + lookback, ])

  return np.array(X), np.array(Y)


X_train_scaled, y_train_scaled = create_datasets(X_train_scaled, y_train_scaled)

batch_size, timestep, feature_count = X_train_scaled.shape

print(f'Batch size: {batch_size}')
print(f'Timestep (=lookback): {timestep}')
print(f'Number of features: {feature_count}')

# Model building 
inputs = Input(shape=(timestep, feature_count))
outputs = layers.LSTM(30)(inputs)
# outputs = layers.Dropout(0.2)(outputs)
outputs = layers.Dense(1)(outputs)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(loss='mean_squared_error', optimizer='adam')

# Stop training if validation loss doesn't improve 
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, min_delta=1e-4)

hist = model.fit(
  X_train_scaled, 
  y_train_scaled, 
  batch_size=10, 
  epochs=10, 
  validation_split=0.33, 
  callbacks=[early_stopping]
)

# Evaluate model on training data
y_train_pred = model.predict(X_train_scaled)
y_train_pred = np.ravel(y_train_scaler.inverse_transform(y_train_pred))
y_train_true = np.ravel(y_train_scaler.inverse_transform(y_train_scaled))

train_rmse = np.sqrt(mean_squared_error(y_train_true, y_train_pred))

model.save('./models/model.keras')

# Test data
X_test_scaler = MinMaxScaler()
y_test_scaler = MinMaxScaler()

X_test_scaled = X_test_scaler.fit_transform(X_test)
y_test_scaled = y_test_scaler.fit_transform(y_test)

X_test_scaled, y_test_scaled = create_datasets(X_test_scaled, y_test_scaled)

# Evaluate model on test data
y_test_pred = model.predict(X_test_scaled)
y_test_pred = np.ravel(y_test_scaler.inverse_transform(y_test_pred))
y_test_true = np.ravel(y_test_scaler.inverse_transform(y_test_scaled))

test_rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
print(f'Train RMSE: {train_rmse}')
print(f'Test RMSE: {test_rmse}')


# Make plots
fig, ax0 = plt.subplots(figsize=(15, 5))

## Close price train vs test
ax0.plot(np.concatenate((y_train_true, y_test_true)), label='True')
ax0.plot(y_train_pred, label='Train Prediction')

y_test_padding = np.empty(len(y_train_pred))
y_test_padding[: ] = np.nan

ax0.plot(np.concatenate((y_test_padding, y_test_pred)), label='Test Prediction')

ax0.set_title('Train - Test Dataset Performance')
ax0.legend()

plt.savefig('./results/train_test_performance.png')

## Loss function vs epochs 
fig, ax1 = plt.subplots(figsize=(15, 5))
ax1.plot(hist.history['loss'], label='Training Loss')
ax1.plot(hist.history['val_loss'], label='Validation Loss')

ax1.set_title('Training vs Validation Loss')

ax1.set_xticks([i for i in range(len(hist.history['loss']))])
ax1.legend()

plt.savefig('./results/loss_curves.png')

plt.show()
