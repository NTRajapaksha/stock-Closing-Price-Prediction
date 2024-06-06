import math
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import streamlit as st
import datetime

st.write('<h1 style="color:blue; font-size:48px; text-align:center;font-family:monospace;">Stock Closer Prediction</h1>', unsafe_allow_html=True)

st.write("")
st.write("")

st.markdown("<p style='font-size: 15px; text-align: center;'>AAPL = Apple Inc</p>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 15px; text-align: center;'>GOOG = Alphabet Inc</p>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 15px; text-align: center;'>005930.KS = Samsung Electronics Co., Ltd.</p>", unsafe_allow_html=True)

user_input = st.selectbox(
    'Choose a BRAND',
    ('AAPL', 'GOOG', '005930.KS')
)
start = '2014-03-10'
end = datetime.date.today()

# Style the date display using HTML/CSS
styled_date = f"<div style='font-size: 15px;'>{end}</div>"

# Display the styled date
st.write('Date to predict:')
st.write(styled_date, unsafe_allow_html=True)

@st.cache_data
def load_data(ticker):
    return yf.download(ticker, start, end)

df = load_data(user_input)

st.write("")
st.write("")
st.subheader('Data from 2014 - 2024')
st.write(df)

st.write("")
st.subheader('Closing Price over Time')
fig = plt.figure(figsize=(16, 8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Time', fontsize=18)
plt.ylabel('Price', fontsize=18)
st.pyplot(fig)

st.write("")
st.subheader('Closing Price Vs Time with 100MA and 200MA')
st.write('<p>The Moving Average(MA) is a statistical method used to smooth out fluctuations in data over time, revealing underlying trends or patterns. It calculates the average value of a series of data points by continuously updating the average as new data becomes available.</p>', unsafe_allow_html=True)
ma100 = df['Close'].rolling(100).mean()
ma200 = df['Close'].rolling(200).mean()
fig2 = plt.figure(figsize=(12, 6))
plt.plot(df['Close'])
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.legend(['Closing Price', 'MA100', 'MA200'], loc='lower right')
st.pyplot(fig2)

# Data preprocessing
data = df.filter(['Close'])
dataset = data.values
train_len = math.ceil(len(dataset) * .8)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

train_data = scaled_data[0:train_len, :]
x_train, y_train = [], []

for i in range(100, len(train_data)):
    x_train.append(train_data[i-100:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

@st.cache_resource
def load_trained_model():
    return load_model('s_model.keras')

model = load_trained_model()

test_data = scaled_data[train_len - 100:, :]
x_test = []
y_test = dataset[train_len:, :]

for i in range(100, len(test_data)):
    x_test.append(test_data[i-100:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

train = data[:train_len]
valid = data[train_len:]
valid['Predictions'] = predictions

st.write("")
st.subheader('Predictions')
fig3 = plt.figure(figsize=(16, 8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
st.pyplot(fig3)

st.write("")
st.subheader('Model Predictions Overtime')
st.write(valid)

st.write("")
st.subheader('Predicted Closing Price for Today ($):')

def predict_today_price():
    latest_data = df.filter(['Close'])[-100:].values
    latest_data_scaled = scaler.transform(latest_data)
    X_test = []
    X_test.append(latest_data_scaled)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_price = model.predict(X_test)
    predicted_price = scaler.inverse_transform(predicted_price)
    return predicted_price[0]

today_predicted_price = predict_today_price()
st.write(f"${today_predicted_price[0]:.2f}", text_align="center")
