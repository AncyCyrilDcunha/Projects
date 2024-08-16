import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import yfinance as yf
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pi
import plotly.express as px
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta , datetime
import streamlit as st


# Define the start and end dates
start_date = "2022-01-01"
end_date = "2024-07-20"


st.title('Stock Trend Prediction')
user_input = st.text_input('Enter Stock Ticker', 'AAPL')
ticker_symbol = user_input
# user_date_input_start = st.text_input('Enter Start Date to train Model', "2022-01-01")
# user_date_input_end = st.text_input('Enter Start Date to train Model', "2024-07-20")


# start_date = user_date_input_start
# end_date = user_date_input_end
ticker_symbol = user_input

# date_object = datetime.strptime(end_date, '%Y-%m-%d')
# def is_leap_year(i):
#     return i % 4 == 0 and (i % 100 != 0 or i % 400 == 0)

# Streamlit input for the number of years
#user_input_years = st.number_input('Enter the number of years for stock price prediction:', min_value=1, step=1)

# # Calculate the total number of days
# total_days = 0
# current_year = date_object.year

# for i in range(current_year + user_input_years):
#     if is_leap_year(i):
#         total_days += 366
#     else:
#         total_days += 365



# Fetch the historical data
stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
stock_data=stock_data.reset_index()
stock_data.head()

# Display the first few rows of the data
st.subheader('Stock Data')
st.write(stock_data.head())
st.subheader('Recent Stock Data')
st.write(stock_data.tail())

#Describing data
st.subheader('Data Summary')
st.write(stock_data.describe())

#Plotting Daily volume trade
st.subheader('Daily Volume Trade')
fig1 = px.line(stock_data, x='Date', y='Volume')
fig1.update_xaxes(title='Date')
fig1.update_yaxes(title='Volume')
fig1.update_layout(template='plotly_dark')
#fig.show()
st.plotly_chart(fig1)




#Closing Prices Over Time
st.subheader('Closing Prices Over Time')
fig = px.line(stock_data, x='Date', y='Close')
fig.update_xaxes(title='Date')
fig.update_yaxes(title='Closing Price')
fig.update_layout(template='plotly_dark')
st.plotly_chart(fig)

#Visualiztions
st.subheader('100 Days Moving Average')
fig = px.line(stock_data.Close.rolling(100).mean())
#fig=plt.figure(figsize =(12,6))
#plt.plot(ma100)
fig.update_xaxes(title='Date')
fig.update_yaxes(title='Closing Price')
fig.update_layout(template='plotly_dark')
#plt.plot(stock_data.Close)
st.plotly_chart(fig)


st.subheader('Closing Price vs Time chart with 100MA & 200MA')
# Calculate the moving averages
ma100 = stock_data['Close'].rolling(100).mean()
ma200 = stock_data['Close'].rolling(200).mean()

# Create a figure
fig = go.Figure()

# Add the original closing prices
fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close'))

# Add the 100-day MA
fig.add_trace(go.Scatter(x=stock_data.index, y=ma100, mode='lines', name='100MA', line=dict(color='red')))

# Add the 200-day MA
fig.add_trace(go.Scatter(x=stock_data.index, y=ma200, mode='lines', name='200MA', line=dict(color='green')))

# Update layout
fig.update_layout(
                  xaxis_title='Date',
                  yaxis_title='Price',
                  template='plotly_dark')

st.plotly_chart(fig)



stock_data['Date'] = pd.to_datetime(stock_data['Date'])
stock_data = stock_data.sort_values('Date')
stock = stock_data[['Date', 'Close', 'High', 'Low', 'Open', 'Volume']]
scaler = MinMaxScaler()
normalized_data = stock[['Open', 'High', 'Low', 'Volume', 'Close']].copy()
normalized_data = scaler.fit_transform(normalized_data)

#Splitting data into training and testing
train_data, test_data = train_test_split(normalized_data, test_size=0.2, shuffle=False)
train_df = pd.DataFrame(train_data, columns=['Open', 'High', 'Low', 'Volume', 'Close'])
test_df = pd.DataFrame(test_data, columns=['Open', 'High', 'Low', 'Volume', 'Close'])
def generate_sequences(df, seq_length=50):
    X = df[['Open', 'High', 'Low', 'Volume', 'Close']].reset_index(drop=True)
    y = df[['Open', 'High', 'Low', 'Volume', 'Close']].reset_index(drop=True)

    sequences = []
    labels = []

    for index in range(len(X) - seq_length + 1):
        sequences.append(X.iloc[index : index + seq_length].values)
        labels.append(y.iloc[index + seq_length - 1].values)

    sequences = np.array(sequences)
    labels = np.array(labels)

    return sequences, labels

train_sequences, train_labels = generate_sequences(train_df)
test_sequences, test_labels = generate_sequences(test_df)

#Load my model
model =load_model('keras_model.keras')

#Testing Part
train_predictions = model.predict(train_sequences)
test_predictions = model.predict(test_sequences)
fig = make_subplots(rows=1, cols=1, subplot_titles=('Close Predictions'))

train_close_pred = train_predictions[:, 0]
train_close_actual = train_labels[:, 0]

# st.subheader('Close Price Predictions on Training Data')

# fig.add_trace(go.Scatter(x=np.arange(len(train_close_actual)), y=train_close_actual, mode='lines', name='Actual', opacity=0.9))
# fig.add_trace(go.Scatter(x=np.arange(len(train_close_pred)), y=train_close_pred, mode='lines', name='Predicted', opacity=0.6))

# fig.update_layout( template='plotly_dark')
# st.plotly_chart(fig)

# fig = make_subplots(rows=1, cols=1, subplot_titles=('Close Predictions'))

test_close_pred = test_predictions[0, :]
test_close_actual = test_labels[0, :]

# st.subheader('Close Price Predictions on Test Data')

# fig.add_trace(go.Scatter(x=np.arange(len(test_close_actual)), y=test_close_actual, mode='lines', name='Actual', opacity=0.9))
# fig.add_trace(go.Scatter(x=np.arange(len(test_close_pred )), y=test_close_pred, mode='lines', name='Predicted', opacity=0.6))

# fig.update_layout( template='plotly_dark')
# #fig.show()
# st.plotly_chart(fig)

#Next 10 days prediction
latest_prediction = []
last_seq = test_sequences[-1]  # Assuming test_sequences is a numpy array

for _ in range(11):
    prediction = model.predict(last_seq.reshape(1, last_seq.shape[0], last_seq.shape[1]))
    latest_prediction.append(prediction)
    last_seq = np.vstack((last_seq[1:], prediction))

# Prepare data for the final prediction
predicted_data_next = np.array(latest_prediction).squeeze().T

from datetime import datetime, timedelta

# Generate dates for the next 10 days
last_date = stock_data['Date'].max()
dates = [last_date + timedelta(days=i) for i in range(0, 11)]


#Plotting the normalised predicted prices against dates
# fig = go.Figure()

# # Add the predicted data
# fig.add_trace(go.Scatter(x=dates, y=predicted_data_next[-1], mode='lines+markers',line=dict(color='blue')))

# # Update layout
# fig.update_layout(
#     title='Predicted Close Prices for the Next 10 Days',
#     xaxis_title='Date',
#     yaxis_title='Predicted Price',
    
    
# )
# fig.update_xaxes( showline=True, showgrid=True, gridwidth=0.1, gridcolor='white')
# fig.update_yaxes(tickformat=".2f", showline=True, showgrid=True, gridwidth=0.1, gridcolor='white')

# # Display the plot in Streamlit
# st.plotly_chart(fig)


min_close = stock_data['Close'].min()
max_close = stock_data['Close'].max()


# Denormalize using MinMaxScaler formula: denormalized_value = (normalized_value * (max - min)) + min
denormalized_predicted_prices = predicted_data_next[-1] * (max_close - min_close) + min_close
print(denormalized_predicted_prices )

# Generate dates for the next 10 days
last_date = stock_data['Date'].max()  # Replace 'stock' with your dataframe name
# dates = [last_date + timedelta(days=i) for i in range(1, 12)]  # Start from 1 as you already have the prediction for day 0

dates = []
i = 1
while len(dates) < 10:
    next_date = last_date + timedelta(days=i)
    if next_date.weekday() < 5:  # 0 = Monday, 1 = Tuesday, ..., 4 = Friday
        dates.append(next_date)
    i += 1

#Plotting the denormalised predicted values
st.subheader('Predicted Close Prices for the Next 10 Days')

fig = go.Figure()

# Add the predicted data
fig.add_trace(go.Scatter(x=dates, y=denormalized_predicted_prices, mode='lines+markers',line=dict(color='blue')))

# Update layout
fig.update_layout(
    
    xaxis_title='Date',
    yaxis_title='Predicted Price',
    
    
)
fig.update_xaxes( showline=True)
fig.update_yaxes(tickformat=".2f", showline=True)

# Display the plot in Streamlit
st.plotly_chart(fig)















#Next 10 days prediction
# latest_prediction = []
# last_seq = train_sequences[-1]  # Assuming test_sequences is a numpy array

# for _ in range(366):
#     prediction = model.predict(last_seq.reshape(1, last_seq.shape[0], last_seq.shape[1]))
#     latest_prediction.append(prediction)
#     last_seq = np.vstack((last_seq[1:], prediction))

# # Prepare data for the final prediction
# predicted_data_next = np.array(latest_prediction).squeeze().T

# from datetime import datetime, timedelta
# # Generate dates for the next 10 days
# last_date = stock_data['Date'].max()
# dates = [last_date + timedelta(days=i) for i in range(0, 366)]
# min_close = stock_data['Close'].min()
# max_close = stock_data['Close'].max()


# # Denormalize using MinMaxScaler formula: denormalized_value = (normalized_value * (max - min)) + min
# denormalized_predicted_prices = predicted_data_next[-1] * (max_close - min_close) + min_close
# print(denormalized_predicted_prices )

# # Generate dates for the next 10 days
# last_date = stock_data['Date'].max()  # Replace 'stock' with your dataframe name
# # dates = [last_date + timedelta(days=i) for i in range(1, 12)]  # Start from 1 as you already have the prediction for day 0

# dates = []
# i = 1
# while len(dates) < 366:
#     next_date = last_date + timedelta(days=i)
#     if next_date.weekday() < 5:  # 0 = Monday, 1 = Tuesday, ..., 4 = Friday
#         dates.append(next_date)
#     i += 1
# #Plotting the denormalised predicted values
# st.subheader('Predicted Close Prices for the Next 10 Days')

# fig = go.Figure()

# # Add the predicted data
# fig.add_trace(go.Scatter(x=dates, y=denormalized_predicted_prices, mode='lines+markers',line=dict(color='blue')))

# # Update layout
# fig.update_layout(
    
#     xaxis_title='Date',
#     yaxis_title='Predicted Price',
    
    
# )
# fig.update_xaxes( showline=True)
# fig.update_yaxes(tickformat=".2f", showline=True)

# # Display the plot in Streamlit
# st.plotly_chart(fig)



