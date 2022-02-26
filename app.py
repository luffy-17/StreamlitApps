# This app is for educational purpose only. Insights gained is not financial advice. Use at your own risk!
import streamlit as st
from PIL import Image
import pandas as pd
import base64
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import requests
import json
import time
import numpy as np
import seaborn as sns
from tensorflow.keras.models import load_model
import joblib
import plotly.express as px

#---------------------------------#
# New feature (make sure to upgrade your streamlit library)
# pip install --upgrade streamlit

#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(layout="wide")
#---------------------------------#
# Title

image = Image.open('logo.jpg')

st.image(image, width = 500)

st.title('Crypto Price App')
st.markdown("""
This app retrieves cryptocurrency prices for the top 100 cryptocurrency from the **CoinMarketCap**!

""")
#---------------------------------#
# About
expander_bar = st.expander("About")
expander_bar.markdown("""
* **Python libraries:** base64, pandas, streamlit, numpy, matplotlib, seaborn, BeautifulSoup, requests, json, time
* **Data source:** [CoinMarketCap](http://coinmarketcap.com).
* **Credit:** Web scraper adapted from the Medium article *[Web Scraping Crypto Prices With Python](https://towardsdatascience.com/web-scraping-crypto-prices-with-python-41072ea5b5bf)* written by [Bryan Feng](https://medium.com/@bryanf).
""")


#---------------------------------#
# Page layout (continued)
## Divide page to 3 columns (col1 = sidebar, col2 and col3 = page contents)
col1 = st.sidebar
col2, col3 = st.columns((2,1))

#---------------------------------#
# Sidebar + Main panel
col1.header('Input Options')

## Sidebar - Currency price unit
currency_price_unit = col1.selectbox('Select currency for price', ('USD', 'BTC', 'ETH'))

# Web scraping of CoinMarketCap data
def load_data():
    cmc = requests.get('https://coinmarketcap.com')
    soup = BeautifulSoup(cmc.content, 'html.parser')

    data = soup.find('script', id='__NEXT_DATA__', type='application/json')
    coins = {}
    coin_data = json.loads(data.contents[0])
    # listings = coin_data['props']['initialState']['cryptocurrency']['listingLatest']['data']
    # for i in listings:
    #   coins[str(i['id'])] = i['slug']
    listings = coin_data['props']['initialState']['cryptocurrency']['listingLatest']['data']
    for i in range(1,101):
        coins[str(listings[i][8])] = listings[i][132]
    coin_name = []
    coin_symbol = []
    market_cap = []
    percent_change_1h = []
    percent_change_24h = []
    percent_change_7d = []
    price = []
    volume_24h = []

    for i in range(1,51):
      coin_name.append(listings[i][132])
      coin_symbol.append(listings[i][133])
      price.append(listings[i][66])
      percent_change_1h.append(listings[i][60])
      percent_change_24h.append(listings[i][61])
      percent_change_7d.append(listings[i][64])
      market_cap.append(listings[i][57])
      volume_24h.append(listings[i][68])

    df = pd.DataFrame(columns=['coin_name', 'coin_symbol', 'price','market_cap', 'percent_change_1h', 'percent_change_24h', 'percent_change_7d', 'volume_24h'])
    df['coin_name'] = coin_name
    df['coin_symbol'] = coin_symbol
    df['price'] = price
    df['percent_change_1h'] = percent_change_1h
    df['percent_change_24h'] = percent_change_24h
    df['percent_change_7d'] = percent_change_7d
    df['market_cap'] = market_cap
    df['volume_24h'] = volume_24h
    return df

df = load_data()

## Sidebar - Cryptocurrency selections
sorted_coin = sorted( df['coin_symbol'] )
selected_coin = col1.multiselect('Cryptocurrency', sorted_coin, sorted_coin)

df_selected_coin = df[ (df['coin_symbol'].isin(selected_coin)) ] # Filtering data

## Sidebar - Number of coins to display
num_coin = col1.slider('Display Top N Coins', 1, 100, 100)
df_coins = df_selected_coin[:num_coin]

## Sidebar - Percent change timeframe
percent_timeframe = col1.selectbox('Percent change time frame',
                                    ['7d','24h', '1h'])
percent_dict = {"7d":'percent_change_7d',"24h":'percent_change_24h',"1h":'percent_change_1h'}
selected_percent_timeframe = percent_dict[percent_timeframe]

## Sidebar - Sorting values
sort_values = col1.selectbox('Sort values?', ['Yes', 'No'])

col2.subheader('Price Data of Selected Cryptocurrency')
col2.write('Data Dimension: ' + str(df_selected_coin.shape[0]) + ' rows and ' + str(df_selected_coin.shape[1]) + ' columns.')

col2.dataframe(df_coins)

# Download CSV data
# https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="crypto.csv">Download CSV File</a>'
    return href

col2.markdown(filedownload(df_selected_coin), unsafe_allow_html=True)

#---------------------------------#
# Preparing data for Bar plot of % Price change
col2.subheader('Table of % Price Change')
df_change = pd.concat([df_coins.coin_symbol, df_coins.percent_change_1h, df_coins.percent_change_24h, df_coins.percent_change_7d], axis=1)
df_change = df_change.set_index('coin_symbol')
df_change['positive_percent_change_1h'] = df_change['percent_change_1h'] > 0
df_change['positive_percent_change_24h'] = df_change['percent_change_24h'] > 0
df_change['positive_percent_change_7d'] = df_change['percent_change_7d'] > 0
col2.dataframe(df_change)

# Conditional creation of Bar plot (time frame)
col3.subheader('Bar plot of % Price Change')

if percent_timeframe == '7d':
    if sort_values == 'Yes':
        df_change = df_change.sort_values(by=['percent_change_7d'])
    col3.write('*7 days period*')
    plt.figure(figsize=(5,10))
    plt.subplots_adjust(top = 1, bottom = 0)
    df_change['percent_change_7d'].plot(kind='barh', color=df_change.positive_percent_change_7d.map({True: 'g', False: 'r'}))
    col3.pyplot(plt)
elif percent_timeframe == '24h':
    if sort_values == 'Yes':
        df_change = df_change.sort_values(by=['percent_change_24h'])
    col3.write('*24 hour period*')
    plt.figure(figsize=(5,10))
    plt.subplots_adjust(top = 1, bottom = 0)
    df_change['percent_change_24h'].plot(kind='barh', color=df_change.positive_percent_change_24h.map({True: 'g', False: 'r'}))
    col3.pyplot(plt)
else:
    if sort_values == 'Yes':
        df_change = df_change.sort_values(by=['percent_change_1h'])
    col3.write('*1 hour period*')
    plt.figure(figsize=(5,10))
    plt.subplots_adjust(top = 1, bottom = 0)
    df_change['percent_change_1h'].plot(kind='barh', color=df_change.positive_percent_change_1h.map({True: 'g', False: 'r'}))
    col3.pyplot(plt)


#############################################  Prediction  ########################################
col2.subheader('Price prediction of BNB')
def data_cleaning(df):
  df = df.iloc[::-1].reset_index()
  df = df.drop('index', axis=1)
  df = df.drop(['unix', 'symbol', 'Volume BNB', "Volume USDT", 'tradecount'], axis=1)
  df.columns = [col.capitalize() for col in df.columns]
  df = df.fillna(df['Open'].mean())
  df['Date'] = [i[:10] for i in df['Date']]
  return df

df = pd.read_csv('Binance_BNBUSDT_d(1).csv')
df = data_cleaning(df)

cols = list(df)[1:6]
df_for_training = df[cols].astype(float)
train_dates = pd.to_datetime(df['Date'], dayfirst=True)
scaler = joblib.load('scaler.save') 
df_for_training_scaled = scaler.transform(df_for_training)

#Empty lists to be populated using formatted training data
def prep_training(df):
  trainX = []
  trainY = []
  n_future = 10   # Number of days we want to look into the future based on the past days.
  n_past = 30  # Number of past days we want to use to predict the future.

  for i in range(n_past, len(df_for_training_scaled) - n_future +1):
      trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
      trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])
  return np.array(trainX), np.array(trainY)
trainX, trainY = prep_training(df_for_training_scaled)

def prediction_fun(model,trainX,train_dates,df_for_training):
  n_days_for_prediction=30
  predict_period_dates = pd.date_range(list(train_dates)[-1], periods=n_days_for_prediction).tolist()
  #Make prediction
  prediction = model.predict(trainX[-n_days_for_prediction:])
  # prediction = prediction[30:]
  prediction_copies = np.repeat(prediction, df_for_training.shape[1], axis=-1)
  y_pred_future = scaler.inverse_transform(prediction_copies)[:,0]
  forecast_dates = []
  for time_i in predict_period_dates:
      forecast_dates.append(time_i.date())
      
  df_forecast = pd.DataFrame({'Date':np.array(forecast_dates), 'Open':y_pred_future})
  df_forecast['Date']=pd.to_datetime(df_forecast['Date'])

  original = df[['Date', 'Open']]
  original['Date']=pd.to_datetime(original['Date'], dayfirst=True)

  fig = px.line(x=original['Date'], y=original['Open'], title='Prediction of BNB', labels= {'y':'Price', 'x':'Date'})
  fig = px.line(x=df_forecast['Date'], y=df_forecast['Open'],  labels= {'y':'Price', 'x':'Date'})
  fig.update_layout(width=1000, height=500)
  st.plotly_chart(fig)
  return y_pred_future

model = load_model('BNB_model_200_epoch.h5')
y_pred_future = prediction_fun(model,trainX,train_dates,df_for_training)