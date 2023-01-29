import os
import talib
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.dates as mdates
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.experimental import enable_hist_gradient_boosting

# Ask the user for the stock ticker symbol
stock_ticker = input("Enter the stock ticker symbol: ")

# Download the stock data for the last year
data = yf.download(stock_ticker, start='2022-01-15')

if data.empty:
    print("No data available for the stock ticker symbol: ", stock_ticker)
else:
    # Convert the date column to a datetime object
    data['Date'] = pd.to_datetime(data.index)

    # Set the date column as the index
    data.set_index('Date', inplace=True)

    # Sort the data by date
    data.sort_index(inplace=True)

    # Get the data for the last year
    last_year = data.iloc[-365:].copy()

    # Calculate the Tenkan-sen
    last_year.loc[:,'tenkan_sen'] = talib.SMA(last_year['Close'], timeperiod=9)

    # Calculate the Kijun-sen
    last_year.loc[:,'kijun_sen'] = talib.SMA(last_year['Close'], timeperiod=26)

    # Calculate the Senkou Span A (Leading Span A)
    last_year.loc[:,'senkou_span_a'] = (last_year['tenkan_sen'] + last_year['kijun_sen']) / 2

    # Calculate the Senkou Span B (Leading Span B)
    last_year.loc[:,'senkou_span_b'] = talib.SMA(last_year['Close'], timeperiod=52)

    # Calculate the Chikou Span
    last_year.loc[:,'chikou_span'] = talib.SMA(last_year['Close'], timeperiod=26)

    # Split the data into X (features) and y (target)
    X = last_year[['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span']]
    y = last_year['Close']

    # Create an HistGradientBoostingRegressor instance
    model = HistGradientBoostingRegressor()

    # Fit the model with the data
    model.fit(X, y)

    # Make predictions for the next 30 days
    future_dates = pd.date_range(start=data.index[-1], periods=30, freq='D')
    future_data = pd.DataFrame(index=future_dates, columns=['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span'])
    future_data['tenkan_sen'] = last_year['tenkan_sen'].iloc[-1]
    future_data['kijun_sen'] = last_year['kijun_sen'].iloc[-1]
    future_data['senkou_span_a'] = last_year['senkou_span_a'].iloc[-1]
    future_data['senkou_span_b'] = last_year['senkou_span_b'].iloc[-1]
    future_data['chikou_span'] = last_year['chikou_span'].iloc[-1]

    predictions = model.predict(future_data)
    predictions_df = pd.DataFrame(predictions, index=future_dates, columns=['Close'])

    # Calculate the standard deviation of the last year's close prices
    std_dev = last_year['Close'].std()

    # Generate random values with a standard deviation of 0.5 * the last year's close prices standard deviation
    random_values = np.random.normal(0, 0.2 * std_dev, predictions.shape)

    # Add the random values to the predicted prices
    predictions += random_values 
    predictions_df = pd.DataFrame(predictions, index=future_dates, columns=['Close'])

    # Concatenate the last_year and predictions dataframes
    predictions_df = pd.concat([last_year, predictions_df])

    # Recalculate Ichimoku Indicator for the next 30 days
    predictions_df.loc[:,'tenkan_sen'] = talib.SMA(predictions_df['High'], timeperiod=9)
    predictions_df.loc[:,'kijun_sen'] = talib.SMA(predictions_df['High'], timeperiod=26)
    predictions_df.loc[:,'senkou_span_a'] = (predictions_df['tenkan_sen'] + predictions_df['kijun_sen']) / 2
    predictions_df.loc[:,'senkou_span_b'] = talib.SMA(predictions_df['High'], timeperiod=52)
    predictions_df.loc[:,'chikou_span'] = predictions_df['Close'].shift(-26)

    # Set the style to dark theme
    style.use('dark_background')

    # Create the plot
    fig, ax = plt.subplots()

    # Plot the predicted close prices for the next 30 days
    ax.plot(predictions_df.index, predictions_df['Close'], color='green' if predictions_df['Close'][-1] >= last_year['Close'][-1] else 'red', label='Predicted')

    # Plot the actual close prices for the last year
    ax.plot(last_year.index, last_year['Close'], color='blue', label='Actual')

    # Plot the Ichimoku Indicator lines
    #ax.plot(predictions_df.index, predictions_df['tenkan_sen'], label='Tenkan-sen')
    #ax.plot(predictions_df.index, predictions_df['kijun_sen'], label='Kijun-sen')
    ax.plot(predictions_df.index, predictions_df['senkou_span_a'], color='green', label='Senkou Span A')
    ax.plot(predictions_df.index, predictions_df['senkou_span_b'], color='red', label='Senkou Span B')
    #ax.plot(predictions_df.index, predictions_df['chikou_span'], label='Chikou Span')

    # Add semi-transparent colors for filling the area between Senkou Span A and Senkou Span B
    if last_year['senkou_span_a'].iloc[-1] < last_year['senkou_span_b'].iloc[-1]:
        color = 'green'

    if last_year['senkou_span_a'].iloc[-1] > last_year['senkou_span_b'].iloc[-1]:
        color = 'red'

    plt.fill_between(last_year.index, last_year['senkou_span_a'], last_year['senkou_span_b'], color=color, alpha=0.5)

    # Set the x-axis to display the dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)

    # Add the legend and title
    ax.legend()
    plt.title(f'{stock_ticker} Ichimoku Price Prediction')

    # Show the plot
    plt.show()