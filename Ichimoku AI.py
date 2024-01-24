import os
import talib
import warnings
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.dates as mdates
from sklearn.ensemble import HistGradientBoostingRegressor

# Suppress specific warnings
warnings.filterwarnings('ignore', category=FutureWarning, message="Series.__getitem__ treating keys as positions is deprecated.*")

while True:
    # Ask the user for the stock ticker symbol
    stock_ticker = input("Enter the stock ticker symbol or 'exit' to finish: ")
    if stock_ticker.lower() == 'exit':
        break

    # Get today's date
    today = datetime.datetime.now().date()

    # Subtract 365 days from today's date
    one_year_ago = today - datetime.timedelta(days=365)

    # Use the date one year ago as the start parameter in yf.download()
    data = yf.download(stock_ticker, start=one_year_ago)

    if data.empty:
        print("No data available for the stock ticker symbol:", stock_ticker, ". Please try another symbol.")
        continue
    else:
        # Convert the date column to a datetime object
        data['Date'] = pd.to_datetime(data.index)

        # Set the date column as the index
        data.set_index('Date', inplace=True)

        # Sort the data by date
        data.sort_index(inplace=True)

        # Get the data for the last year
        last_year = data.iloc[-365:].copy()

        # Calculate the Ichimoku components
        last_year['tenkan_sen'] = talib.SMA(last_year['Close'], timeperiod=9)
        last_year['kijun_sen'] = talib.SMA(last_year['Close'], timeperiod=26)
        last_year['senkou_span_a'] = (last_year['tenkan_sen'] + last_year['kijun_sen']) / 2
        last_year['senkou_span_b'] = talib.SMA(last_year['Close'], timeperiod=52)
        last_year['chikou_span'] = talib.SMA(last_year['Close'], timeperiod=26)

        # Split the data into X (features) and y (target)
        X = last_year[['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span']]
        y = last_year['Close']

        # Create and fit the model
        model = HistGradientBoostingRegressor()
        model.fit(X, y)

        # Predictions for the next 30 days
        future_dates = pd.date_range(start=data.index[-1], periods=30, freq='D')
        future_data = pd.DataFrame(index=future_dates, columns=X.columns)
        for column in future_data.columns:
            future_data[column] = last_year[column].iloc[-1]

        predictions = model.predict(future_data)
        predictions_df = pd.DataFrame(predictions, index=future_dates, columns=['Close'])

        # Adding randomness to the predictions (not a part of actual financial models)
        std_dev = last_year['Close'].std()
        random_values = np.random.normal(0, 0.2 * std_dev, predictions.shape)
        predictions += random_values 
        predictions_df = pd.DataFrame(predictions, index=future_dates, columns=['Close'])
        predictions_df = pd.concat([last_year, predictions_df])

        # Update Ichimoku components for the new predictions
        predictions_df['tenkan_sen'] = talib.SMA(predictions_df['High'], timeperiod=9)
        predictions_df['kijun_sen'] = talib.SMA(predictions_df['High'], timeperiod=26)
        predictions_df['senkou_span_a'] = (predictions_df['tenkan_sen'] + predictions_df['kijun_sen']) / 2
        predictions_df['senkou_span_b'] = talib.SMA(predictions_df['High'], timeperiod=52)
        predictions_df['chikou_span'] = predictions_df['Close'].shift(-26)

        # Set style and create plot
        style.use('dark_background')
        fig, ax = plt.subplots()

        # Plot the predicted close prices for the next 30 days
        ax.plot(predictions_df.index, predictions_df['Close'], color='green' if predictions_df['Close'][-1] >= last_year['Close'][-1] else 'red', label='Predicted')

        # Plot the actual close prices for the last year
        ax.plot(last_year.index, last_year['Close'], color='blue', label='Actual')

        # Plot the Ichimoku Indicator lines with dynamic colors
        for i in range(len(predictions_df)):
            color_a = 'green' if predictions_df['senkou_span_a'].iloc[i] > predictions_df['senkou_span_b'].iloc[i] else 'red'
            color_b = 'red' if predictions_df['senkou_span_a'].iloc[i] > predictions_df['senkou_span_b'].iloc[i] else 'green'
            if i < len(predictions_df) - 1:
                ax.plot(predictions_df.index[i:i+2], predictions_df['senkou_span_a'].iloc[i:i+2], color=color_a)
                ax.plot(predictions_df.index[i:i+2], predictions_df['senkou_span_b'].iloc[i:i+2], color=color_b)

        # Create dummy lines for legend
        line_a, = ax.plot([None], [None], color='green', label='Senkou Span A')
        line_b, = ax.plot([None], [None], color='red', label='Senkou Span B')

        # Add the legend to the plot
        ax.legend(handles=[line_a, line_b])

        # Add semi-transparent colors for filling the area between Senkou Span A and Senkou Span B
        plt.fill_between(predictions_df.index, predictions_df['senkou_span_a'], predictions_df['senkou_span_b'], 
                         where=(predictions_df['senkou_span_a'] > predictions_df['senkou_span_b']), 
                         color='green', alpha=0.5)
        plt.fill_between(predictions_df.index, predictions_df['senkou_span_a'], predictions_df['senkou_span_b'], 
                         where=(predictions_df['senkou_span_b'] > predictions_df['senkou_span_a']), 
                         color='red', alpha=0.5)

        # Set the x-axis to display the dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)

        # Add the legend and title
        ax.legend()
        plt.title(stock_ticker.upper() + " Ichimoku Price Prediction")

        # Show the plot
        plt.show()
