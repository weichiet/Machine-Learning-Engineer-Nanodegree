#%%
import numpy as np
import pandas as pd
import os
import fix_yahoo_finance as yf

def symbol_to_path(symbol, base_dir=os.path.join("data")):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))

def get_stock_indices_web(symbol, start_date, end_date):
    """ Get the stock data from Yahoo Finance and save it as CSV file"""
    df = yf.download(symbol, start_date, end_date)
    df.to_csv(symbol_to_path(symbol))
    
def read_data(symbols, dates, addSPY=True):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if addSPY and 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols = ['SPY'] + symbols

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        # Change the colume to stock index name
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])
    
    # Filling the NaN value by using 'forward fill' method, follows by 'backward fill'
    df = df.fillna(method='ffill')
    df = df.fillna(method='bfill')

    return df