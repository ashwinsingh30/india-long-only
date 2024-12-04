import datetime

import pandas as pd

from database.persistence.EquitiesPriceDataPersistence import persist_equities_data
from utils.DateUtils import parse_date
from utils.TradingPlatformUtils import equities_composite_hash


def get_symbol_from_bbticker(val):
    return val.split(' ')[0]


def hash_equities(row):
    return equities_composite_hash(row['trade_date'], row['script_name'])


back_data = pd.read_csv("NSEIndices.csv", index_col=[0])
back_data = back_data[back_data['ticker'] != 'NSE200 Index']
back_data = back_data.rename(columns={'date': 'trade_date',
                                      'PX_LAST': 'last_price',
                                      'PX_OPEN': 'open_price',
                                      'PX_HIGH': 'high_price',
                                      'PX_LOW': 'low_price',
                                      'PX_VOLUME': 'volume'})

back_data.trade_date = back_data.trade_date.apply(parse_date)
back_data['script_name'] = back_data['ticker'].apply(get_symbol_from_bbticker)
back_data = back_data.drop('ticker', axis=1)
back_data['close_price'] = back_data['last_price']
back_data['equities_hash'] = back_data.apply(hash_equities, axis=1)
scripts = back_data['script_name'].unique()

transformed_df = pd.DataFrame()
for stock in scripts:
    print(stock)
    stock_data = back_data[back_data['script_name'] == stock]
    print(stock_data)
    stock_data = stock_data.sort_values('trade_date')
    stock_data.dropna(axis=0, subset=['close_price'], inplace=True)
    stock_data['previous_close'] = stock_data.shift(periods=1)['close_price']
    transformed_df = pd.concat([transformed_df, stock_data])

transformed_df['diff'] = (transformed_df['close_price'] - transformed_df['previous_close']) / transformed_df[
    'previous_close']
transformed_df['last_price'] = transformed_df['close_price']
transformed_df.set_index('equities_hash', inplace=True)
transformed_df.dropna(axis=0, subset=['volume'], inplace=True)
print(transformed_df)
persist_equities_data(transformed_df)
