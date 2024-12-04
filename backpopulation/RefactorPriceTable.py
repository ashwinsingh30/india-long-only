from database.finders.EquitiesPriceDataFinder import get_complete_price_table_between_dates
from database.persistence.EquitiesPriceDataPersistence import persist_equities_data
from utils.DateUtils import parse_date

start_date = parse_date('2008-01-01')
end_date = parse_date('2023-10-25')

price_data = get_complete_price_table_between_dates(start_date, end_date)
scripts = price_data['script_name'].unique()

for stock in scripts:
    print(stock)
    stock_data = price_data[price_data['script_name'] == stock]
    stock_data = stock_data.sort_values('trade_date')
    stock_data.dropna(axis=0, subset=['close_price'], inplace=True)
    stock_data['previous_close'] = stock_data.shift(periods=1)['close_price']
    stock_data['diff'] = (stock_data['close_price'] - stock_data['previous_close']) / stock_data['previous_close']
    stock_data['last_price'] = stock_data['close_price']
    stock_data.set_index('equities_hash', inplace=True)
    print(stock_data)
    persist_equities_data(stock_data)
