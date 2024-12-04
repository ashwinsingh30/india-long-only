import pandas as pd

from backtest.config.BacktestConfig import get_pulse_platform_backtest_config

config = get_pulse_platform_backtest_config()

def process_new_buy_order(buy_orders, trade_df, trade_date):
    for index in buy_orders.index:
        trade_data = pd.Series()
        trade_data['script_name'] = index
        trade_data['trade_date'] = trade_date
        trade_data['position_type'] = "LONG"
        trade_data['position_size'] = buy_orders.loc[index]['quantity']
        trade_data['position_price'] = buy_orders.loc[index]['trigger_price']
        trade_data['portfolio_id'] = config.portfolio
        trade_data['strategy_id'] = config.strategy
        trade_data['time'] = "10:00"
        trade_df = trade_df.append(trade_data, ignore_index=True)
    return trade_df


def process_close_buy_order(close_buy_orders, trade_df, trade_date):
    for index in close_buy_orders.index:
        trade_data = pd.Series()
        trade_data['script_name'] = index
        trade_data['trade_date'] = trade_date
        trade_data['position_type'] = "CLOSE_LONG"
        trade_data['position_size'] = close_buy_orders.loc[index]['quantity']
        trade_data['position_price'] = close_buy_orders.loc[index]['trigger_price']
        trade_data['portfolio_id'] = config.portfolio
        trade_data['strategy_id'] = config.strategy
        trade_data['time'] = "15:00"
        trade_df = trade_df.append(trade_data, ignore_index=True)
    return trade_df

def process_new_sell_order(sell_orders, trade_df, trade_date):
    for index in sell_orders.index:
        trade_data = pd.Series()
        trade_data['script_name'] = index
        trade_data['trade_date'] = trade_date
        trade_data['position_type'] = "SHORT"
        trade_data['position_size'] = sell_orders.loc[index]['quantity']
        trade_data['position_price'] = sell_orders.loc[index]['trigger_price']
        trade_data['portfolio_id'] = config.portfolio
        trade_data['strategy_id'] = config.strategy
        trade_data['time'] = "10:00"
        trade_df = trade_df.append(trade_data, ignore_index=True)
    return trade_df

def process_close_sell_order(close_sell, trade_df, trade_date):
    for index in close_sell.index:
        trade_data = pd.Series()
        trade_data['script_name'] = index
        trade_data['trade_date'] = trade_date
        trade_data['position_type'] = "CLOSE_SHORT"
        trade_data['position_size'] = close_sell.loc[index]['quantity']
        trade_data['position_price'] = close_sell.loc[index]['trigger_price']
        trade_data['portfolio_id'] = config.portfolio
        trade_data['strategy_id'] = config.strategy
        trade_data['time'] = "15:00"
        trade_df = trade_df.append(trade_data, ignore_index=True)
    return trade_df


