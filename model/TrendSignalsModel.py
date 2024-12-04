import os

import numpy as np
import pandas as pd
from hurst import compute_Hc
from pandas import DataFrame
from scipy.stats import linregress

from config.ConfiguredLogger import get_logger
from model.EquitiesSignalProcessingModel import norm_ranked

log = get_logger(os.path.basename(__file__))


def rolling_apply(df, window, function):
    return pd.Series([df.iloc[i - window: i].pipe(function)
                      if i >= window else None
                      for i in range(1, len(df) + 1)],
                     index=df.index)


def SMA(series, period):
    return series.rolling(window=period).mean()


def momentum(ohlc: DataFrame, column: str = "close_price", period: int = 250):
    short_period = int(period / 10)
    return ohlc[column].rolling(short_period).mean() / ohlc[column].rolling(period).mean()


def volatility(ohlc: DataFrame, column: str = "close_price", period: int = 250):
    return ohlc[column].pct_change().rolling(period).std()


def volume_breakout(volumes, vol_lag=7):
    volumes = (volumes - np.mean(volumes)) / np.std(volumes)
    z = np.arctan(np.exp(np.mean(volumes[-vol_lag:])))
    return z


def find_momentum(prices, momentum_window=50):
    prices = prices[:momentum_window]
    x = np.arange(len(prices))
    prices = np.log(prices)
    slope, _, rvalue, _, _ = linregress(x, prices)
    return slope


def find_momentum2(prices, momentum_window=50):
    prices = prices[:momentum_window]
    x = np.arange(len(prices))
    prices = np.log(prices)
    slope, _, rvalue, _, _ = linregress(x, prices)
    slope2 = slope / np.std(prices)
    return slope2


def find_momentum3(data):
    o = data.open_price.values
    c = data.close_price.values
    r = (c[-1] - c[0]) / c[0]
    up_days = ((c - o) > 0).sum()
    down_days = ((c - o) < 0).sum()
    return r * (up_days - down_days)


def find_momentum4(data):
    c = data.close_price.values
    v = data.volume.values
    v_z = np.clip((v - v.mean() / v.std()), -100, 100)
    v = np.arctan(np.exp(v_z))
    z = (np.diff(c) * v[1:]) / c[1:]
    up_days_volume_weighted = z[z > 0].sum()
    down_days_volume_weighted = z[z < 0].sum()
    return up_days_volume_weighted + down_days_volume_weighted


def hurst_exponent(stock_rolling_prices):
    if len(stock_rolling_prices.index) >= 100:
        return compute_Hc(stock_rolling_prices['close_price'], kind='price', simplified=True)[0]
    else:
        return np.nan


def trend_signal_1(data, window=250):
    data['52_week_high'] = data['high_price'].rolling(window).max()
    data['away_from_52_week_high'] = data['close_price'] / (data['close_price'] - data['52_week_high']).abs()
    return SMA(np.arctan(np.exp(data['away_from_52_week_high'])), 120)


def trend_signal_2(data, window=250):
    data['52_week_low'] = data['low_price'].rolling(window).min()
    data['away_from_52_week_low'] = (-data['close_price'] + data['52_week_low']).abs() / data['close_price']
    return SMA(np.arctan(np.exp(data['away_from_52_week_low'])), 120)


def trend_signal_3(data):
    return data['trend_signal_1'] * data['trend_signal_2']


def trend_signal_4(data, window=250):
    return SMA(data['volume'].rolling(window).apply(volume_breakout), 30)


def trend_signal_5(data):
    return data['trend_signal_1'] * data['trend_signal_4']


def trend_signal_6(data):
    return data['trend_signal_2'] * data['trend_signal_4']


def trend_signal_7(data):
    return data['trend_signal_4'] * data['trend_signal_1'] * data['trend_signal_2']


def trend_signal_8(data, momentum_window=50, momentum_lag=150):
    return data['mid_price'].rolling(momentum_window + momentum_lag).apply(find_momentum)


def trend_signal_9(data):
    return data['trend_signal_8'] * data['trend_signal_4']


def trend_signal_10(data):
    return data['trend_signal_8'] * data['trend_signal_1']


def trend_signal_11(data):
    return data['trend_signal_8'] * data['trend_signal_2']


def trend_signal_12(data):
    return data['trend_signal_8'] * data['trend_signal_7']


def trend_signal_13(data, momentum_window=50, momentum_lag=150):
    return data['mid_price'].rolling(momentum_window + momentum_lag).apply(find_momentum2)


def trend_signal_14(data):
    return data['trend_signal_13'] * data['trend_signal_4']


def trend_signal_15(data):
    return data['trend_signal_13'] * data['trend_signal_1']


def trend_signal_16(data):
    return data['trend_signal_13'] * data['trend_signal_2']


def trend_signal_17(data):
    return data['trend_signal_13'] * data['trend_signal_7']


def trend_signal_18(data, window=100):
    return rolling_apply(data, window, find_momentum3)


def trend_signal_19(data, window=100):
    return (data['close_price'] - data['close_price'].shift(window)) / data['close_price'].shift(window)


def trend_signal_20(data):
    return data['trend_signal_8'] * data['hurst500']


def trend_signal_21(data):
    return data['trend_signal_13'] * data['hurst500']


def trend_signal_22(data, window_size=120):
    data['Close_MA120'] = data['close_price'].rolling(window=window_size).mean()
    data['Close MA120 1diff'] = data['Close_MA120'].diff()
    data['Indicator Increasing'] = np.where(data['Close MA120 1diff'] > 0, 1, 0)
    data['Indicator Decreasing'] = np.where(data['Close MA120 1diff'] < 0, 1, 0)
    data['Increasing days'] = data['Indicator Increasing'].rolling(window=window_size).sum()
    data['Decreasing days'] = data['Indicator Decreasing'].rolling(window=window_size).sum()
    data['Ratio'] = np.arctan(data['Increasing days'] / data['Decreasing days'])
    return data['Ratio']


def trend_signal_23(data):
    return data['trend_signal_22'] * data['hurst500']


def trend_signal_24(data):
    window_size = 120
    data['Close_MA'] = data['close_price'].rolling(window=window_size).mean()
    return data['Close_MA'].diff(150) / data['Close_MA']


def trend_signal_25(data):
    return data['trend_signal_24'] * data['hurst500']


def trend_signal_26(data):
    window_size = 120
    data['Close_Median'] = data['close_price'].rolling(window=window_size).median()
    return data['Close_Median'].diff(140) / data['Close_Median']


def trend_signal_27(data):
    return data['trend_signal_26'] * data['hurst500']


def trend_signal_28(data, window=100):
    return rolling_apply(data, window, find_momentum4)


def short_term_momentum(data):
    return data['close_price'].rolling(5).mean() / data['close_price'].rolling(50).mean()


def long_term_momentum(data):
    return data['close_price'].rolling(25).mean() / data['close_price'].rolling(250).mean()


def calculate_trends_signals(data):
    ohlcv = data.sort_values('trade_date').copy()
    ohlcv['trend_signal_1'] = trend_signal_1(ohlcv)
    ohlcv['trend_signal_2'] = trend_signal_2(ohlcv)
    ohlcv['trend_signal_3'] = trend_signal_3(ohlcv)
    ohlcv['trend_signal_4'] = trend_signal_4(ohlcv)
    ohlcv['trend_signal_5'] = trend_signal_5(ohlcv)
    ohlcv['trend_signal_6'] = trend_signal_6(ohlcv)
    ohlcv['trend_signal_7'] = trend_signal_7(ohlcv)
    ohlcv['mid_price'] = (ohlcv['open_price'] + ohlcv['close_price']) / 2
    ohlcv['trend_signal_8'] = trend_signal_8(ohlcv)
    ohlcv['trend_signal_9'] = trend_signal_9(ohlcv)
    ohlcv['trend_signal_10'] = trend_signal_10(ohlcv)
    ohlcv['trend_signal_11'] = trend_signal_11(ohlcv)
    ohlcv['trend_signal_12'] = trend_signal_12(ohlcv)
    ohlcv['trend_signal_13'] = trend_signal_13(ohlcv)
    ohlcv['trend_signal_14'] = trend_signal_14(ohlcv)
    ohlcv['trend_signal_15'] = trend_signal_15(ohlcv)
    ohlcv['trend_signal_16'] = trend_signal_16(ohlcv)
    ohlcv['trend_signal_17'] = trend_signal_17(ohlcv)
    ohlcv['trend_signal_18'] = trend_signal_18(ohlcv)
    ohlcv['trend_signal_19'] = trend_signal_19(ohlcv)
    ohlcv['hurst500'] = rolling_apply(ohlcv, 500, hurst_exponent)
    ohlcv['trend_signal_20'] = trend_signal_20(ohlcv)
    ohlcv['trend_signal_21'] = trend_signal_21(ohlcv)
    ohlcv['trend_signal_22'] = trend_signal_22(ohlcv)
    ohlcv['trend_signal_23'] = trend_signal_23(ohlcv)
    ohlcv['trend_signal_24'] = trend_signal_24(ohlcv)
    ohlcv['trend_signal_25'] = trend_signal_25(ohlcv)
    ohlcv['trend_signal_26'] = trend_signal_26(ohlcv)
    ohlcv['trend_signal_27'] = trend_signal_27(ohlcv)
    ohlcv['trend_signal_28'] = trend_signal_28(ohlcv)
    ohlcv['short_term_momentum'] = short_term_momentum(ohlcv)
    ohlcv['long_term_momentum'] = long_term_momentum(ohlcv)
    ohlcv['momentum_100'] = momentum(ohlcv, period=100)
    ohlcv['momentum_250'] = momentum(ohlcv, period=250)
    ohlcv['momentum_500'] = momentum(ohlcv, period=500)
    ohlcv['vol_250'] = volatility(ohlcv, period=250)
    ohlcv['vol_500'] = volatility(ohlcv, period=500)
    ohlcv['hurst250'] = rolling_apply(ohlcv, 250, hurst_exponent)
    return ohlcv
