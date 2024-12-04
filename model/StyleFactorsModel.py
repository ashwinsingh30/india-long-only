import os

import numpy as np
import pandas as pd

from config.ConfiguredLogger import get_logger
from model.EquitiesSignalProcessingModel import normalize_signal_min_max

log = get_logger(os.path.basename(__file__))


def composite_factor(signal, factors):
    factors = factors[factors != 0]
    signal = signal.copy()
    signal[factors.index] = signal[factors.index].fillna(0)
    signal[factors.name] = 0
    for factor in factors.index:
        signal[factors.name] += factors[factor] * normalize_signal_min_max(
            signal[['trade_date', 'script_name', factor]],
            factor)
    return signal[factors.name]


def populate_style_factors(signals, universe, style_factor_df):
    signals = signals.loc[np.intersect1d(universe.index, signals.index)]
    signal_df = pd.DataFrame(index=signals.index)
    signal_df['script_name'] = signals['script_name']
    signal_df['trade_date'] = signals['trade_date']
    signal_df['value'] = composite_factor(signals, style_factor_df['value'])
    signal_df['volatility'] = composite_factor(signals, style_factor_df['volatility'])
    signal_df['quality'] = composite_factor(signals, style_factor_df['quality'])
    signal_df['leverage'] = composite_factor(signals, style_factor_df['leverage'])
    signal_df['profitability'] = composite_factor(signals, style_factor_df['profitability'])
    signal_df['short_term_trend'] = composite_factor(signals, style_factor_df['short_term_trend'])
    signal_df['long_term_trend'] = composite_factor(signals, style_factor_df['long_term_trend'])
    signal_df['size'] = composite_factor(signals, style_factor_df['size'])
    signal_df['analyst_rating'] = composite_factor(signals, style_factor_df['analyst_rating'])
    signal_df['overcrowded_stocks'] = signal_df['analyst_rating'] + \
                                      composite_factor(signals, style_factor_df['falling_liquidity'])
    return signal_df
