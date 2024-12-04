import os

import numpy as np
import pandas as pd

from config.ConfiguredLogger import get_logger
from database.finders.EquitiesPriceDataFinder import get_latest_price_and_signals_securities
from model.EquitiesSignalProcessingModel import create_neutralization_membership_matrix, \
    centroid_vector_from_sort, select_long_stocks, \
    select_short_stocks
from utils.Constants import sector_map, nse_500_equities

log = get_logger(os.path.basename(__file__))


def create_signal_from_membership_matrix(signal, alpha_weights, membership_matrix, sectors, signal_type='long_short'):
    members = ['universe', 'sector_neutral_signal', 'beta_neutral_signal']
    weight_df = pd.DataFrame(0, index=membership_matrix.index, columns=members)
    for alpha in signal.columns:
        membership_signal = membership_matrix.multiply(signal[alpha], axis='index')
        membership_signal = membership_signal.rank(na_option='keep')
        membership_signal = -1 * membership_signal.sub(membership_signal.mean())
        if not membership_signal.dropna(axis=1, how='all').empty:
            membership_signal.fillna(0, inplace=True)
            membership_signal = membership_signal / membership_signal.abs().sum()
            membership_signal['universe'] = membership_signal['universe']
            membership_signal['sector_neutral_signal'] = membership_signal[sectors].mean(axis=1)
            membership_signal['beta_neutral_signal'] = (membership_signal['high_beta'] +
                                                        membership_signal['low_beta']) / 2
            if signal_type == 'long_short':
                membership_signal[members] = alpha_weights[alpha] * membership_signal[members]
            if signal_type == 'long':
                membership_signal[members] = membership_signal[members].apply(select_long_stocks,
                                                                              args=(alpha_weights[alpha],), axis=0)
            if signal_type == 'short':
                membership_signal[members] = membership_signal[members].apply(select_short_stocks,
                                                                              args=(alpha_weights[alpha],), axis=0)
            weight_df['universe'] += membership_signal['universe']
            weight_df['sector_neutral_signal'] += membership_signal['sector_neutral_signal']
            weight_df['beta_neutral_signal'] += membership_signal['beta_neutral_signal']

    weight_df = weight_df / weight_df.abs().sum()
    return weight_df.mean(axis=1)


def create_hybrid_signal_from_membership_matrix(signal, alpha_weights, membership_matrix, sectors,
                                                signal_type='long_short'):
    members = ['universe', 'sector_neutral_signal', 'beta_neutral_signal']
    weight_df = pd.DataFrame(0, index=membership_matrix.index, columns=members)
    for alpha in alpha_weights.index:
        pv_alpha = alpha.split('|')[0]
        fundamental_alpha = alpha.split('|')[1].replace('_conj', '')
        if '_conj' in pv_alpha:
            pv_alpha = pv_alpha.replace('_conj', '')
            multiplier = -1
        else:
            multiplier = 1
        pv_signal = membership_matrix.multiply(signal[pv_alpha], axis='index')
        pv_signal = pv_signal.rank(na_option='keep')
        pv_signal = -1 * multiplier * pv_signal.sub(pv_signal.mean())
        fundamental_signal = membership_matrix.multiply(signal[fundamental_alpha], axis='index')
        fundamental_signal = fundamental_signal.rank(na_option='keep')
        fundamental_signal = fundamental_signal.sub(fundamental_signal.mean())
        membership_signal = (pv_signal + fundamental_signal) / 2
        if not membership_signal.dropna(axis=1, how='all').empty:
            membership_signal.fillna(0, inplace=True)
            membership_signal = membership_signal / membership_signal.abs().sum()
            membership_signal['universe'] = membership_signal['universe']
            membership_signal['sector_neutral_signal'] = membership_signal[sectors].mean(axis=1)
            membership_signal['beta_neutral_signal'] = (membership_signal['high_beta'] +
                                                        membership_signal['low_beta']) / 2
            if signal_type == 'long_short':
                membership_signal[members] = alpha_weights[alpha] * membership_signal[members]
            if signal_type == 'long':
                membership_signal[members] = membership_signal[members].apply(select_long_stocks,
                                                                              args=(alpha_weights[alpha],), axis=0)
            if signal_type == 'short':
                membership_signal[members] = membership_signal[members].apply(select_short_stocks,
                                                                              args=(alpha_weights[alpha],), axis=0)
            weight_df['universe'] += membership_signal['universe']
            weight_df['sector_neutral_signal'] += membership_signal['sector_neutral_signal']
            weight_df['beta_neutral_signal'] += membership_signal['beta_neutral_signal']
    weight_df = weight_df / weight_df.abs().sum()
    return weight_df.mean(axis=1)


def get_basket_auxxere_pulse(trade_date, securities, pulse_df, old_signal=None, data=None):
    if data is None:
        equities_data = get_latest_price_and_signals_securities(securities, trade_date)
    else:
        equities_data = data.copy()
        equities_data = equities_data[equities_data.script_name.isin(securities)]
    equities_data = equities_data.set_index('script_name')
    indicators = np.intersect1d(equities_data.columns, pulse_df.index)
    equities_data = equities_data.dropna(subset=indicators, how='all')
    equities_data[indicators] = equities_data[indicators].astype(float)
    pulse_df = pulse_df.loc[indicators]
    membership_matrix = create_neutralization_membership_matrix(equities_data, sector_map, securities,
                                                                ['beta'])
    long_short_signal = pulse_df['long_short']
    long_short_signal = long_short_signal[long_short_signal != 0]
    long_short_weight = create_signal_from_membership_matrix(equities_data[long_short_signal.index], long_short_signal,
                                                             membership_matrix, sector_map.unique())

    rank_vector = long_short_weight.rank(ascending=False)
    weight = rank_vector.apply(centroid_vector_from_sort, args=(len(rank_vector.dropna().index),))
    weight = weight / weight.abs().sum()
    optimization_matrix = equities_data[['beta', 'adt']]
    optimization_matrix['Weight'] = weight
    optimization_matrix['sector_map'] = sector_map
    optimization_matrix['old_signal'] = pd.Series(old_signal, index=optimization_matrix.index,
                                                  dtype='float64').fillna(0)
    optimization_matrix['Weight'] = optimization_matrix['Weight'].fillna(0)
    return optimization_matrix


def get_hybrid_basket_auxxere_pulse(trade_date, securities, pulse_df, old_signal=None, data=None):
    if data is None:
        equities_data = get_latest_price_and_signals_securities(securities, trade_date)
    else:
        equities_data = data.copy()
        equities_data = equities_data[equities_data.script_name.isin(securities)]
    equities_data = equities_data.set_index('script_name')
    hybrid_indicators = pulse_df.index
    indicators = set([])
    for indicator in hybrid_indicators:
        pv_indicator = indicator.split('|')[0].replace('_conj', '')
        fundamental_indicator = indicator.split('|')[1].replace('_conj', '')
        indicators.add(pv_indicator)
        indicators.add(fundamental_indicator)
    indicators.add('momentum')
    indicators = list(indicators)
    indicators = np.intersect1d(equities_data.columns, indicators)
    equities_data = equities_data.dropna(subset=indicators, how='all')
    equities_data[indicators] = equities_data[indicators].astype(float)
    membership_matrix = create_neutralization_membership_matrix(equities_data, sector_map, securities,
                                                                ['beta'])
    long_short_signal = pulse_df['long_short']
    long_short_signal = long_short_signal[long_short_signal != 0]
    long_short_weight = create_hybrid_signal_from_membership_matrix(equities_data[indicators],
                                                                    long_short_signal, membership_matrix,
                                                                    sector_map.unique())

    rank_vector = long_short_weight.rank(ascending=False)
    weight = rank_vector.apply(centroid_vector_from_sort, args=(len(rank_vector.dropna().index),))
    weight = weight / weight.abs().sum()

    optimization_matrix = equities_data[['beta', 'adt']]
    optimization_matrix['Weight'] = weight
    optimization_matrix['sector_map'] = sector_map
    optimization_matrix['old_signal'] = pd.Series(old_signal, index=optimization_matrix.index, dtype='float64').fillna(
        0)
    optimization_matrix['Weight'] = optimization_matrix['Weight'].fillna(0)
    return optimization_matrix


def get_factor_pulse(trade_date, securities, factors, data=None):
    if data is None:
        equities_data = get_latest_price_and_signals_securities(securities, trade_date)
    else:
        equities_data = data.copy()
        equities_data = equities_data[equities_data.script_name.isin(securities)]
    equities_data = equities_data.set_index('script_name')
    indicators = np.intersect1d(equities_data.columns, factors)
    equities_data = equities_data.dropna(subset=indicators, how='all')
    equities_data[indicators] = equities_data[indicators].astype(float)
    membership_matrix = create_neutralization_membership_matrix(equities_data, sector_map, securities, ['beta'])
    signal_matrix = pd.DataFrame(index=securities)
    for factor in factors:
        signal_matrix[factor] = create_signal_from_membership_matrix(equities_data[[factor]],
                                                                     pd.Series({factor: -1}),
                                                                     membership_matrix, sector_map.unique())
    return signal_matrix


def get_long_only_factor_pulse(trade_date, securities, factors, data=None):
    if data is None:
        equities_data = get_latest_price_and_signals_securities(securities, trade_date)
    else:
        equities_data = data.copy()
        equities_data = equities_data[equities_data.script_name.isin(securities)]
    equities_data = equities_data.set_index('script_name')
    indicators = np.intersect1d(equities_data.columns, factors)
    equities_data = equities_data.dropna(subset=indicators, how='all')
    equities_data[indicators] = equities_data[indicators].astype(float)
    membership_matrix = create_neutralization_membership_matrix(equities_data, sector_map, securities,
                                                                ['beta'])
    signal_matrix = pd.DataFrame(index=securities)
    for factor in factors:
        signal_matrix[factor] = create_signal_from_membership_matrix(equities_data[[factor]],
                                                                     pd.Series({factor: -1}),
                                                                     membership_matrix, sector_map.unique())
        signal_matrix.loc[signal_matrix[factor] < 0, factor] = 0
        signal_matrix[factor] = signal_matrix[factor] / signal_matrix[factor].abs().sum()
    return signal_matrix
