import os
import warnings

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from backtest.config.BacktestConfig import get_pulse_platform_backtest_config
from config.ConfiguredLogger import get_logger
from config.PulsePlatformConfig import get_pulse_platform_config
from database.domain.PrimarySignals import PrimarySignals
from database.finders.EquitiesPriceDataFinder import get_price_with_signals_security_list_between_dates, data_scratch, \
    get_benchmark_returns_for_dates
from utils.Constants import sector_map
from utils.DateUtils import parse_date
from utils.TradingPlatformUtils import get_daily_sampled_nse_500_universe

warnings.filterwarnings("ignore")
log = get_logger(os.path.basename(__file__))
config = get_pulse_platform_config()
backtest_config = get_pulse_platform_backtest_config()

if config.run_mode == 'backtest':
    data_scratch.refresh_equities_data_scratch(backtest_config.start_date, backtest_config.end_date, look_back_days=200)


def get_rebalance_dates(start_date, end_date, trade_dates, delta=relativedelta(months=1)):
    rebalance_dates = [trade_dates[0], trade_dates[-1]]
    current = start_date
    while current <= end_date:
        rebalance_date = trade_dates[np.where(trade_dates <= current)]
        if not len(rebalance_date) == 0:
            rebalance_date = rebalance_date[-1]
            rebalance_dates.append(rebalance_date)
        current += delta
    return np.unique(rebalance_dates)


def expand_time_series_to_trade_dates(signal, trade_dates):
    signal = signal.sort_values('trade_date').copy()
    scripts = signal['script_name'].unique()
    expanded_signal = pd.DataFrame()
    for script in scripts:
        script_signal = signal[signal['script_name'] == script].copy()
        script_signal = script_signal.set_index('trade_date').sort_index()
        dates = pd.DataFrame({'trade_date': trade_dates}).set_index('trade_date')
        script_signal = script_signal.join(dates, how='outer')
        script_signal['long_weight'] = script_signal['long_weight'].shift(1)
        script_signal['long_weight_shifted'] = script_signal['long_weight'].shift(1)
        script_signal['short_weight'] = script_signal['short_weight'].shift(1)
        script_signal['short_weight_shifted'] = script_signal['short_weight'].shift(1)
        script_signal = script_signal.ffill().dropna(subset=['long_weight', 'short_weight']).fillna(0)
        expanded_signal = pd.concat([expanded_signal, script_signal.reset_index()], axis=0)
    return expanded_signal


def long_threshold(signal):
    if signal['signal_value'] >= signal['quantile_upper']:
        return 1
    return 0.0


def short_threshold(signal):
    if signal['signal_value'] <= signal['quantile_lower']:
        return 1
    return 0.0


def update_monthly_alpha_performance(start_date, end_date, alphas):
    log.info('Updating Alpha Long Only Performances')
    universe = get_daily_sampled_nse_500_universe(start_date, end_date)
    securities = universe['script_name'].unique()
    equities_data = get_price_with_signals_security_list_between_dates(securities, start_date, end_date)
    common = np.intersect1d(equities_data.index, universe.index)
    equities_data = equities_data.loc[common]
    equities_data = equities_data.dropna(subset=alphas, how='all')
    trade_dates = equities_data.trade_date.sort_values().unique()
    rebalance_dates = get_rebalance_dates(start_date, end_date, trade_dates)
    rebalance_signals = equities_data[equities_data.trade_date.isin(rebalance_dates)]
    rebalance_signals[alphas] = rebalance_signals[alphas].astype(float)
    rebalance_signals = rebalance_signals[np.append(alphas, ['script_name', 'trade_date'])]
    benchmark_returns = get_benchmark_returns_for_dates('NIFTY', start_date - relativedelta(months=1), end_date)
    long_only_alpha_performances = pd.DataFrame()
    for alpha in alphas:
        log.info(alpha)
        alpha_signal = rebalance_signals[['script_name', 'trade_date', alpha]].copy()
        alpha_signal['signal_value'] = alpha_signal[alpha]
        quantile_upper = alpha_signal.groupby('trade_date')[['signal_value']].quantile(q=0.8)
        quantile_lower = alpha_signal.groupby('trade_date')[['signal_value']].quantile(q=0.2)
        alpha_signal = alpha_signal.reset_index().set_index('trade_date')
        alpha_signal['quantile_upper'] = quantile_upper['signal_value']
        alpha_signal['quantile_lower'] = quantile_lower['signal_value']
        alpha_signal.reset_index(inplace=True)
        alpha_signal['long_weight'] = alpha_signal.apply(long_threshold, axis=1)
        alpha_signal['short_weight'] = alpha_signal.apply(short_threshold, axis=1)
        long_sum = alpha_signal.groupby('trade_date').sum()[['long_weight']]
        short_sum = alpha_signal.groupby('trade_date').sum()[['short_weight']]
        alpha_signal = alpha_signal.set_index('trade_date')
        alpha_signal['long_weight_sum'] = long_sum['long_weight'].abs()
        alpha_signal['long_weight'] = alpha_signal['long_weight'] / alpha_signal['long_weight_sum']
        alpha_signal['short_weight_sum'] = short_sum['short_weight'].abs()
        alpha_signal['short_weight'] = -1 * alpha_signal['short_weight'] / alpha_signal['short_weight_sum']
        alpha_signal = alpha_signal.reset_index()
        alpha_signal = expand_time_series_to_trade_dates(alpha_signal[['trade_date', 'script_name', 'long_weight',
                                                                       'short_weight']], trade_dates)
        alpha_signal = alpha_signal.set_index(['script_name', 'trade_date']). \
            join(equities_data.set_index(['script_name', 'trade_date'])[['diff']], how='inner').reset_index()
        alpha_signal['gross_long_return'] = alpha_signal['long_weight'] * alpha_signal['diff']
        tx_cost = 0.0015
        alpha_signal['long_turnover'] = (alpha_signal['long_weight'] - alpha_signal['long_weight_shifted']).abs()
        alpha_signal['net_long_return'] = alpha_signal['gross_long_return'] - \
                                          (alpha_signal['long_turnover'] * tx_cost)

        alpha_signal['gross_short_return'] = alpha_signal['short_weight'] * alpha_signal['diff']
        alpha_signal['short_turnover'] = (alpha_signal['short_weight'] - alpha_signal['short_weight_shifted']).abs()
        alpha_signal['net_short_return'] = alpha_signal['gross_short_return'] - \
                                           (alpha_signal['short_turnover'] * tx_cost)

        performance = alpha_signal.groupby('trade_date')[['gross_long_return', 'long_turnover',
                                                          'net_long_return', 'gross_short_return',
                                                          'short_turnover', 'net_short_return']].sum()
        performance['benchmark'] = benchmark_returns
        performance['net_return'] = (performance['net_long_return'] + performance['net_short_return']) / 2
        performance['turnover'] = (performance['long_turnover'] + performance['short_turnover']) / 2
        performance['long_minus_benchmark'] = (performance['net_long_return'] - performance['benchmark']) / 2
        performance['alpha'] = alpha
        long_only_alpha_performances = pd.concat([long_only_alpha_performances, performance])
    long_only_alpha_performances.to_csv('PrimarySignals.csv')
    long_only_alpha_performances.to_pickle('PrimarySignals.pkl')


def update_monthly_sector_regression_model_performance(start_date, end_date):
    log.info('Updating Regression Model Performances')
    universe = get_daily_sampled_nse_500_universe(start_date, end_date)
    universe = universe.reset_index().set_index('script_name').join(sector_map).reset_index().set_index('equities_hash')
    regression_weight = pd.read_csv(
        r'D:\Project\trading-platform-longonly\utils\model_configs\regression_sector_neutral.csv', index_col=[0])
    sectors = regression_weight.sector.unique()
    tx_cost = 0.0015
    model_performances = pd.DataFrame()
    for sector in sectors:
        log.info('Updating Regression Model Performances for sector ' + sector)
        sector_regression_wt = regression_weight[regression_weight['sector'] == sector]
        print(sector_regression_wt)
        sector_universe = universe[universe['sector'] == sector]
        securities = sector_universe['script_name'].unique()
        equities_data = get_price_with_signals_security_list_between_dates(securities, start_date, end_date)
        common = np.intersect1d(equities_data.index, sector_universe.index)
        equities_data = equities_data.loc[common]
        indicators = sector_regression_wt[sector_regression_wt['model_type'] == 'independent'].pulse.unique()
        hybrid_indicators = sector_regression_wt[sector_regression_wt['model_type'] == 'hybrid'].pulse.unique()
        for indicator in hybrid_indicators:
            pv_indicator = indicator.split('|')[0].replace('_conj', '')
            fundamental_indicator = indicator.split('|')[1].replace('_conj', '')
            indicators = np.append(indicators, pv_indicator)
            indicators = np.append(indicators, fundamental_indicator)
        indicators = np.unique(indicators)
        trade_dates = equities_data.trade_date.sort_values().unique()
        rebalance_dates = get_rebalance_dates(start_date, end_date, trade_dates)
        rebalance_signals = equities_data[equities_data.trade_date.isin(rebalance_dates)]
        rebalance_signals = rebalance_signals.dropna(subset=indicators, how='all')
        rebalance_signals[indicators] = rebalance_signals[indicators].astype(float)
        rebalance_signals = rebalance_signals[np.append(indicators, ['script_name', 'trade_date'])]
        benchmark_returns = get_benchmark_returns_for_dates('NIFTY', start_date - relativedelta(months=1), end_date)
        models = sector_regression_wt.model.unique()
        for model in models:
            log.info('Updating performance for ' + model)
            model_weights = sector_regression_wt[sector_regression_wt['model'] == model].set_index('pulse')
            model_type = model_weights['model_type'].unique()[0]
            if model_type == 'independent':
                model_indicators = model_weights.index
            elif model_type == 'hybrid':
                hybrids = model_weights.index
                model_indicators = []
                for model_indicator in hybrids:
                    pv_indicator = model_indicator.split('|')[0].replace('_conj', '')
                    fundamental_indicator = model_indicator.split('|')[1].replace('_conj', '')
                    model_indicators = np.append(model_indicators, pv_indicator)
                    model_indicators = np.append(model_indicators, fundamental_indicator)
            else:
                model_indicators = []
            model_indicators = np.unique(model_indicators)
            model_signal = rebalance_signals[np.append(['script_name', 'trade_date'], model_indicators)].copy()
            rank_matrix = model_signal[np.append(model_indicators, 'trade_date')]. \
                groupby('trade_date').rank(na_option='keep')
            model_signal = model_signal[['script_name', 'trade_date']].join(rank_matrix)
            mid_point = model_signal.groupby('trade_date')[model_indicators].mean()
            model_signal = model_signal.reset_index().set_index('trade_date')
            model_signal = model_signal.join(mid_point, rsuffix='_mean')
            model_signal = model_signal.reset_index()
            for indicator in model_indicators:
                model_signal[indicator] = (model_signal[indicator] - model_signal[indicator + '_mean']).fillna(0)
                model_signal = model_signal.drop(indicator + '_mean', axis=1)
            if model_type == 'hybrid':
                for hybrid_alpha in model_weights.index:
                    pv_alpha = hybrid_alpha.split('|')[0]
                    fundamental_alpha = hybrid_alpha.split('|')[1]
                    if '_conj' in pv_alpha:
                        model_signal[hybrid_alpha] = (-1 * model_signal[pv_alpha.replace('_conj', '')] +
                                                      model_signal[fundamental_alpha]) / 2
                    else:
                        model_signal[hybrid_alpha] = (model_signal[pv_alpha] + model_signal[fundamental_alpha]) / 2
                model_indicators = model_weights.index
            abs_sum = model_signal.copy()
            abs_sum[model_indicators] = abs_sum[model_indicators].abs()
            abs_sum = abs_sum.groupby('trade_date')[model_indicators].sum()
            model_signal = model_signal.set_index('trade_date').join(abs_sum, rsuffix='_normaliser')
            model_signal = model_signal.reset_index()
            for indicator in model_indicators:
                model_signal[indicator] = (model_signal[indicator] / model_signal[indicator + '_normaliser']).fillna(0)
                model_signal = model_signal.drop(indicator + '_normaliser', axis=1)
            model_signal['weight'] = model_signal[model_indicators]. \
                dot(model_weights.reindex(model_indicators)['long_short'])
            model_signal.loc[model_signal['weight'] > 0, 'long_weight'] = model_signal['weight']
            model_signal.loc[model_signal['weight'] < 0, 'short_weight'] = model_signal['weight']
            model_signal[['long_weight', 'short_weight']] = model_signal[['long_weight', 'short_weight']].fillna(0)
            long_sum = model_signal.groupby('trade_date')['long_weight'].sum()
            short_sum = model_signal.groupby('trade_date')['short_weight'].sum()
            model_signal = model_signal.set_index('trade_date')
            model_signal['long_weight_sum'] = long_sum
            model_signal['long_weight'] = model_signal['long_weight'] / model_signal['long_weight_sum']
            model_signal['short_weight_sum'] = short_sum.abs()
            model_signal['short_weight'] = model_signal['short_weight'] / model_signal['short_weight_sum']
            model_signal = model_signal.reset_index()
            model_signal = expand_time_series_to_trade_dates(model_signal[['trade_date', 'script_name', 'long_weight',
                                                                           'short_weight']], trade_dates)
            model_signal = model_signal.set_index(['script_name', 'trade_date']). \
                join(equities_data.set_index(['script_name', 'trade_date'])[['diff']], how='inner').reset_index()
            model_signal['gross_long_return'] = model_signal['long_weight'] * model_signal['diff']
            model_signal['long_turnover'] = (model_signal['long_weight'] - model_signal['long_weight_shifted']).abs()
            model_signal['net_long_return'] = model_signal['gross_long_return'] - \
                                              (model_signal['long_turnover'] * tx_cost)

            model_signal['gross_short_return'] = model_signal['short_weight'] * model_signal['diff']
            model_signal['short_turnover'] = (model_signal['short_weight'] - model_signal['short_weight_shifted']).abs()
            model_signal['net_short_return'] = model_signal['gross_short_return'] - \
                                               (model_signal['short_turnover'] * tx_cost)
            performance = model_signal.groupby('trade_date')[['gross_long_return', 'long_turnover',
                                                              'net_long_return', 'gross_short_return',
                                                              'short_turnover', 'net_short_return']].sum()
            performance['benchmark'] = benchmark_returns
            performance['net_return'] = (performance['net_long_return'] + performance['net_short_return']) / 2
            performance['turnover'] = (performance['long_turnover'] + performance['short_turnover']) / 2
            performance['long_minus_benchmark'] = (performance['net_long_return'] - performance['benchmark']) / 2
            performance['model'] = model
            performance['sector'] = sector
            model_performances = pd.concat([model_performances, performance])
    model_performances.to_csv('SectorPerformances.csv')
    model_performances.to_pickle('SectorPerformances.pkl')


def update_monthly_regression_model_performance(start_date, end_date):
    log.info('Updating Regression Model Monthly Rebalanced Performances')
    universe = get_daily_sampled_nse_500_universe(start_date, end_date)
    securities = universe['script_name'].unique()
    equities_data = get_price_with_signals_security_list_between_dates(securities, start_date, end_date)
    common = np.intersect1d(equities_data.index, universe.index)
    equities_data = equities_data.loc[common]
    sector_regression_wt = \
        pd.read_csv(r'D:\Project\trading-platform-longonly\utils\model_configs\monthly_regression.csv')
    indicators = sector_regression_wt[sector_regression_wt['model_type'] == 'independent'].pulse.unique()
    hybrid_indicators = sector_regression_wt[sector_regression_wt['model_type'] == 'hybrid'].pulse.unique()
    for indicator in hybrid_indicators:
        pv_indicator = indicator.split('|')[0].replace('_conj', '')
        fundamental_indicator = indicator.split('|')[1].replace('_conj', '')
        indicators = np.append(indicators, pv_indicator)
        indicators = np.append(indicators, fundamental_indicator)
    indicators = np.unique(indicators)
    trade_dates = equities_data.trade_date.sort_values().unique()
    rebalance_dates = get_rebalance_dates(start_date, end_date, trade_dates)
    rebalance_signals = equities_data[equities_data.trade_date.isin(rebalance_dates)]
    rebalance_signals = rebalance_signals.dropna(subset=indicators, how='all')
    rebalance_signals[indicators] = rebalance_signals[indicators].astype(float)
    rebalance_signals = rebalance_signals[np.append(indicators, ['script_name', 'trade_date'])]
    benchmark_returns = get_benchmark_returns_for_dates('NIFTY', start_date - relativedelta(months=1), end_date)
    models = sector_regression_wt.model.unique()
    model_performances = pd.DataFrame()
    for model in models:
        log.info('Updating performance for ' + model)
        model_weights = sector_regression_wt[sector_regression_wt['model'] == model].set_index('pulse')
        model_type = model_weights['model_type'].unique()[0]
        if model_type == 'independent':
            model_indicators = model_weights.index
        elif model_type == 'hybrid':
            hybrids = model_weights.index
            model_indicators = []
            for model_indicator in hybrids:
                pv_indicator = model_indicator.split('|')[0].replace('_conj', '')
                fundamental_indicator = model_indicator.split('|')[1].replace('_conj', '')
                model_indicators = np.append(model_indicators, pv_indicator)
                model_indicators = np.append(model_indicators, fundamental_indicator)
        else:
            model_indicators = []
        model_indicators = np.unique(model_indicators)
        model_signal = rebalance_signals[np.append(['script_name', 'trade_date'], model_indicators)].copy()
        rank_matrix = model_signal[np.append(model_indicators, 'trade_date')]. \
            groupby('trade_date').rank(na_option='keep')
        model_signal = model_signal[['script_name', 'trade_date']].join(rank_matrix)
        mid_point = model_signal.groupby('trade_date')[model_indicators].mean()
        model_signal = model_signal.reset_index().set_index('trade_date')
        model_signal = model_signal.join(mid_point, rsuffix='_mean')
        model_signal = model_signal.reset_index()
        for indicator in model_indicators:
            model_signal[indicator] = (model_signal[indicator] - model_signal[indicator + '_mean']).fillna(0)
            model_signal = model_signal.drop(indicator + '_mean', axis=1)
        if model_type == 'hybrid':
            for hybrid_alpha in model_weights.index:
                pv_alpha = hybrid_alpha.split('|')[0]
                fundamental_alpha = hybrid_alpha.split('|')[1]
                if '_conj' in pv_alpha:
                    model_signal[hybrid_alpha] = (-1 * model_signal[pv_alpha.replace('_conj', '')] +
                                                  model_signal[fundamental_alpha]) / 2
                else:
                    model_signal[hybrid_alpha] = (model_signal[pv_alpha] + model_signal[fundamental_alpha]) / 2
            model_indicators = model_weights.index
        abs_sum = model_signal.copy()
        abs_sum[model_indicators] = abs_sum[model_indicators].abs()
        abs_sum = abs_sum.groupby('trade_date')[model_indicators].sum()
        model_signal = model_signal.set_index('trade_date').join(abs_sum, rsuffix='_normaliser')
        model_signal = model_signal.reset_index()
        tx_cost = 0.0015
        for indicator in model_indicators:
            model_signal[indicator] = (model_signal[indicator] / model_signal[indicator + '_normaliser']).fillna(0)
            model_signal = model_signal.drop(indicator + '_normaliser', axis=1)
        model_signal['weight'] = model_signal[model_indicators]. \
            dot(model_weights.reindex(model_indicators)['long_short'])
        model_signal.loc[model_signal['weight'] > 0, 'long_weight'] = model_signal['weight']
        model_signal.loc[model_signal['weight'] < 0, 'short_weight'] = model_signal['weight']
        model_signal[['long_weight', 'short_weight']] = model_signal[['long_weight', 'short_weight']].fillna(0)
        long_sum = model_signal.groupby('trade_date')['long_weight'].sum()
        short_sum = model_signal.groupby('trade_date')['short_weight'].sum()
        model_signal = model_signal.set_index('trade_date')
        model_signal['long_weight_sum'] = long_sum
        model_signal['long_weight'] = model_signal['long_weight'] / model_signal['long_weight_sum']
        model_signal['short_weight_sum'] = short_sum.abs()
        model_signal['short_weight'] = model_signal['short_weight'] / model_signal['short_weight_sum']
        model_signal = model_signal.reset_index()
        model_signal = expand_time_series_to_trade_dates(model_signal[['trade_date', 'script_name', 'long_weight',
                                                                       'short_weight']], trade_dates)
        model_signal = model_signal.set_index(['script_name', 'trade_date']). \
            join(equities_data.set_index(['script_name', 'trade_date'])[['diff']], how='inner').reset_index()
        model_signal['gross_long_return'] = model_signal['long_weight'] * model_signal['diff']
        model_signal['long_turnover'] = (model_signal['long_weight'] - model_signal['long_weight_shifted']).abs()
        model_signal['net_long_return'] = model_signal['gross_long_return'] - \
                                          (model_signal['long_turnover'] * tx_cost)

        model_signal['gross_short_return'] = model_signal['short_weight'] * model_signal['diff']
        model_signal['short_turnover'] = (model_signal['short_weight'] - model_signal['short_weight_shifted']).abs()
        model_signal['net_short_return'] = model_signal['gross_short_return'] - \
                                           (model_signal['short_turnover'] * tx_cost)
        performance = model_signal.groupby('trade_date')[['gross_long_return', 'long_turnover',
                                                          'net_long_return', 'gross_short_return',
                                                          'short_turnover', 'net_short_return']].sum()
        performance['benchmark'] = benchmark_returns
        performance['net_return'] = (performance['net_long_return'] + performance['net_short_return']) / 2
        performance['turnover'] = (performance['long_turnover'] + performance['short_turnover']) / 2
        performance['long_minus_benchmark'] = (performance['net_long_return'] - performance['benchmark']) / 2
        performance['model'] = model
        model_performances = pd.concat([model_performances, performance])
    model_performances.to_csv('FundamentalLongOnly.csv')
    model_performances.to_pickle('FundamentalLongOnly.pkl')


# update_monthly_alpha_performance(parse_date('2012-01-01'), parse_date('2023-08-31'),
#                                  np.setdiff1d(PrimarySignals.__table__.columns.keys(),
#                                               ['equities_hash', 'script_name', 'trade_date']))
update_monthly_sector_regression_model_performance(parse_date('2015-01-01'), parse_date('2024-11-30'))
