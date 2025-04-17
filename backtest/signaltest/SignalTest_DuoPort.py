import os
import warnings

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from signalgeneration.PulsePlatformModelBank import india_long_only_pulse, india_small_long_only_pulse

warnings.filterwarnings("ignore")
from backtest.config.BacktestConfig import get_pulse_platform_backtest_config
from backtest.utils.BackTestUtils import get_portfolio_stats, plot_cumulative_returns
from config.ConfiguredLogger import get_logger
from database.finders.EquitiesPriceDataFinder import get_close_prices_for_date_security_list, \
    data_scratch, previous_trading_day, \
    get_historical_price_table_between_dates, get_benchmark_returns_for_dates

config = get_pulse_platform_backtest_config()
log = get_logger(os.path.basename(__file__), '/back_test.log')

start_date = config.start_date
end_date = config.end_date

capital = 10000000000


def find_nearest_date(dates, date):
    return dates[dates[np.where(dates <= date)].argmax()]


def get_portfolio_gain(signal, current_date, delta, portfolio_value):
    securities = list(signal.index)
    old_closing_prices = get_close_prices_for_date_security_list(securities,
                                                                 previous_trading_day(current_date + delta)) \
        .rename(columns={'close_price': 'Old_Prices'})
    new_closing_prices = get_close_prices_for_date_security_list(securities,
                                                                 previous_trading_day(current_date + delta)) \
        .rename(columns={'close_price': 'New_Prices'})
    signal = signal.join(old_closing_prices, how='inner')
    signal = signal.join(new_closing_prices, how='inner')
    signal = signal.dropna()

    signal['Weight'] = signal['Weight'] / signal['Weight'].abs().sum()
    signal['Value'] = signal['Weight'] * portfolio_value
    signal['No_of_Shares'] = signal['Value'] / signal['Old_Prices']

    return np.sum(signal['No_of_Shares'] *
                  (signal['New_Prices'] - signal['Old_Prices']))


def backtest_performance_with_tc(signal_function, *args):
    if config.leverage > (1 / config.margin_rate):
        log.info("Excessive Leverage - Max Levergae possible at margin rate = " + str(1 / config.margin_rate))
        exit()
    margin_account = capital
    gross_exposure = margin_account * config.leverage
    old_signal = pd.DataFrame()
    old_signal['momentum_signal'] = np.nan
    old_signal['fundamental_signal'] = np.nan
    old_signal['Weight'] = np.nan
    old_signal['ExposureWeights'] = np.nan
    old_signal['No_of_Shares'] = np.nan
    delta = relativedelta(months=1)
    dates = data_scratch.trade_dates
    current = start_date
    current_month = start_date.month
    log.info(str(dates))
    performances = pd.DataFrame()
    signal = pd.DataFrame()
    while current <= end_date:
        log.info(current)
        if current.month != current_month:
            data_scratch.refresh_equities_data_scratch(current - relativedelta(months=1),
                                                       (current + relativedelta(months=2, days=5)), 5)
            current_month = current.month
        current_date = previous_trading_day(current)
        next_re_balance_date = previous_trading_day(current + delta)
        optimised_signal = signal_function(current_date, old_signal[['momentum_signal', 'fundamental_signal', 'Weight']],
                                           *args)[['Weight', 'momentum_signal', 'fundamental_signal']]
        optimised_signal['ExposureWeights'] = optimised_signal['Weight'] / optimised_signal['Weight'].abs().sum()
        if not optimised_signal.empty:
            securities = list(optimised_signal.index)
            price_df = get_historical_price_table_between_dates(securities, current_date,
                                                                next_re_balance_date).sort_index()
            price_df = price_df.ffill().dropna(how='all').dropna(axis=1)
            securities = np.intersect1d(price_df.columns, securities)
            optimised_signal = optimised_signal.loc[securities]
            current_signal = optimised_signal.copy()
            current_signal['trade_date'] = current
            signal = pd.concat([signal, current_signal], axis=0)
            long_signal = optimised_signal[optimised_signal['Weight'] > 0]
            short_signal = optimised_signal[optimised_signal['Weight'] < 0]
            long_exposure = long_signal['ExposureWeights'].sum()
            short_exposure = short_signal['ExposureWeights'].abs().sum()
            large_weight = optimised_signal['Weight'].abs().sort_values(ascending=False).head(10)
            log.info('\n' + optimised_signal.loc[large_weight.index].sort_values('Weight').to_string())
            current_price = price_df.iloc[0].T
            long_portfolio = (long_signal['ExposureWeights'] * gross_exposure *
                              long_exposure) / current_price.reindex(long_signal.index)
            short_portfolio = short_signal['ExposureWeights'] * gross_exposure * \
                              short_exposure / current_price.reindex(short_signal.index)
            if len(price_df) > 1:
                price_df_change = price_df.diff().fillna(0).iloc[1:]
                long_portfolio_value = price_df.reindex(long_portfolio.index, axis=1).dot(long_portfolio).shift()
                short_portfolio_value = price_df.reindex(short_portfolio.index, axis=1).dot(short_portfolio).shift()
                total_portfolio_value = long_portfolio_value + short_portfolio_value.abs()
                long_performance = price_df_change.reindex(long_portfolio.index, axis=1).dot(long_portfolio)
                short_performance = price_df_change.reindex(short_portfolio.index, axis=1).dot(short_portfolio)
                total_performance = long_performance + short_performance
                long_performance = (long_performance / long_portfolio_value).dropna()
                short_performance = (short_performance / short_portfolio_value).dropna()
                total_performance = (total_performance / total_portfolio_value).dropna()
                turnover_df = optimised_signal[['ExposureWeights']].join(old_signal[['ExposureWeights']], how='outer',
                                                                         rsuffix='_old').fillna(0)
                turnover = (turnover_df['ExposureWeights'] - turnover_df['ExposureWeights_old']).abs().sum()
                tc = turnover * config.transaction_cost
                total_performance.iloc[0] = total_performance.iloc[0] - tc
            else:
                price_df_change = pd.DataFrame()
                long_performance = pd.Series()
                short_performance = pd.Series()
                total_performance = pd.Series()
                turnover = pd.Series()
                tc = 0
            performance_df = pd.DataFrame(index=price_df_change.index)
            performance_df['LongGain'] = long_performance
            performance_df['ShortGain'] = short_performance
            performance_df['ExposureReturn'] = total_performance
            performance_df['Turnover'] = 0
            performance_df['TransactionCost'] = 0
            if not performance_df.empty:
                performance_df['Turnover'].iloc[0] = turnover
                performance_df['TransactionCost'].iloc[0] = tc
            performances = pd.concat([performances, performance_df], axis=0)
            log.info("Exposure Return for the period " + str(current) + " " +
                     str(performance_df['ExposureReturn'].sum() * 100))
            log.info("Cumulative Return " + str(current) + " " +
                     str(performances['ExposureReturn'].sum() * 100))
            old_signal = optimised_signal
        current = current + delta
    return performances, signal


strategy_name = ''
strategy = india_small_long_only_pulse

gain_df, signal_df = backtest_performance_with_tc(strategy)
benchmark_returns = get_benchmark_returns_for_dates("NSE500", start_date, end_date)
benchmark_returns = benchmark_returns.reindex(gain_df.index)
log.info(strategy.__name__)
log.info("----------------------------Exposure Returns Stats-------------------------")
portfolio_stats = get_portfolio_stats(gain_df['ExposureReturn'], benchmark_returns)
log.info('\n' + portfolio_stats.to_string())
plot_cumulative_returns(gain_df['ExposureReturn'], benchmark_returns)
gain_df['BenchmarkReturn'] = benchmark_returns
gain_df.to_csv('ReturnsDF' + strategy_name + '.csv')
signal_df.to_csv('SignalDF' + strategy_name + '.csv')
