import os
import warnings

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from backtest.config.BacktestConfig import get_pulse_platform_backtest_config
from backtest.utils.BackTestUtils import get_portfolio_stats_slippage_adjusted, plot_cumulative_returns
from config.ConfiguredLogger import get_logger
from database.finders.EquitiesPriceDataFinder import get_close_prices_for_date_security_list, \
    get_benchmark_returns_for_dates, previous_trading_day, data_scratch
from signalgeneration.AdhocSignal import get_adhoc_signal
from signalgeneration.LongOnlyFundamentalPulse import get_fundamental_long_only_pulse
from signalgeneration.PulsePlatformModelBank import india_long_only_pulse
from signalgeneration.SmallLongOnlyPulse import get_small_long_only_pulse

warnings.filterwarnings("ignore")

config = get_pulse_platform_backtest_config()
log = get_logger(os.path.basename(__file__), '/back_test.log')

start_date = config.start_date
end_date = config.end_date

capital = 10000000000


def get_average_price(data):
    weight_old = data['Weight_old']
    weight_new = data['ExposureWeights']
    old_average = data['pav_old']
    current = data['Old_Prices']
    if weight_old == 0:
        return current
    if np.sign(weight_old) != np.sign(weight_new):
        return current
    if abs(weight_new) < 0.0025:
        return np.nan
    if abs(weight_new) < abs(weight_old):
        return old_average
    else:
        if abs(weight_old) < 0.0025:
            old_average = current
        return (old_average * current * weight_new) / \
            (old_average * weight_new + weight_old * (current - old_average))


def get_portfolio_gain(signal, current_date, delta, portfolio_value):
    securities = list(signal.index)
    old_closing_prices = get_close_prices_for_date_security_list(securities,
                                                                 previous_trading_day(current_date - delta)) \
        .rename(columns={'close_price': 'Old_Prices'})
    new_closing_prices = get_close_prices_for_date_security_list(securities, current_date) \
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
    gain_data = pd.DataFrame()
    transaction_data = pd.DataFrame()
    position_data = pd.DataFrame()
    signal_data = pd.DataFrame()
    margin_account = capital
    gross_exposure = margin_account * config.leverage
    old_signal = pd.DataFrame()
    old_signal['Weight'] = np.nan
    old_signal['No_of_Shares'] = np.nan
    delta = relativedelta(days=1)
    dates = data_scratch.trade_dates
    current_month = start_date.month
    log.info(str(dates))
    for i in range(1, len(dates)):
        current_date = dates[i]
        log.info(current_date)
        if current_date.month != current_month:
            data_scratch.refresh_equities_data_scratch(current_date - relativedelta(months=1),
                                                       (current_date + relativedelta(months=1, days=5)), 5)
            current_month = current_date.month
        gain_series = pd.Series()
        optimised_signal = signal_function(dates[i - 1], old_signal['Weight'], *args)[['Weight']]
        optimised_signal['ExposureWeights'] = optimised_signal['Weight'] / optimised_signal['Weight'].abs().sum()
        leverage_fraction = optimised_signal['Weight'].abs().sum()
        leverage = config.leverage * leverage_fraction
        if not optimised_signal.empty:
            long_signal = optimised_signal[optimised_signal['Weight'] > 0]
            short_signal = optimised_signal[optimised_signal['Weight'] < 0]
            long_exposure = long_signal['ExposureWeights'].sum()
            short_exposure = short_signal['ExposureWeights'].abs().sum()
            log.info('\n' + optimised_signal[optimised_signal['Weight'] != 0].sort_values('Weight').to_string())

            long_gain = get_portfolio_gain(long_signal, current_date, delta, gross_exposure * long_exposure)
            short_gain = get_portfolio_gain(short_signal, current_date, delta, gross_exposure * short_exposure)

            securities = list(optimised_signal.index)
            old_closing_prices = get_close_prices_for_date_security_list(securities,
                                                                         previous_trading_day(current_date - delta)) \
                .rename(columns={'close_price': 'Old_Prices'})
            new_closing_prices = get_close_prices_for_date_security_list(securities, current_date) \
                .rename(columns={'close_price': 'New_Prices'})
            optimised_signal = optimised_signal.join(old_closing_prices, how='inner')
            optimised_signal = optimised_signal.join(new_closing_prices, how='inner')
            optimised_signal = optimised_signal.dropna()
            optimised_signal['ExposureWeights'] = optimised_signal['ExposureWeights'] / \
                                                  optimised_signal['ExposureWeights'].abs().sum()
            optimised_signal['Value'] = optimised_signal['ExposureWeights'] * gross_exposure
            optimised_signal['No_of_Shares'] = optimised_signal['Value'] / optimised_signal['Old_Prices']
            optimised_signal['Diff'] = (optimised_signal['New_Prices'] / optimised_signal['Old_Prices']) - 1
            optimised_signal['Date'] = current_date
            signal_data = pd.concat([signal_data, optimised_signal.reset_index()], axis=0)
            positions = optimised_signal['Value'].copy()
            positions['date'] = current_date
            positions['cash'] = margin_account
            gain = np.sum(optimised_signal['No_of_Shares'] *
                          (optimised_signal['New_Prices'] - optimised_signal['Old_Prices']))

            turnover_df = old_signal[['No_of_Shares']] \
                .join(optimised_signal[['No_of_Shares']], lsuffix='_Old', rsuffix='_New', how='outer')
            securities = list(turnover_df.index)
            prices = get_close_prices_for_date_security_list(securities, current_date) \
                .rename(columns={'close_price': 'Prices'})
            turnover_df = turnover_df.join(prices, how='outer')
            turnover_df = turnover_df.fillna(0)
            turnover_df['trade_date'] = current_date
            turnover_df['Old_Value'] = turnover_df['No_of_Shares_Old'] * turnover_df['Prices']
            turnover_df['New_Value'] = turnover_df['No_of_Shares_New'] * turnover_df['Prices']
            turnover_df['Transaction'] = turnover_df['New_Value'] - turnover_df['Old_Value']
            turnover_df['Change_in_value'] = turnover_df['Transaction'].abs()
            turnover_df['TransactionCost'] = turnover_df['Change_in_value'] * config.transaction_cost
            turnover_df['Transaction'] += turnover_df['TransactionCost']
            turnover_df['Transaction'] = turnover_df['Transaction'] / gross_exposure
            tc = np.sum(turnover_df['Change_in_value']) * config.transaction_cost
            turnover = np.sum(turnover_df['Change_in_value']) * 100 / gross_exposure

            gain_series['Date'] = current_date
            gain_series['MarginAccount'] = margin_account
            gain_series['GrossExposure'] = gross_exposure
            gain_series['LongExposure'] = gross_exposure * long_exposure
            gain_series['ShortExposure'] = gross_exposure * short_exposure
            gain_series['DailyPnLPlusMTM'] = gain
            gain_series['LongGain'] = long_gain / (gross_exposure * long_exposure)
            gain_series['ShortGain'] = short_gain / (gross_exposure * short_exposure)
            gain_series['ExposureReturn'] = (gain - tc) / gross_exposure
            gain_series['CapitalReturn'] = (gain - tc) / margin_account
            gain_series['CumulativeCapitalReturn'] = (margin_account + gain - tc - capital) / capital
            gain_series['TransactionCost'] = tc
            gain_series['Turnover'] = turnover
            gain_series['Leverage'] = leverage
            old_signal = optimised_signal
            margin_account += gain - tc
            if margin_account < config.maintainance_margin * gross_exposure:
                margin_call = (config.maintainance_margin * gross_exposure - margin_account)
                margin_account = config.maintainance_margin * gross_exposure
            else:
                margin_call = 0
            gain_series['MarginCall'] = margin_call
            gross_exposure = margin_account * leverage
            log.info("Capital Return for the period " + str(gain_series['Date']) + " " +
                     str(gain_series['CapitalReturn'] * 100))
            log.info("Exposure Return for the period " + str(gain_series['Date']) + " " +
                     str(gain_series['ExposureReturn'] * 100))
            log.info("Cumulative Capital Return " + str(gain_series['Date']) + " " +
                     str(gain_series['CumulativeCapitalReturn'] * 100))
            log.info("Portfolio Leverage " + str(gain_series['Date']) + " " +
                     str(gain_series['Leverage']))
            gain_data = pd.concat([gain_data, gain_series.to_frame().T], axis=1)

    return gain_data.set_index('Date'), signal_data


strategy_name = ''
strategy = get_small_long_only_pulse

gain_df, signal_df = backtest_performance_with_tc(strategy)
benchmark_returns = get_benchmark_returns_for_dates("NIFTY", start_date, end_date)
gain_df = gain_df.fillna(0)
log.info(strategy.__name__)
log.info("----------------------------Capital Returns Stats-------------------------")
portfolio_stats = get_portfolio_stats_slippage_adjusted(gain_df['CapitalReturn'], benchmark_returns,
                                                        gain_df['Turnover'].mean())
log.info('\n' + portfolio_stats.to_string())
log.info("----------------------------Exposure Returns Stats-------------------------")
portfolio_stats = get_portfolio_stats_slippage_adjusted(gain_df['ExposureReturn'], benchmark_returns,
                                                        gain_df['Turnover'].mean())
log.info('\n' + portfolio_stats.to_string())
plot_cumulative_returns(gain_df['CapitalReturn'], benchmark_returns)
gain_df.to_csv('ReturnsDF' + strategy_name + '.csv')
signal_df.to_csv('SignalDF' + strategy_name + '.csv', index=False)
