import os

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from backtest.config.BacktestConfig import get_pulse_platform_backtest_config
from backtest.utils.BackTestUtils import get_capital_for_year
from config.ConfiguredLogger import get_logger
from config.PulsePlatformConfig import get_pulse_platform_config
from database.finders.EquitiesPriceDataFinder import get_latest_price_and_signals, \
    get_historical_price_table_between_dates
from markowitzoptimization import expected_returns
from model.EquitiesSignalProcessingModel import norm_ranked, optimize_long_only_with_constraints
from model.EquitiesSignalProcessingModel import centroid_vector_from_sort
from model.InteriorPointPortfolioOptimization import OptimizerConstraints, InteriorPointPortfolioOptimization
from utils.Constants import nse_500_equities, sector_map
from utils.DateUtils import parse_date
from utils.TradingPlatformUtils import get_daily_sampled_nse_500_universe

log = get_logger(os.path.basename(__file__))
config = get_pulse_platform_config()
universe = None

if config.run_mode == 'backtest':
    test_config = get_pulse_platform_backtest_config()
    universe = get_daily_sampled_nse_500_universe(test_config.start_date - relativedelta(months=1),
                                                  test_config.end_date)


hedging_ratios = pd.read_pickle(r'D:\Project\trading-platform-longonly\analysis\HedgingRatios.pkl')


def get_latest_hedging_ratios(trade_date):
    latest_date = hedging_ratios[hedging_ratios.index < trade_date].sort_index().index[-1]
    return hedging_ratios.loc[latest_date]['hedging_ratio']


def get_long_only_pulse(trade_date, old_signal=None, data=None):
    global universe
    if config.run_mode == 'prod':
        securities = nse_500_equities
        capital = 1E7
    else:
        securities = universe[universe.trade_date == trade_date].script_name.unique()
        base_capital = 1E8
        capital = get_capital_for_year(base_capital, trade_date)
    if data is None:
        data = get_latest_price_and_signals(trade_date)
    data = data.set_index('script_name').reindex(securities)
    signal = data[['momentum_100', 'momentum_250', 'momentum_500', 'vol_250', 'vol_500', 'hurst250', 'hurst500']]. \
        apply(norm_ranked)
    signal[['vol_250', 'vol_500']] = -1 * data[['vol_250', 'vol_500']]
    signal['signal'] = signal.mean(axis=1)
    signal = signal[signal['signal'] >= signal['signal'].quantile(q=0.8)]
    optimization_constraints = OptimizerConstraints(single_stock_bound=(0, 0.025),
                                                    beta_constraint=(0.7, 1.1),
                                                    gross_sector_constraint=0.25,
                                                    turnover_constraint=0.4,
                                                    adt_constraint=0.10,
                                                    liquidity_constraint=0.10)
    signal = optimize_long_only_with_constraints(signal.index, trade_date, data, capital, optimization_constraints,
                                                 old_signal)
    signal['Price'] = data['close_price']
    return signal[['Weight', 'Price']]


def get_long_minus_benchmark_pulse(trade_date, old_signal=None, data=None):
    global universe
    if config.run_mode == 'prod':
        securities = nse_500_equities
        capital = config.portfolios['nse500_pulse']['capital']
    else:
        securities = universe[universe.trade_date == trade_date].script_name.unique()
        base_capital = config.portfolios['nse500_pulse']['capital']
        capital = get_capital_for_year(base_capital, trade_date)
    if data is None:
        data = get_latest_price_and_signals(trade_date)
    data = data.set_index('script_name').reindex(securities)
    signal = data[['momentum_100', 'momentum_250', 'momentum_500', 'vol_250', 'vol_500', 'hurst250', 'hurst500']]. \
        apply(norm_ranked)
    signal[['vol_250', 'vol_500']] = -1 * data[['vol_250', 'vol_500']]
    signal['signal'] = signal.mean(axis=1)
    signal = signal[signal['signal'] >= signal['signal'].quantile(q=0.8)]
    if old_signal is not None and 'NIFTY' in old_signal.index:
        old_signal = old_signal.drop('NIFTY')
        old_signal = old_signal / old_signal.abs().sum()
    optimization_constraints = OptimizerConstraints(single_stock_bound=(0, 0.025),
                                                    beta_constraint=(0.6, 1.1),
                                                    gross_sector_constraint=0.25,
                                                    turnover_constraint=0.4,
                                                    adt_constraint=0.10,
                                                    liquidity_constraint=0.10)
    signal = optimize_long_only_with_constraints(signal.index, trade_date, data, capital, optimization_constraints,
                                                 old_signal)
    hedging_ratio = get_latest_hedging_ratios(trade_date)
    signal['Weight'] = (1 - hedging_ratio) * signal['Weight']
    signal.loc['NIFTY', 'Weight'] = -1 * hedging_ratio
    return signal[['Weight']]
