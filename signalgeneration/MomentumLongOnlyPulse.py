import os

import pandas as pd
from dateutil.relativedelta import relativedelta

from backtest.config.BacktestConfig import get_pulse_platform_backtest_config
from backtest.utils.BackTestUtils import get_capital_for_year
from config.ConfiguredLogger import get_logger
from config.PulsePlatformConfig import get_pulse_platform_config
from database.finders.EquitiesPriceDataFinder import get_latest_price_and_signals
from model.EquitiesSignalProcessingModel import norm_ranked, optimize_long_only_with_constraints, \
    optimize_long_only_active
from model.InteriorPointPortfolioOptimization import OptimizerConstraints
from signalgeneration.AuxxerePulse import get_long_only_factor_pulse, get_factor_pulse
from utils.Constants import nse_500_equities
from utils.TradingPlatformUtils import get_daily_sampled_nse_500_universe

log = get_logger(os.path.basename(__file__))
config = get_pulse_platform_config()
universe = None

if config.run_mode == 'backtest':
    test_config = get_pulse_platform_backtest_config()
    universe = get_daily_sampled_nse_500_universe(test_config.start_date - relativedelta(months=1),
                                                  test_config.end_date)


def get_momentum_long_only_pulse(trade_date, old_signal=None, data=None):
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
    signal = data[['mom_100', 'mom_250', 'mom_500', 'vol_250', 'vol_500', 'hurst250', 'hurst500']]. \
        apply(norm_ranked)
    signal[['vol_250', 'vol_500']] = -1 * data[['vol_250', 'vol_500']]
    signal['signal'] = signal.mean(axis=1)
    signal = signal[signal['signal'] >= signal['signal'].quantile(q=0.8)]
    optimization_constraints = OptimizerConstraints(single_stock_bound=(0, 0.05),
                                                    beta_constraint=(0.8, 1.),
                                                    gross_sector_constraint=0.25,
                                                    turnover_constraint=0.4,
                                                    adt_constraint=0.10,
                                                    liquidity_constraint=0.10,
                                                    )
    signal = optimize_long_only_with_constraints(signal.index, trade_date, data, capital, optimization_constraints,
                                                 old_signal)
    signal['Price'] = data['close_price']
    return signal[['Weight', 'Price']]


def get_small_momentum_long_only_pulse(trade_date, max_value, old_signal=None, data=None, single_stock_limit=0.1):
    global universe
    if config.run_mode == 'prod':
        securities = nse_500_equities
    else:
        securities = universe[universe.trade_date == trade_date].script_name.unique()
    if data is None:
        data = get_latest_price_and_signals(trade_date)
    q_cut = (4 / 3) * single_stock_limit + (23 / 30)
    raw_data = data.copy().set_index('script_name').reindex(securities)
    raw_data['shares_possible'] = (max_value / raw_data['close_price'])
    raw_data = raw_data[raw_data['shares_possible'] > 1]
    signal = raw_data[['mom_100', 'mom_250', 'mom_500', 'vol_250', 'vol_500', 'hurst250', 'hurst500']]. \
        apply(norm_ranked)
    signal[['vol_250', 'vol_500']] = -1 * raw_data[['vol_250', 'vol_500']]
    signal['signal'] = signal.fillna(0).mean(axis=1)
    signal = signal[signal['signal'] >= signal['signal'].quantile(q=q_cut)]
    optimization_constraints = OptimizerConstraints(single_stock_bound=(0, single_stock_limit),
                                                    beta_constraint=(0.5, 1.1),
                                                    gross_sector_constraint=0.3,
                                                    turnover_constraint=0.4,
                                                    adt_constraint=0.10,
                                                    liquidity_constraint=0.10)
    signal = optimize_long_only_with_constraints(signal.index, trade_date, data.set_index('script_name'),
                                                 1E7, optimization_constraints,
                                                 old_signal)
    return signal[['Weight']]