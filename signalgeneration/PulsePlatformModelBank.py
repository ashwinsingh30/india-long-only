import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

from database.finders.EquitiesPriceDataFinder import get_latest_price_and_signals
from model.EquitiesSignalProcessingModel import optimize_small_long_portfolio
from model.LongOnlyPortfolioOptimization import LongOnlyOptimizerConstraints
from signalgeneration.FundamentalLongOnlyPulse import get_fundamental_long_only_pulse, \
    get_small_fundamental_long_only_pulse
from signalgeneration.MomentumLongOnlyPulse import get_momentum_long_only_pulse, get_small_momentum_long_only_pulse
from utils.Constants import sector_map


def india_long_only_pulse(trade_date, old_signal=None, data=None):
    data = get_latest_price_and_signals(trade_date)
    if not old_signal is None:
        momentum_pulse = get_momentum_long_only_pulse(trade_date,
                                                      old_signal=old_signal['momentum_signal'], data=data)['Weight']
        fundamental_pulse = get_fundamental_long_only_pulse(trade_date, old_signal=old_signal['fundamental_signal'],
                                                            data=data)['Weight']
    else:
        momentum_pulse = get_momentum_long_only_pulse(trade_date, data=data)['Weight']
        fundamental_pulse = get_fundamental_long_only_pulse(trade_date, data=data)['Weight']
    securities = np.union1d(momentum_pulse.index, fundamental_pulse.index)
    signal_matrix = pd.DataFrame(index=securities)
    signal_matrix['momentum_signal'] = momentum_pulse
    signal_matrix['fundamental_signal'] = fundamental_pulse
    signal_matrix = signal_matrix.fillna(0)
    signal_matrix['Weight'] = (signal_matrix['momentum_signal'] + signal_matrix['fundamental_signal']) / 2
    signal_matrix['Weight'] = signal_matrix['Weight'] / signal_matrix['Weight'].abs().sum()
    signal_matrix = signal_matrix[signal_matrix['Weight'] != 0]
    signal_matrix['Price'] = data.set_index('script_name')['close_price']
    signal_matrix['sector'] = sector_map
    return signal_matrix[['Weight', 'momentum_signal', 'fundamental_signal', 'Price', 'sector']]


def india_small_long_only_pulse(trade_date, old_signal=None, data=None):
    data = get_latest_price_and_signals(trade_date)
    portfolio_value = 5E4
    single_stock_limit = max(0.025, (0.1 * 5E4) / portfolio_value)
    max_value = portfolio_value * single_stock_limit
    if not old_signal is None:
        momentum_pulse = get_small_momentum_long_only_pulse(trade_date - relativedelta(months=1), max_value,
                                                            old_signal=old_signal['momentum_signal'],
                                                            data=data,
                                                            single_stock_limit=single_stock_limit)['Weight']
        fundamental_pulse = get_small_fundamental_long_only_pulse(trade_date, max_value,
                                                                  old_signal=old_signal['fundamental_signal'],
                                                                  data=data,
                                                                  single_stock_limit=single_stock_limit)['Weight']
    else:
        momentum_pulse = get_small_momentum_long_only_pulse(trade_date - relativedelta(months=1), max_value,
                                                            data=data)['Weight']
        fundamental_pulse = get_small_fundamental_long_only_pulse(trade_date, max_value, data=data)['Weight']
    securities = np.union1d(momentum_pulse.index, fundamental_pulse.index)
    signal_matrix = pd.DataFrame(index=securities)
    signal_matrix['momentum_signal'] = momentum_pulse
    signal_matrix['fundamental_signal'] = fundamental_pulse
    signal_matrix = signal_matrix.fillna(0)
    signal_matrix['Weight'] = (signal_matrix['momentum_signal'] + signal_matrix['fundamental_signal']) / 2
    signal_matrix = signal_matrix[signal_matrix['Weight'] != 0]
    signal_matrix['Weight'] = signal_matrix['Weight'] / signal_matrix['Weight'].abs().sum()
    optimization_constraints = LongOnlyOptimizerConstraints(single_stock_bound=(0, single_stock_limit),
                                                            turnover_constraint=0.6,
                                                            adt_constraint=0.10,
                                                            liquidity_constraint=0.10)
    signal = optimize_small_long_portfolio(signal_matrix['Weight'], trade_date, data.set_index('script_name'),
                                           portfolio_value, optimization_constraints, old_signal['Weight'])
    signal_matrix['Weight'] = signal['Weight']
    signal_matrix['no_of_shares'] = signal['no_of_shares']
    signal_matrix['Price'] = data.set_index('script_name')['close_price']
    signal_matrix['sector'] = sector_map
    return signal_matrix[['Weight', 'momentum_signal', 'fundamental_signal', 'Price', 'sector']]
