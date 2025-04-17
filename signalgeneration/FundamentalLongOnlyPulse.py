import os

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from backtest.config.BacktestConfig import get_pulse_platform_backtest_config
from config.ConfiguredLogger import get_logger
from config.PulsePlatformConfig import get_pulse_platform_config
from database.finders.EquitiesPriceDataFinder import get_latest_price_and_signals
from database.finders.ModelPerfromanceFinder import get_performance_for_models_and_universe
from model.EquitiesSignalProcessingModel import optimize_long_only_with_constraints, \
    centroid_vector_from_sort, optimize_long_only_active
from model.InteriorPointPortfolioOptimization import OptimizerConstraints
from model.PortfolioCombination import get_model_simulations_moving, optimize_portfolio
from signalgeneration.AuxxerePulse import get_basket_auxxere_pulse, get_long_only_factor_pulse, get_factor_pulse
from utils.Constants import nse_500_equities
from utils.TradingPlatformUtils import get_daily_sampled_nse_500_universe

log = get_logger(os.path.basename(__file__))
config = get_pulse_platform_config()
universe = None
returns = None
regression_weights = pd.read_csv(r'D:\Project\india-long-only\utils\model_configs\monthly_regression.csv')

if config.run_mode == 'backtest':
    test_config = get_pulse_platform_backtest_config()
    universe = get_daily_sampled_nse_500_universe(test_config.start_date - relativedelta(months=1),
                                                  test_config.end_date)
    returns, turnover = get_performance_for_models_and_universe(regression_weights.model.unique(), 'NSE500')


def get_greedy_factor_weights(trade_date, look_back, model_returns, bounds=None):
    model_returns_moving = get_model_simulations_moving(model_returns, trade_date, look_back)
    weights = optimize_portfolio(model_returns_moving, bounds=bounds)
    weights = weights / weights.abs().sum()
    return weights


def convert_to_centroid_vector(weight_vector):
    weight_vector = weight_vector[weight_vector != 0]
    rank_vector = weight_vector.rank(ascending=False)
    centroid_vector = rank_vector.apply(centroid_vector_from_sort, args=(len(rank_vector.index),))
    centroid_vector = centroid_vector / centroid_vector.abs().sum()
    return centroid_vector


def get_fundamental_long_only_pulse(trade_date, old_signal=None, data=None):
    global universe
    global returns
    if config.run_mode == 'prod':
        securities = nse_500_equities
        returns, turnover = get_performance_for_models_and_universe(regression_weights.model.unique(), 'NSE500')
    else:
        securities = universe[universe.trade_date == trade_date].script_name.unique()
    if data is None:
        data = get_latest_price_and_signals(trade_date)
    signal_matrix = pd.DataFrame(index=securities)
    model_weights = get_greedy_factor_weights(trade_date, 250, returns, bounds=(0, 0.3))
    model_weights = model_weights[model_weights != 0]
    log.info('\n' + model_weights.sort_values().to_string())
    if old_signal is not None:
        old_signal = old_signal / old_signal.abs().sum()
    for model in model_weights.index:
        pulse = regression_weights[regression_weights['model'] == model].set_index('pulse')
        signal_matrix[model] = -1 * get_basket_auxxere_pulse(trade_date, securities,
                                                             pulse, old_signal, data=data)['Weight']
    signal_matrix = signal_matrix.fillna(0)
    signal_matrix['Weight'] = signal_matrix[model_weights.index].dot(model_weights)
    signal_matrix['Weight'] = signal_matrix['Weight'] / signal_matrix['Weight'].abs().sum()
    rank_vector = signal_matrix['Weight'].rank(ascending=False)
    signal_matrix['Weight'] = rank_vector.apply(centroid_vector_from_sort, args=(len(rank_vector.index),))
    signal_matrix = signal_matrix[signal_matrix['Weight'] > signal_matrix['Weight'].quantile(q=0.8)]
    signal_matrix['Weight'] = signal_matrix['Weight'] / signal_matrix['Weight'].abs().sum()
    factor_constraints = ['value', 'quality', 'profitability', 'leverage', 'volatility', 'analyst_rating',
                          'size', 'long_term_trend', 'short_term_trend']
    factor_loadings = get_factor_pulse(trade_date, securities, factor_constraints, data)
    optimization_constraints = OptimizerConstraints(single_stock_bound=(0, 0.05),
                                                    beta_constraint=(0.8, 1.),
                                                    gross_sector_constraint=0.25,
                                                    turnover_constraint=0.4,
                                                    adt_constraint=0.10,
                                                    liquidity_constraint=0.10,
                                                    factor_constraints=factor_constraints)
    signal = optimize_long_only_with_constraints(signal_matrix.index, trade_date, data.set_index('script_name'),
                                                 5E8, optimization_constraints, old_signal, factor_loadings)
    return signal[['Weight']]


def get_small_fundamental_long_only_pulse(trade_date, max_value, old_signal=None, data=None, single_stock_limit=0.1):
    global universe
    global returns
    if config.run_mode == 'prod':
        securities = nse_500_equities
        returns, turnover = get_performance_for_models_and_universe(regression_weights.model.unique(), 'NSE500')
    else:
        securities = universe[universe.trade_date == trade_date].script_name.unique()
    if data is None:
        data = get_latest_price_and_signals(trade_date)
    q_cut = (4 / 3) * single_stock_limit + (23 / 30)
    raw_data = data.copy()
    raw_data['shares_possible'] = (max_value / raw_data['close_price'])
    raw_data = raw_data[raw_data['shares_possible'] > 1]
    securities = np.intersect1d(securities, raw_data.script_name.unique())
    signal_matrix = pd.DataFrame(index=securities)
    model_weights = get_greedy_factor_weights(trade_date, 250, returns, bounds=(0, 0.3))
    model_weights = model_weights[model_weights != 0]
    if old_signal is not None and 'NIFTY' in old_signal.index:
        old_signal = old_signal.drop('NIFTY')
        old_signal = old_signal / old_signal.abs().sum()
    for model in model_weights.index:
        pulse = regression_weights[regression_weights['model'] == model].set_index('pulse')
        signal_matrix[model] = -1 * get_basket_auxxere_pulse(trade_date, securities,
                                                             pulse, old_signal, data=raw_data)['Weight']
    signal_matrix = signal_matrix.fillna(0)
    signal_matrix['Weight'] = signal_matrix[model_weights.index].dot(model_weights)
    signal_matrix['Weight'] = signal_matrix['Weight'] / signal_matrix['Weight'].abs().sum()
    rank_vector = signal_matrix['Weight'].rank(ascending=False)
    signal_matrix['Weight'] = rank_vector.apply(centroid_vector_from_sort, args=(len(rank_vector.index),))
    signal_matrix = signal_matrix[signal_matrix['Weight'] > signal_matrix['Weight'].quantile(q=q_cut)]
    signal_matrix['Weight'] = signal_matrix['Weight'] / signal_matrix['Weight'].abs().sum()
    optimization_constraints = OptimizerConstraints(single_stock_bound=(0, single_stock_limit),
                                                    beta_constraint=(0.5, 1.1),
                                                    gross_sector_constraint=0.3,
                                                    turnover_constraint=0.4,
                                                    adt_constraint=0.10,
                                                    liquidity_constraint=0.10)
    signal = optimize_long_only_with_constraints(signal_matrix.index, trade_date, data.set_index('script_name'),
                                                 1E7, optimization_constraints, old_signal)
    return signal[['Weight']]
