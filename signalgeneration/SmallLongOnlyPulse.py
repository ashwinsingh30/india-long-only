import os

import pandas as pd
from dateutil.relativedelta import relativedelta

from backtest.config.BacktestConfig import get_pulse_platform_backtest_config
from config.ConfiguredLogger import get_logger
from config.PulsePlatformConfig import get_pulse_platform_config
from database.finders.EquitiesPriceDataFinder import get_latest_price_and_signals
from database.finders.ModelPerfromanceFinder import get_performance_for_models_and_universe
from model.EquitiesSignalProcessingModel import centroid_vector_from_sort
from model.EquitiesSignalProcessingModel import norm_ranked, optimize_small_long_portfolio
from model.LongOnlyPortfolioOptimization import LongOnlyOptimizerConstraints
from model.PortfolioCombination import get_model_simulations_moving, optimize_portfolio
from signalgeneration.AuxxerePulse import get_basket_auxxere_pulse
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


def momentum_pulse(data, securities):
    signal = data.reindex(securities)[
        ['mom_100', 'mom_250', 'mom_500', 'vol_250', 'vol_500', 'hurst250', 'hurst500']].apply(
        norm_ranked)
    signal[['vol_250', 'vol_500']] = -1 * data[['vol_250', 'vol_500']]
    signal['signal'] = signal.mean(axis=1)
    signal = signal[signal['signal'] >= signal['signal'].quantile(q=0.8)]
    signal['momentum_weight'] = 1 / len(signal.index)
    return signal[['momentum_weight']]


def fundamental_pulse(trade_date, data, securities):
    signal_matrix = pd.DataFrame(index=securities)
    model_weights = get_greedy_factor_weights(trade_date, 250, returns, bounds=(0, 0.3))
    model_weights = model_weights[model_weights != 0]
    for model in model_weights.index:
        pulse = regression_weights[regression_weights['model'] == model].set_index('pulse')
        signal_matrix[model] = -1 * get_basket_auxxere_pulse(trade_date, securities,
                                                             pulse, data=data)['Weight']
    signal_matrix = signal_matrix.fillna(0)
    signal_matrix['fundamental_weight'] = signal_matrix[model_weights.index].dot(model_weights)
    signal_matrix['fundamental_weight'] = signal_matrix['fundamental_weight'] / signal_matrix[
        'fundamental_weight'].abs().sum()
    rank_vector = signal_matrix['fundamental_weight'].rank(ascending=False)
    signal_matrix['fundamental_weight'] = rank_vector.apply(centroid_vector_from_sort, args=(len(rank_vector.index),))
    signal_matrix = signal_matrix[
        signal_matrix['fundamental_weight'] > signal_matrix['fundamental_weight'].quantile(q=0.8)]
    signal_matrix['fundamental_weight'] = signal_matrix['fundamental_weight'] / signal_matrix[
        'fundamental_weight'].sum()
    return signal_matrix[['fundamental_weight']]


def get_small_long_only_pulse(trade_date, old_signal=None, data=None):
    global universe
    if config.run_mode == 'prod':
        securities = nse_500_equities
    else:
        securities = universe[universe.trade_date == trade_date].script_name.unique()
    if data is None:
        data = get_latest_price_and_signals(trade_date)
    data = data.set_index('script_name')
    momentum_signal = momentum_pulse(data, securities)
    fundamental_signal = fundamental_pulse(trade_date, data.reset_index(), securities)
    signal = pd.concat([momentum_signal, fundamental_signal], axis=1).fillna(0)
    signal['Weight'] = signal.mean(axis=1)
    signal = signal[signal.Weight > signal.Weight.quantile(q=0.5)]
    print(signal)
    signal['Weight'] = signal['Weight'] / signal['Weight'].sum()
    optimization_constraints = LongOnlyOptimizerConstraints(single_stock_bound=(0, 0.05),
                                                            beta_constraint=(0.5, 1),
                                                            gross_sector_constraint=0.30,
                                                            turnover_constraint=0.6,
                                                            adt_constraint=0.10,
                                                            liquidity_constraint=0.10)
    signal = optimize_small_long_portfolio(signal['Weight'], trade_date, data, 5E5,
                                           optimization_constraints,
                                           old_signal)
    signal['Price'] = data['close_price']
    return signal[['Weight', 'Price', 'no_of_shares']]

# import warnings
# warnings.filterwarnings("ignore")
# get_small_long_only_pulse(pd.to_datetime('2024-11-29').date()).to_csv('Data.csv')
