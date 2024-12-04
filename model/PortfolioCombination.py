import pandas as pd
import numpy as np

from database.finders.EquitiesPriceDataFinder import get_X_day_historical_price_securities
from markowitzoptimization import expected_returns, risk_models
from markowitzoptimization.efficient_frontier import EfficientFrontier


def get_model_simulations_moving(model_returns, trade_date, moving_window):
    model_returns_moving = model_returns[model_returns.index <= trade_date]
    return model_returns_moving.sort_index().tail(moving_window)


def get_model_weights(trade_date, look_back, returns):
    model_returns_moving = get_model_simulations_moving(returns, trade_date, look_back)
    return optimize_portfolio(model_returns_moving, bounds=(0, 0.1))


def model_weights_mid_point(trade_date, returns):
    weights_20 = get_model_weights(trade_date, 60, returns)
    weights_60 = get_model_weights(trade_date, 250, returns)
    weights_120 = get_model_weights(trade_date, 120, returns)
    all_pulses = np.append(np.append(weights_20.index, weights_60.index), weights_120.index)
    all_pulses = np.unique(all_pulses)
    model_weights = pd.DataFrame(index=all_pulses)
    model_weights['trail_20'] = weights_20
    model_weights['trail_60'] = weights_60
    model_weights['trail_120'] = weights_120
    return model_weights.fillna(0).mean(axis=1)


def get_leverage_fraction(trade_date, returns):
    model_returns_moving = get_model_simulations_moving(returns, trade_date, 250)
    return_expectation = model_returns_moving.mean()
    return_expectation.loc[return_expectation >= 0] = 1
    return_expectation.loc[return_expectation < 0] = 0
    return np.ceil((0.3 + 0.7 * return_expectation.mean()) * 5) / 5



def optimize_long_portfolio(long_portfolio, trade_date, moving_window):
    securities = long_portfolio.index
    prices = get_X_day_historical_price_securities(securities, trade_date, moving_window)
    prices = prices.pivot_table(index='trade_date', columns='script_name', values='close_price')
    prices.dropna(axis=1)
    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.sample_cov(prices)
    ef = EfficientFrontier(mu, S, weight_bounds=(0,0.15))
    ef.max_sharpe(risk_free_rate=0.10)
    return pd.Series(ef.clean_weights())


def optimize_short_portfolio(long_portfolio, trade_date, moving_window):
    securities = long_portfolio.index
    prices = get_X_day_historical_price_securities(securities, trade_date, moving_window)
    prices = prices.pivot_table(index='trade_date', columns='script_name', values='close_price')
    prices.dropna(axis=1)
    mu = expected_returns.mean_historical_return(prices)
    mu *= -1
    S = risk_models.sample_cov(prices)
    ef = EfficientFrontier(mu, S, weight_bounds=(0,0.15))
    ef.max_sharpe(risk_free_rate=0.15)
    return pd.Series(ef.clean_weights())


def optimize_portfolio(returns, bounds):
    mu = returns.mean()
    std = returns.cov()
    ef = EfficientFrontier(mu, std, weight_bounds=bounds)
    ef.max_sharpe(risk_free_rate=0)
    return pd.Series(ef.clean_weights())


def top_pulses(model_returns_moving, n):
    mean = model_returns_moving.mean()
    std = model_returns_moving.std()
    sharpe = (mean / std).sort_values().tail(n)
    return pd.Series((1 / n), index=sharpe.index, dtype='float64')