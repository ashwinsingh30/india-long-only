import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from scipy.linalg import eigh
from scipy.stats import norm

from database.finders.EquitiesPriceDataFinder import get_historical_price_table_between_dates, \
    get_X_day_historical_price_table
from markowitzoptimization import expected_returns
from model.InteriorPointPortfolioOptimization import InteriorPointPortfolioOptimization
from model.LongOnlyPortfolioOptimization import LongOnlyPortfolioOptimization
from utils.Constants import sector_map


def norm_min_max(series, target_range=(-1, 1)):
    return target_range[0] + (series - series.min()) / (series.max() - series.min()) * \
        (target_range[1] - target_range[0])


def norm_zscore(series):
    return (series - series.mean()) / series.std()


def norm_ranked(series):
    rank_vector = series.rank()
    rank_vector = (rank_vector - rank_vector.mean())
    rank_vector = rank_vector / rank_vector.abs().sum()
    return rank_vector


def create_neutralization_membership_matrix(data, sector_map, securities, neutral_dimensions, neutralise_sectors=True):
    membership_matrix = pd.DataFrame(index=securities)
    membership_matrix['universe'] = 1
    for dimension in neutral_dimensions:
        dim_data = data[dimension]
        high_dim = pd.Series(1, dtype='float64', index=dim_data[dim_data >= dim_data.quantile(q=0.5)].index)
        low_dim = pd.Series(1, dtype='float64', index=dim_data[dim_data < dim_data.quantile(q=0.5)].index)
        membership_matrix['high_' + dimension] = high_dim
        membership_matrix['low_' + dimension] = low_dim
    if neutralise_sectors:
        sectors = sector_map.unique()
        for sector in sectors:
            sector_stocks = sector_map[sector_map == sector].index
            sector_series = pd.Series(1, dtype='float64', index=sector_stocks)
            membership_matrix[sector] = sector_series
    return membership_matrix


def select_long_stocks(weights, alpha_weight):
    stock_weights = alpha_weight * weights
    long_stocks = stock_weights[stock_weights > 0]
    long_stocks = np.abs(alpha_weight) * (long_stocks / long_stocks.abs().sum())
    return pd.Series(long_stocks, index=weights.index).fillna(0)


def select_short_stocks(weights, alpha_weight):
    stock_weights = alpha_weight * weights
    short_stocks = stock_weights[stock_weights < 0]
    short_stocks = np.abs(alpha_weight) * (short_stocks / short_stocks.abs().sum())
    return pd.Series(short_stocks, index=weights.index).fillna(0)


def normalize_signal_min_max(equities_data, series, target_range=(-1, 1)):
    data = equities_data.copy()
    min = data.groupby('trade_date').min()[[series]]
    max = data.groupby('trade_date').max()[[series]]
    data = data.reset_index().set_index('trade_date')
    data['min'] = min[series]
    data['max'] = max[series]
    data = data.reset_index().set_index('equities_hash')
    return target_range[0] + (data[series] - data['min']) / (data['max'] - data['min']) * \
        (target_range[1] - target_range[0])


def normalize_signal_zscore(equities_data, series):
    data = equities_data.copy()
    mean = data.groupby('trade_date').mean()[[series]]
    std = data.groupby('trade_date').std()[[series]]
    data = data.reset_index().set_index('trade_date')
    data['mean'] = mean[series]
    data['std'] = std[series]
    data = data.reset_index().set_index('equities_hash')
    return (data[series] - data['mean']) / data['std']


def get_covariance_matrix_for_securities(trade_date, securities, look_back):
    price_data = get_X_day_historical_price_table(securities, trade_date, look_back)
    price_data.sort_index(inplace=True)
    price_data = price_data.pct_change()
    return price_data.cov()


def centroid_vector_from_sort(value, n):
    a = 0.4424
    b = 0.1185
    beta = 0.21
    alpha = a - b * np.power(n, -1 * beta)
    return norm.ppf(((n + 1 - value - alpha) / (n - 2 * alpha + 1)))


def convert_to_centroid_vector(weight_vector):
    weight_vector = weight_vector[weight_vector != 0]
    rank_vector = weight_vector.rank(ascending=False)
    centroid_vector = rank_vector.apply(centroid_vector_from_sort, args=(len(rank_vector.index),))
    centroid_vector = centroid_vector / centroid_vector.abs().sum()
    return centroid_vector


def normalize_long_short_portfolio_weights(weight_vector):
    long_portfolio = weight_vector[weight_vector >= 0]
    short_portfolio = weight_vector[weight_vector < 0]
    long_multiplier = long_portfolio.sum() / (long_portfolio.sum() + short_portfolio.abs().sum())
    if long_multiplier > 0.545:
        long_multiplier = 0.545
        short_multiplier = 0.455
    else:
        short_multiplier = short_portfolio.abs().sum() / (long_portfolio.sum() + short_portfolio.abs().sum())
    long_portfolio = long_multiplier * long_portfolio / long_portfolio.sum()
    short_portfolio = short_multiplier * short_portfolio / short_portfolio.abs().sum()
    return pd.concat([long_portfolio, short_portfolio], axis=0)


def optimize_small_long_portfolio(weight_vector, trade_date, data, capital, constraints, old_signal):
    optimization_matrix = pd.DataFrame(index=weight_vector.index)
    optimization_matrix['Weight'] = weight_vector
    if old_signal is not None:
        old_signal = old_signal[old_signal != 0]
        old_signal = pd.Series(old_signal, dtype='float64').fillna(0)
        old_signal.name = 'old_signal'
        optimization_matrix = optimization_matrix.join(old_signal, how='outer')
    else:
        optimization_matrix['old_signal'] = 0
    optimization_matrix = optimization_matrix.fillna(0)
    historical_prices = get_historical_price_table_between_dates(np.append(optimization_matrix.index, 'NIFTY'),
                                                                 trade_date - relativedelta(years=1),
                                                                 trade_date)
    historical_prices.sort_index(inplace=True)
    expectation = expected_returns.mean_historical_return(historical_prices)
    covariance_matrix = historical_prices.pct_change().cov()
    market_volatility = covariance_matrix.loc['NIFTY']['NIFTY']
    covariance_matrix.drop('NIFTY', axis=0, inplace=True)
    covariance_matrix.drop('NIFTY', axis=1, inplace=True)
    optimization_matrix['sector_map'] = sector_map
    optimization_matrix['close_price'] = data['close_price']
    optimization_matrix['expected_returns'] = expectation
    optimization_matrix['adt'] = data['adt']
    optimization_matrix['beta'] = data['beta']
    optimization_matrix['adt'] = optimization_matrix['adt'].fillna(0)
    optimization_matrix['beta'] = optimization_matrix[['beta']].fillna(1)
    covariance_matrix = covariance_matrix.loc[optimization_matrix.index][optimization_matrix.index]
    optimizer = LongOnlyPortfolioOptimization(optimization_matrix, covariance_matrix, market_volatility, capital,
                                              constraints, old_signal)
    optimization_matrix[['Weight', 'no_of_shares']] = optimizer.optimize_long_only_portfolio()
    optimization_matrix['Weight'] = optimization_matrix['Weight'] / optimization_matrix['Weight'].sum()
    return optimization_matrix[['Weight', 'no_of_shares']]


def optimize_long_short_with_constraints(weight_vector, trade_date, data, capital, constraints, old_signal):
    security_list = weight_vector.index
    optimization_matrix = pd.DataFrame(index=security_list)
    if old_signal is not None:
        old_signal = old_signal[old_signal != 0]
        old_signal = pd.Series(old_signal, dtype='float64').fillna(0)
        old_signal.name = 'old_signal'
        optimization_matrix = optimization_matrix.join(old_signal, how='outer')
        optimization_matrix['old_signal'] = optimization_matrix['old_signal'].fillna(0)
    else:
        optimization_matrix['old_signal'] = 0
    historical_prices = get_historical_price_table_between_dates(np.append(security_list, 'NIFTY'),
                                                                 trade_date - relativedelta(years=2),
                                                                 trade_date)
    historical_prices.sort_index(inplace=True)
    expectation = expected_returns.mean_historical_return(historical_prices)
    covariance_matrix = historical_prices.pct_change().cov().dropna(how='all')
    market_volatility = covariance_matrix.loc['NIFTY']['NIFTY']
    covariance_matrix.drop('NIFTY', axis=0, inplace=True)
    covariance_matrix.drop('NIFTY', axis=1, inplace=True)
    common = np.intersect1d(covariance_matrix.index, optimization_matrix.index)
    optimization_matrix = optimization_matrix.loc[common]
    covariance_matrix = covariance_matrix[common][common]
    optimization_matrix['centroid_vector'] = weight_vector
    optimization_matrix['centroid_vector'] = optimization_matrix['centroid_vector'] / \
                                             optimization_matrix['centroid_vector'].abs().sum()
    optimization_matrix['sector_map'] = sector_map
    optimization_matrix['expected_returns'] = expectation
    optimization_matrix['adt'] = data['adt']
    optimization_matrix['beta'] = data['beta']
    optimization_matrix['adt'] = optimization_matrix['adt'].fillna(0)
    optimization_matrix['beta'] = optimization_matrix[['beta']].fillna(1)
    covariance_matrix = covariance_matrix.loc[common][common]
    optimizer = InteriorPointPortfolioOptimization(optimization_matrix, covariance_matrix, market_volatility,
                                                   capital, constraints, security_list)
    optimization_matrix['Weight'] = optimizer.optimize_portfolio_cvxopt()
    optimization_matrix['Weight'] = optimization_matrix['Weight'] / optimization_matrix['Weight'].abs().sum()
    optimization_matrix.loc[((optimization_matrix['Weight'] >= -0.001) &
                             (optimization_matrix['Weight'] <= 0.001)), 'Weight'] = 0
    optimization_matrix['Weight'] = normalize_long_short_portfolio_weights(optimization_matrix['Weight'])
    return optimization_matrix[['Weight']]


def optimize_long_only_with_constraints(security_list, trade_date, data, capital, constraints, old_signal):
    optimization_matrix = pd.DataFrame(index=security_list)
    optimization_matrix['Weight'] = 1 / len(security_list)
    if old_signal is not None:
        old_signal = old_signal[old_signal != 0]
        old_signal = pd.Series(old_signal, dtype='float64').fillna(0)
        old_signal.name = 'old_signal'
        optimization_matrix = optimization_matrix.join(old_signal, how='outer')
    else:
        optimization_matrix['old_signal'] = 0
    optimization_matrix = optimization_matrix.fillna(0)
    historical_prices = get_historical_price_table_between_dates(np.append(optimization_matrix.index, 'NIFTY'),
                                                                 trade_date - relativedelta(years=1),
                                                                 trade_date)
    historical_prices.sort_index(inplace=True)
    expectation = expected_returns.mean_historical_return(historical_prices)
    covariance_matrix = historical_prices.pct_change().cov()
    market_volatility = covariance_matrix.loc['NIFTY']['NIFTY']
    covariance_matrix.drop('NIFTY', axis=0, inplace=True)
    covariance_matrix.drop('NIFTY', axis=1, inplace=True)
    optimization_matrix['sector_map'] = sector_map
    optimization_matrix['expected_returns'] = expectation
    optimization_matrix['adt'] = data['adt']
    optimization_matrix['beta'] = data['beta']
    optimization_matrix['adt'] = optimization_matrix['adt'].fillna(0)
    optimization_matrix['beta'] = optimization_matrix[['beta']].fillna(1)
    covariance_matrix = covariance_matrix.loc[optimization_matrix.index][optimization_matrix.index]
    optimizer = InteriorPointPortfolioOptimization(optimization_matrix, covariance_matrix, market_volatility,
                                                   capital, constraints, security_list)
    optimization_matrix['Weight'] = optimizer.optimize_long_only_portfolio()
    optimization_matrix['Weight'] = optimization_matrix['Weight'] / optimization_matrix['Weight'].sum()
    return optimization_matrix[['Weight']]