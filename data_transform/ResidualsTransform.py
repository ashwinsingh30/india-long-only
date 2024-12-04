import numpy as np
import pandas as pd

from model.EquitiesSignalProcessingModel import create_neutralization_membership_matrix
from utils.Constants import sector_map


def create_membership_and_residuals_matrix(equities_data, securities, neutralise_sectors=True):
    membership_matrix = create_neutralization_membership_matrix(equities_data, sector_map, securities,
                                                                neutral_dimensions=['beta'],
                                                                neutralise_sectors=neutralise_sectors).fillna(0)
    q, r = np.linalg.qr(membership_matrix)
    residual_matrix = np.identity(q.shape[0]) - np.dot(q, q.T)
    return membership_matrix, residual_matrix


def get_price_residuals(membership_matrix, residual_matrix, latest_price, previous_price):
    securities = membership_matrix.index
    latest_price = latest_price[latest_price['script_name'].isin(securities)] \
        .set_index('script_name').reindex(securities)
    ohlc = ['open_price', 'high_price', 'low_price', 'close_price']
    latest_price_change = (latest_price[ohlc]).divide(previous_price, axis=0) - 1
    latest_price_change = latest_price_change.dropna()
    price_residuals = pd.DataFrame(np.dot(residual_matrix, latest_price_change),
                                   index=membership_matrix.index, columns=ohlc)
    price_residuals = (price_residuals + 1).multiply(previous_price, axis=0)
    price_residuals['trade_date'] = latest_price['trade_date']
    price_residuals['volume'] = latest_price['volume']
    price_residuals['previous_close'] = latest_price['previous_close']
    price_residuals['diff'] = (price_residuals['close_price'] / price_residuals['previous_close']) - 1
    price_residuals.index.name = 'script_name'
    price_residuals.reset_index(inplace=True)
    return price_residuals


def get_alpha_residuals(membership_matrix, residual_matrix, alpha_score_df, alpha_names):
    alpha_matrix = pd.DataFrame(index=membership_matrix.index)
    for alpha in alpha_names:
        if '_conj' in alpha:
            multiplier = -1
        else:
            multiplier = 1
        alpha_name = alpha.replace('_conj', '')
        weights = alpha_score_df[alpha_name].rank(na_option='keep')
        weights = -1 * multiplier * (weights - weights.mean())
        weights = weights / weights.abs().sum()
        weights.name = alpha
        alpha_matrix = pd.concat([alpha_matrix, weights], axis=1)
    alpha_matrix = alpha_matrix.fillna(0)
    residual_alphas = np.dot(residual_matrix, np.array(alpha_matrix))
    return residual_alphas


def get_long_only_alpha_residuals(membership_matrix, residual_matrix, alpha_score_df, alpha_names):
    alpha_matrix = pd.DataFrame(index=membership_matrix.index)
    for alpha in alpha_names:
        if '_conj' in alpha:
            multiplier = -1
        else:
            multiplier = 1
        alpha_name = alpha.replace('_conj', '')
        weights = alpha_score_df[alpha_name].rank(na_option='keep')
        weights = -1 * multiplier * (weights - weights.mean())
        weights.loc[weights < 0] = 0
        weights = weights / weights.abs().sum()
        alpha_matrix[alpha] = weights.fillna(0)
    residual_alphas = np.dot(residual_matrix, np.array(alpha_matrix))
    return residual_alphas


def get_prime_alpha_residuals(membership_matrix, residual_matrix, alpha_score_df, alpha_names):
    alpha_matrix = pd.DataFrame(index=membership_matrix.index)
    for alpha in alpha_names:
        if '_conj' in alpha:
            multiplier = -1
        else:
            multiplier = 1
        alpha_name = alpha.replace('_conj', '')
        weights = alpha_score_df[alpha_name].rank(na_option='keep')
        weights = -1 * multiplier * (weights - weights.mean())
        weights = weights / weights.abs().sum()
        long_cut_off = weights.quantile(q=0.9)
        short_cut_off = weights.quantile(q=0.1)
        weights.loc[(weights < long_cut_off) & (weights > short_cut_off)] = 0
        weights.loc[weights > 0] = 1
        weights.loc[weights < 0] = -1
        alpha_matrix[alpha] = weights.fillna(0)
    residual_alphas = np.dot(residual_matrix, np.array(alpha_matrix))
    return residual_alphas


def get_return_residuals(membership_matrix, residual_matrix, forward_price):
    forward_dates = forward_price.trade_date.sort_values().unique()
    forward_price = forward_price.pivot_table(index='script_name', columns='trade_date', values='close_price')
    forward_returns = pd.DataFrame(index=membership_matrix.index)
    for i in range(1, 21):
        forward_returns[str(i) + 'D'] = (forward_price[forward_dates[i]] / forward_price[forward_dates[0]]) - 1
    forward_returns = forward_returns.fillna(0)
    return_residuals = np.dot(residual_matrix, forward_returns)
    return return_residuals


def residuals_for_regression(equities_data, forward_price, alpha_names, trade_date, neutralise_sectors=True):
    securities = equities_data.index.values
    membership_matrix, residual_matrix = create_membership_and_residuals_matrix(equities_data, securities,
                                                                                neutralise_sectors=neutralise_sectors)
    alpha_residuals = get_alpha_residuals(membership_matrix, residual_matrix, equities_data, alpha_names)
    alpha_residuals = pd.DataFrame(alpha_residuals, columns=alpha_names)
    return_residuals = get_return_residuals(membership_matrix, residual_matrix, forward_price)
    columns = [str(i) + 'D' for i in range(1, 21)]
    return_residuals = pd.DataFrame(return_residuals, columns=columns)
    day_regression_df = pd.concat([return_residuals, alpha_residuals], axis=1)
    day_regression_df['trade_date'] = trade_date
    return day_regression_df


def residuals_for_long_only_regression(equities_data, forward_price, alpha_names, trade_date):
    securities = equities_data.index.values
    membership_matrix, residual_matrix = create_membership_and_residuals_matrix(equities_data, securities)
    alpha_residuals = get_long_only_alpha_residuals(membership_matrix, residual_matrix, equities_data, alpha_names)
    alpha_residuals = pd.DataFrame(alpha_residuals, columns=alpha_names)
    return_residuals = get_return_residuals(membership_matrix, residual_matrix, forward_price)
    return_residuals = pd.DataFrame(return_residuals, columns=['1D', '5D'])
    day_regression_df = pd.concat([return_residuals, alpha_residuals], axis=1)
    day_regression_df['trade_date'] = trade_date
    return day_regression_df
