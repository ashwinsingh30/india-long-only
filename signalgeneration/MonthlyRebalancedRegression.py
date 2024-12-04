import os

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from config.ConfiguredLogger import get_logger
from database.finders.EquitiesPriceDataFinder import get_close_price_after_x_days, \
    get_latest_price_and_signals_securities
from model.RegressionAndClusteringModels import independent_regression_weights, regress_alphas
from utils.DateUtils import parse_date
from utils.TradingPlatformUtils import get_daily_sampled_nse_500_universe

log = get_logger(os.path.basename(__file__))


def populate_monthly_regression_df(start_date, end_date, alpha_names, universe):
    trade_date = start_date
    regression_df = pd.DataFrame()
    while trade_date < end_date:
        log.info('Regression Residuals for: %s', trade_date)
        securities = universe[universe.trade_date == trade_date].script_name.values
        equities_data = get_latest_price_and_signals_securities(securities, trade_date)
        securities = np.intersect1d(securities, equities_data.script_name.unique())
        equities_data = equities_data.copy().set_index('script_name').reindex(securities)
        ranks = equities_data[alpha_names].rank()
        ranks = (ranks - ranks.mean()).fillna(0)
        weights = ranks / ranks.abs().sum()
        forward_price = get_close_price_after_x_days(securities, trade_date, 70)
        benchmark_forward_price = get_close_price_after_x_days(['NSE500'], trade_date, 70)['NSE500']
        benchmark_forward_1m = benchmark_forward_price.iloc[20] / benchmark_forward_price.iloc[0] - 1
        benchmark_forward_3m = benchmark_forward_price.iloc[60] / benchmark_forward_price.iloc[0] - 1
        print(benchmark_forward_1m)
        beta = equities_data['beta']
        print(beta)
        forward_returns = pd.DataFrame(index=weights.index)
        forward_returns['1M'] = forward_price.iloc[20] / forward_price.iloc[0] - 1
        forward_returns['3M'] = forward_price.iloc[60] / forward_price.iloc[0] - 1
        forward_returns['1M'] = forward_returns['1M'] - beta * benchmark_forward_1m
        forward_returns['3M'] = forward_returns['3M'] - beta * benchmark_forward_3m
        print(forward_returns)
        day_regression_df = weights.join(forward_returns.fillna(0), how='inner').reset_index()
        day_regression_df['trade_date'] = trade_date
        print(day_regression_df)
        regression_df = pd.concat([regression_df, day_regression_df], axis=0)
        trade_date = trade_date + relativedelta(months=1)
    return regression_df.reset_index()


start_date = parse_date('2012-01-01')
end_date = parse_date('2018-01-01')
in_sample_end_date = parse_date('2018-01-01')
universe = get_daily_sampled_nse_500_universe(start_date, end_date)
factor_dict = {
    'value': ['book_to_price', 'dividend_yield', 'earnings_to_price', 'earnings_to_price_estimate',
              'cash_flows_to_price', 'free_cash_flows_to_price_estimate', 'earning_yield', 'earning_yield_estimate',
              'sales_to_price', 'sales_to_price_estimate'],
    'momentum': ['one_month_abs_momentum', 'one_month_momentum_vol_adjusted', 'one_month_residual_momentum',
                 'one_month_residual_momentum_vol_adjusted', 'abs_price_momentum_one_year',
                 'residual_price_momentum_one_year', 'abs_price_momentum_one_qtr', 'residual_price_momentum_one_qtr',
                 'momentum_100', 'momentum_250', 'momentum_500', 'vol_250', 'vol_500', 'hurst500', 'hurst250'],
    'quality': ['asset_turnover_ratio', 'cash_to_sales', 'ce_return_on_equity', 'return_on_invested_capital',
                'free_cash_flows_to_invested_capital', 'gross_profit_margin', 'inventory_turnover',
                'minimum_gross_margin', 'net_profit_margin', 'net_profit_margin_estimate', 'operating_cash_flow_margin',
                'return_on_assets', 'roe_coefficient_of_variation'],
    'volatility': ['volatility_one_year', 'volatility_one_quarter', 'volatility_two_year', 'stability_one_year'],
    'growth': ['free_cash_flows_pct_change', 'cash_flow_trend_line', 'earnings_trend_line', 'revenue_trend_line',
               'change_in_free_cash_flows_to_assets', 'dividend_yield_cagr', 'revenue_growth', 'earnings_growth',
               'sales_momentum', 'earnings_to_price_estimate_growth', 'earning_to_price_growth_two_year_estimate',
               'eps_growth_estimate_one_year', 'eps_growth_estimate_five_year'],
    'leverage': ['net_external_financing', 'cash_ratio', 'cash_flow_debt_coverage', 'percentage_debt_change_one_year',
                 'ebitda_interest_coverage', 'total_coverage', 'degree_of_financial_leverage', 'debt_to_assets',
                 'debt_to_equity'],
    'size': ['logged_market_cap', 'logged_assets', 'logged_revenue'],
    'estimates': ['rating_revision_100d', 'revenue_growth_estimate', 'eps_estimates_range', 'target_price_estimate',
                  'rating_revision_20d', 'earning_to_price_fwd_estimate', 'eps_revisions_one_qtr',
                  'eps_revisions_two_qtr', 'eps_revisions_one_year', 'eps_revision_dispersion_one_qtr',
                  'eps_revision_dispersion_two_qtr', 'eps_revision_dispersion_one_year'],
}

alphas = np.array([])
for factor in factor_dict:
    alphas = np.append(alphas, factor_dict[factor])
alphas = np.unique(alphas)
factor_df = pd.DataFrame(index=[a.replace('', '') for a in alphas])
for factor in factor_dict:
    factor_alphas = factor_dict[factor]
    factor_series = pd.Series(dtype='float64')
    for alpha in factor_alphas:
        if '' in alpha:
            alpha = alpha.replace('', '')
            print(alpha)
            factor_series[alpha] = 1
        else:
            factor_series[alpha] = -1
    factor_df['factor_' + factor] = factor_series

factor_df = factor_df.fillna(0)

regression_df = populate_monthly_regression_df(start_date, end_date, alphas, universe)
regression_df.to_pickle('RegressionDFFundamental.pkl')
# regression_df = pd.read_pickle('RegressionDFFundamental.pkl')

dates = regression_df.trade_date.sort_values().unique()
in_sample_start = in_sample_end_date - relativedelta(years=6)
in_sample_dates = dates[np.where((dates >= in_sample_start) & (dates < in_sample_end_date))]
in_sample_data = regression_df[regression_df.trade_date.isin(in_sample_dates)].copy().dropna(how='all', axis=1)
weights = pd.DataFrame()
for factor in factor_dict:
    print(factor)
    regression_alphas = factor_dict[factor]
    alpha_weights = regress_alphas(in_sample_data[regression_alphas].fillna(0),
                                   in_sample_data['1M'].fillna(0), (0, 1), factor)
    # print(alpha_weights)
    if not alpha_weights.empty:
        alpha_weights = alpha_weights[['pulse', 'long_short', 'model', 'model_type']]
        weights = pd.concat([weights, alpha_weights], axis=0)

weights.to_csv(r'D:/Project/trading-platform-longonly/utils/model_configs/monthly_regression.csv')
