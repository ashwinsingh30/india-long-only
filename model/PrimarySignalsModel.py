import os

import empyrical as em
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from scipy.stats import linregress

from config.ConfiguredLogger import get_logger
from utils.TradingPlatformUtils import hash_equities_by_row

# Pre Processing

nan_zero_filled = ['iq_common_issued', 'iq_common_rep', 'iq_minority_interest', 'iq_pref_equity', 'iq_pref_issued',
                   'iq_pref_rep', 'iq_total_debt_issued', 'iq_total_debt_repaid', 'iq_total_div_paid_cf',
                   'iq_change_net_working_capital', 'iq_total_debt_current', 'iq_inventory', 'iq_lt_debt']

sector_neutral_signals = ['free_cash_flows_pct_change', 'cash_flow_trend_line', 'earnings_trend_line',
                          'revenue_trend_line', 'change_in_free_cash_flows_to_assets', 'dividend_yield_cagr',
                          'revenue_growth', 'earnings_growth', 'sales_momentum', 'earnings_to_price_estimate_growth',
                          'earning_to_price_growth_two_year_estimate', 'eps_growth_estimate_one_year',
                          'asset_turnover_ratio', 'cash_to_sales', 'ce_return_on_equity', 'return_on_invested_capital',
                          'free_cash_flows_to_invested_capital', 'gross_profit_margin', 'inventory_turnover',
                          'minimum_gross_margin', 'net_profit_margin', 'net_profit_margin_estimate',
                          'operating_cash_flow_margin', 'return_on_assets', 'roe_coefficient_of_variation',
                          'percentage_debt_change_one_year', 'logged_market_cap', 'logged_assets', 'logged_revenue',
                          'net_external_financing', 'cash_ratio', 'cash_flow_debt_coverage', 'ebitda_interest_coverage',
                          'total_coverage', 'debt_to_assets', 'debt_to_equity', 'rating_revision_100d',
                          'revenue_growth_estimate', 'eps_estimates_range', 'rating_revision_20d',
                          'earning_to_price_fwd_estimate', 'book_to_price', 'dividend_yield', 'earnings_to_price',
                          'earnings_to_price_estimate', 'cash_flows_to_price', 'free_cash_flows_to_price_estimate',
                          'earning_yield', 'earning_yield_estimate', 'sales_to_price', 'sales_to_price_estimate',
                          'target_price_estimate', 'eps_growth_estimate_five_year']

market_neutral_signals = ['price_momentum_one_month', 'price_momentum_one_month_lagged', 'price_momentum_one_quarter',
                          'price_momentum_one_quarter_lagged', 'sector_momentum', 'industry_momentum',
                          'sector_momentum_one_quarter', 'industry_momentum_one_quarter', 'industry_momentum_one_year',
                          'sector_momentum_one_year', 'volatility_one_year', 'volatility_one_quarter',
                          'volatility_two_year', 'stability_one_year', 'sector_stability_one_year',
                          'industry_stability_one_year']

log = get_logger(os.path.basename(__file__))


def rolling_apply(df, window, function):
    return pd.Series([df.iloc[i - window: i].pipe(function)
                      if i >= window else None
                      for i in range(1, len(df) + 1)],
                     index=df.index)


def copy_column(data, column_name):
    unique_values = data[column_name].dropna().unique()
    output = pd.Series(index=data.index, dtype='float64')
    for value in unique_values:
        value_df = data[data[column_name] == value]
        output[value_df.index] = value_df[value]
    return output.sort_index()


def clip_pct_change(series, multiplier=1):
    series.loc[series > 1] = multiplier
    series.loc[series < -1] = -multiplier
    return series


def time_series_trend(series):
    series = series.ffill()
    series_length = len(series)
    no_of_samples = int(np.ceil(series_length / 60))
    sample_indices = np.linspace(0, series_length - 1, num=no_of_samples, dtype='int64')
    if len(sample_indices >= 4):
        regr_result = linregress(x=np.arange(0, no_of_samples), y=series.iloc[sample_indices].values)
        return regr_result.slope / series.mean()
    else:
        return np.nan


def dependence_slope(df):
    df = df.ffill()
    data_length = len(df.index)
    no_of_samples = int(np.ceil(data_length / 60))
    sample_indices = np.linspace(0, data_length - 1, num=no_of_samples, dtype='int64')
    x = df.iloc[sample_indices]['independent']
    y = df.iloc[sample_indices]['dependent']
    if len(sample_indices) >= 4 and len(x.dropna().unique()) > 1:
        regr_result = linregress(x=x.values, y=y.values)
        return regr_result.slope
    else:
        return np.nan


def momentum_from_regression(prices):
    momentum_window = int(len(prices) * 0.3)
    prices = prices.iloc[0:momentum_window]
    x = np.arange(len(prices))
    slope, _, rvalue, _, _ = linregress(x, np.log(prices))
    return slope


def aci_raw(data):
    data = data[data['iq_total_rev'] != 0]
    return data['iq_total_assets'] / data['iq_total_rev']


def free_cash_flow(data):
    return data['iq_cash_oper'] + data['iq_capex'] + data['iq_total_div_paid_cf']


def enterprise_value(data):
    return data['iq_lt_debt'] + data['iq_total_debt_current'] + data['iq_pref_equity'] + data[
        'iq_cash_st_invest'] + data['market_cap'] + data['iq_minority_interest']


def invested_capital(data):
    return data['iq_lt_debt'] + data['iq_total_common_equity'] + data['iq_pref_equity'] + data['iq_minority_interest']


def cash_flows(data):
    return data['iq_ni_avail_excl'] + data['iq_da_cf']


def operating_profit_margin(data):
    data = data[data['iq_total_rev'] != 0]
    return data['iq_oper_inc'] / data['iq_total_rev']


def net_income_estimate(data):
    return data['iq_eps_est_ciq'] * data['iq_sharesoutstanding']


def revenue_per_share(data):
    data = data[data['iq_sharesoutstanding'] != 0]
    return data['iq_total_rev'] / data['iq_sharesoutstanding']


def active_return_sector_one_quarter(data, sector):
    stock_return = data['close_price'].pct_change().rolling(60).sum()
    sector_return = data[sector].rolling(60).sum()
    return stock_return - sector_return


def active_return_sector_one_month(data, sector):
    stock_return = data['close_price'].pct_change().rolling(20).sum()
    market_return = data[sector].rolling(20).sum()
    return stock_return - market_return


def active_return_market_one_quarter(data):
    stock_return = data['close_price'].pct_change().rolling(60).sum()
    market_return = data['benchmark_return'].rolling(60).sum()
    return stock_return - market_return


def active_return_market_one_month(data):
    stock_return = data['close_price'].pct_change().rolling(20).sum()
    market_return = data['benchmark_return'].rolling(20).sum()
    return stock_return - market_return


# Growth Factors


def free_cash_flows_pct_change(data):
    fcf = free_cash_flow(data)
    return clip_pct_change((fcf - fcf.shift(250)) / fcf.shift(250).abs())


def cash_flow_trend_line(data):
    cf = cash_flows(data)
    return cf.rolling(750).apply(time_series_trend)


def earnings_trend_line(data):
    earnings = data['iq_ebitda']
    return earnings.rolling(750).apply(time_series_trend)


def revenue_trend_line(data):
    revenue = data['iq_total_rev']
    return revenue.rolling(750).apply(time_series_trend)


def change_in_free_cash_flows_to_assets(data):
    mean_assets = (data['iq_total_assets'] + data['iq_total_assets'].shift(250)) / 2
    data = data[mean_assets != 0]
    fcf = free_cash_flow(data)
    free_cash_flow_change = fcf - fcf.shift(250)
    return free_cash_flow_change / mean_assets


def dividend_yield_cagr(data):
    return clip_pct_change(np.power((data['iq_dividend_yield'] / data['iq_dividend_yield'].shift(750)), (1 / 3)) - 1)


def revenue_growth(data):
    revenue = data['iq_total_rev']
    return clip_pct_change((revenue - revenue.shift(250)) / revenue.shift(250).abs()).rolling(20).mean()


def earnings_growth(data):
    earnings = data['iq_ebitda']
    return clip_pct_change((earnings - earnings.shift(250)) / earnings.shift(250).abs()).rolling(20).mean()


def sales_momentum(data):
    sales = data['iq_total_rev']
    sales_change = clip_pct_change((sales - sales.shift(250)) / sales.shift(250).abs())
    return sales_change - sales_change.shift(250)


def earnings_to_price_estimate_growth(data):
    ep_est = earnings_to_price_estimate(data)
    return ep_est.diff(250)


def earning_to_price_growth_two_year_estimate(data):
    data = data[data['close_price'] != 0]
    return (data['iq_eps_est_ciq_fy_1'] - data['iq_eps_est_ciq_fy_2']) / data['close_price']


def eps_growth_estimate_one_year(data):
    data = data[data['iq_basic_eps_excl'] != 0]
    return (data['iq_eps_est_ciq_fy_1'] / data['iq_basic_eps_excl']) - 1


def eps_growth_estimate_five_year(data):
    return data['iq_est_eps_growth_5yr_ciq']


# Quality


def asset_turnover_ratio(data):
    data = data[data['iq_total_assets'] != 0]
    return data['iq_total_rev'] / data['iq_total_assets']


def cash_to_sales(data):
    data = data[data['iq_total_rev'] != 0]
    return data['iq_cash_equiv'] / data['iq_total_rev']


def ce_return_on_equity(data):
    data = data[data['iq_total_equity'] != 0]
    return (data['iq_ni_avail_excl'] + data['iq_da_cf'] + data['iq_change_net_working_capital']) / data[
        'iq_total_equity']


def return_on_invested_capital(data):
    ic = invested_capital(data)
    data = data[ic != 0]
    return data['iq_ebit'] / ic


def free_cash_flows_to_invested_capital(data):
    ic = invested_capital(data)
    data = data[ic != 0]
    fcf = free_cash_flow(data)
    return fcf / ic


def gross_profit_margin(data):
    data = data[data['iq_total_rev'] != 0]
    return data['iq_cogs'] / data['iq_total_rev']


def inventory_turnover(data):
    data = data[data['iq_inventory'] != 0]
    return data['iq_cogs'] / data['iq_inventory']


def minimum_gross_margin(data):
    data = data[data['iq_total_rev'] != 0]
    gm = data['iq_gp'] / data['iq_total_rev']
    return -1 * gm.tail(500).min()


def net_profit_margin(data):
    data = data[data['iq_total_rev'] != 0]
    return data['iq_ni_avail_excl'] / data['iq_total_rev']


def net_profit_margin_estimate(data):
    data = data[data['iq_total_rev'] != 0]
    ni_est = net_income_estimate(data)
    return ni_est / data['iq_revenue_median_est_ciq_fq_1']


def operating_cash_flow_margin(data):
    data = data[data['iq_total_rev'] != 0]
    return data['iq_cash_oper'] / data['iq_total_rev']


def return_on_assets(data):
    data = data[data['iq_total_assets'] != 0]
    data = data[data['iq_total_cl'] != 0]
    return clip_pct_change(data['iq_ebit'] / (data['iq_total_assets'] - data['iq_total_cl']))


def roe_coefficient_of_variation(data):
    roe = ce_return_on_equity(data)
    return -1 * roe.rolling(750).std() / roe.rolling(750).mean().abs()


# Size

def market_cap(data):
    data = data[data['market_cap'] > 0]
    return data['market_cap']


def logged_market_cap(data):
    data = data[data['market_cap'] > 0]
    return -1 * pd.Series(np.log(data['market_cap']), index=data.index)


def logged_assets(data):
    data = data[data['iq_total_assets'] > 0]
    return -1 * pd.Series(np.log(data['iq_total_assets']), index=data.index)


def logged_revenue(data):
    data = data[data['iq_total_rev'] > 0]
    return -1 * pd.Series(np.log(data['iq_total_rev']), index=data.index)


# Leverage

def net_external_financing(data):
    data = data[data['iq_total_assets'] != 0]
    return -1 * clip_pct_change(
        (data['iq_total_debt_issued'] + data['iq_total_debt_repaid'] + data['iq_common_issued'] + data[
            'iq_common_rep'] + data['iq_pref_issued'] + data['iq_pref_rep']) / data['iq_total_assets'])


def cash_ratio(data):
    data = data[data['iq_total_cl'] != 0]
    return data['iq_cash_equiv'] / data['iq_total_cl']


def cash_flow_debt_coverage(data):
    data = data[data['iq_total_debt'] != 0]
    return clip_pct_change(data['iq_cash_oper'] / data['iq_total_debt'], multiplier=2)


def percentage_debt_change_one_year(data):
    debt = data['iq_total_debt']
    return -1 * clip_pct_change((debt / debt.shift(250)) - 1)


def ebitda_interest_coverage(data):
    return data['iq_ebitda_int']


def total_coverage(data):
    divisor = data['iq_cash_interest'] - data['iq_total_debt_repaid']
    data = data[divisor != 0]
    return clip_pct_change((data['iq_cash_oper'] + data['iq_cash_interest'] + data['iq_cash_taxes']) / divisor)


def degree_of_financial_leverage(data):
    change_in_sales = (data['iq_total_rev'] - data['iq_total_rev'].shift(100)) / data['iq_total_rev'].shift(100).abs()
    change_in_earnings = (data['iq_ebit'] - data['iq_ebit'].shift(100)) / data['iq_ebit'].shift(100).abs()
    dataframe = pd.DataFrame(index=data.index)
    dataframe['dependent'] = change_in_earnings
    dataframe['independent'] = change_in_sales
    return rolling_apply(dataframe, 500, dependence_slope)


def debt_to_assets(data):
    data = data[data['iq_total_assets'] != 0]
    return clip_pct_change(data['iq_total_debt'] / data['iq_total_assets'], multiplier=2)


def debt_to_equity(data):
    data = data[data['iq_total_equity'] != 0]
    return clip_pct_change(data['iq_total_debt'] / data['iq_total_equity'], multiplier=2)


# Estimates


def rating_revision_100d(data):
    return -1 * clip_pct_change(data['iq_avg_broker_rec_no_ciq'].pct_change(100))


def revenue_growth_estimate(data):
    data = data[data['iq_total_rev'] != 0]
    return (data['iq_total_rev'] - data['iq_revenue_median_est_ciq_fq_1']) / data['iq_total_rev']


def eps_estimates_range(data):
    data = data[data['iq_eps_est_ciq_fy_1'] != 0]
    return -1 * (data['iq_eps_low_est_ciq_fy_1'] - data['iq_eps_high_est_ciq_fy_1']) / data['iq_eps_est_ciq_fy_1']


def short_interest_percentage(data):
    return data['iq_short_interest_percent']


def target_price_estimate(data):
    return data['iq_price_target_ciq'] / data['close_price']


def rating_revision_20d(data):
    return -1 * clip_pct_change(data['iq_avg_broker_rec_no_ciq'].pct_change(20))


def earning_to_price_fwd_estimate(data):
    data = data[data['iq_pe_excl_fwd_ciq'] != 0]
    return 1 / data['iq_pe_excl_fwd_ciq']


def eps_revisions_one_qtr(data):
    eps_estimate = data['iq_eps_est_ciq_fy_1']
    return (eps_estimate - eps_estimate.shift(60)) / eps_estimate.diff().rolling(250).std()


def eps_revisions_two_qtr(data):
    eps_estimate = data['iq_eps_est_ciq_fy_1']
    return (eps_estimate - eps_estimate.shift(120)) / eps_estimate.diff().rolling(250).std()


def eps_revisions_one_year(data):
    eps_estimate = data['iq_eps_est_ciq_fy_1']
    return (eps_estimate - eps_estimate.shift(250)) / eps_estimate.diff().rolling(250).std()


def eps_revision_dispersion_one_qtr(data):
    eps_estimate = (data['iq_eps_est_ciq_fy_1'] + data['iq_eps_est_ciq_fy_2']) / 2
    revisions = eps_estimate.pct_change()
    revisions.loc[revisions.abs() < 0.001] = 0
    revisions.loc[revisions > 0] = 1
    revisions.loc[revisions < 0] = -1
    return revisions.rolling(75).sum() / revisions.abs().rolling(75).sum()


def eps_revision_dispersion_two_qtr(data):
    eps_estimate = (data['iq_eps_est_ciq_fy_1'] + data['iq_eps_est_ciq_fy_2']) / 2
    revisions = eps_estimate.pct_change()
    revisions.loc[revisions.abs() < 0.001] = 0
    revisions.loc[revisions > 0] = 1
    revisions.loc[revisions < 0] = -1
    return revisions.rolling(180).sum() / revisions.abs().rolling(180).sum()


def eps_revision_dispersion_one_year(data):
    eps_estimate = (data['iq_eps_est_ciq_fy_1'] + data['iq_eps_est_ciq_fy_2']) / 2
    revisions = eps_estimate.pct_change()
    revisions.loc[revisions.abs() < 0.001] = 0
    revisions.loc[revisions > 0] = 1
    revisions.loc[revisions < 0] = -1
    return revisions.rolling(250).sum() / revisions.abs().rolling(250).sum()


# Value

def book_to_price(data):
    data = data[data['market_cap'] != 0]
    return data['iq_total_equity'] / data['market_cap']


def dividend_yield(data):
    return data['iq_dividend_yield']


def earnings_to_price(data):
    data = data[data['market_cap'] != 0]
    return data['iq_ni_avail_excl'] / data['market_cap']


def earnings_to_price_estimate(data):
    data = data[data['close_price'] != 0]
    return data['iq_eps_median_est_ciq'] / data['close_price']


def cash_flows_to_price(data):
    data = data[data['market_cap'] != 0]
    cf = cash_flows(data)
    return cf / data['market_cap']


def free_cash_flows_to_price_estimate(data):
    data = data[data['market_cap'] != 0]
    ni_est = net_income_estimate(data)
    fcf_est = ni_est + data['iq_da_cf'] - data['iq_capex']
    return fcf_est / data['market_cap']


def earning_yield(data):
    ev = enterprise_value(data)
    data = data[ev != 0]
    return data['iq_cash_oper'] / ev


def earning_yield_estimate(data):
    ev = enterprise_value(data)
    data = data[ev != 0]
    ni_est = net_income_estimate(data)
    return ni_est / ev


def sales_to_price(data):
    data = data[data['market_cap'] != 0]
    return data['iq_total_rev'] / data['market_cap']


def sales_to_price_estimate(data):
    data = data[data['market_cap'] != 0]
    return data['iq_revenue_median_est_ciq_fq_1'] / data['market_cap']


# Momentum
def one_month_abs_momentum(data):
    return data['close_price'].pct_change().rolling(20).sum()


def one_month_momentum_vol_adjusted(data):
    abs_return = data['close_price'].pct_change()
    return abs_return.rolling(20).sum() / abs_return.rolling(250).std()


def one_month_residual_momentum(data):
    residual_return = data['close_price'].pct_change() - data['benchmark']
    return residual_return.rolling(20).sum()


def one_month_residual_momentum_vol_adjusted(data):
    residual_return = data['close_price'].pct_change() - data['benchmark']
    return residual_return.rolling(20).sum() / residual_return.rolling(250).std()


def abs_price_momentum_one_year(data):
    lagged_price = data['close_price'].shift(20)
    return lagged_price.pct_change(250)


def residual_price_momentum_one_year(data):
    residual_return = data['close_price'].pct_change() - data['benchmark']
    lagged_return = residual_return.shift(20)
    return lagged_return.rolling(250).sum()


def abs_price_momentum_one_qtr(data):
    lagged_price = data['close_price'].shift(20)
    return lagged_price.pct_change().rolling(60).sum()


def residual_price_momentum_one_qtr(data):
    residual_return = data['close_price'].pct_change() - data['benchmark']
    lagged_return = residual_return.shift(20)
    return lagged_return.rolling(60).sum()


# Volatility

def volatility_one_year(data):
    return -1 * data['close_price'].pct_change().rolling(250).std()


def volatility_one_quarter(data):
    return -1 * data['close_price'].pct_change().rolling(60).std()


def volatility_two_year(data):
    return -1 * data['close_price'].pct_change().rolling(500).std()


def stability_one_year(data):
    return data['close_price'].pct_change().rolling(250).apply(em.stability_of_timeseries)


def sector_stability_one_year(data):
    return data['sector_returns'].rolling(250).apply(em.stability_of_timeseries)


def industry_stability_one_year(data):
    return data['industry_returns'].rolling(250).apply(em.stability_of_timeseries)


# Additional Risk factors

def assets_to_inventory(data):
    data = data[data['iq_inventory'] != 0]
    return data['iq_total_assets'] / data['iq_inventory']


def free_cash_flow_to_price(data):
    data = data[data['market_cap'] != 0]
    return (data['iq_cash_oper'] + data['iq_capex'] + data['iq_total_div_paid_cf']) / data['market_cap']


def operating_income_to_enterprise_value(data):
    ev = enterprise_value(data)
    data = data[ev != 0]
    return data['iq_cash_oper'] / ev


def de_leveraging(data):
    data = data[data['iq_total_assets'] != 0]
    return (data['iq_total_debt_repaid'] + data['iq_common_rep'] + data['iq_pref_rep']).abs() / data['iq_total_assets']


def dividend_payout_ratio(data):
    data = data[data['iq_ni_avail_excl'] != 0]
    return clip_pct_change(data['iq_total_div_paid_cf'].abs() / data['iq_ni_avail_excl'])


def short_term_momentum(data):
    return data['close_price'].rolling(5).mean() / data['close_price'].rolling(50).mean()


def long_term_momentum(data):
    return data['close_price'].rolling(25).mean() / data['close_price'].rolling(250).mean()


##############


def populate_primary_signals(script_data):
    script_data = script_data.set_index('trade_date')
    script_data = script_data.sort_index()
    script_data = script_data.ffill()
    primary_signals = pd.DataFrame(index=script_data.index)
    primary_signals['script_name'] = script_data['script_name']
    script_data['market_cap'] = script_data['iq_marketcap']

    ############## Growth #############

    primary_signals['free_cash_flows_pct_change'] = free_cash_flows_pct_change(script_data)
    primary_signals['cash_flow_trend_line'] = cash_flow_trend_line(script_data)
    primary_signals['earnings_trend_line'] = earnings_trend_line(script_data)
    primary_signals['revenue_trend_line'] = revenue_trend_line(script_data)
    primary_signals['change_in_free_cash_flows_to_assets'] = change_in_free_cash_flows_to_assets(script_data)
    primary_signals['dividend_yield_cagr'] = dividend_yield_cagr(script_data)
    primary_signals['revenue_growth'] = revenue_growth(script_data)
    primary_signals['earnings_growth'] = earnings_growth(script_data)
    primary_signals['sales_momentum'] = sales_momentum(script_data)
    primary_signals['earnings_to_price_estimate_growth'] = earnings_to_price_estimate_growth(script_data)
    primary_signals['earning_to_price_growth_two_year_estimate'] = earning_to_price_growth_two_year_estimate(
        script_data)
    primary_signals['eps_growth_estimate_one_year'] = eps_growth_estimate_one_year(script_data)
    primary_signals['eps_growth_estimate_five_year'] = eps_growth_estimate_five_year(script_data)

    ############# Quality #############

    primary_signals['asset_turnover_ratio'] = asset_turnover_ratio(script_data)
    primary_signals['cash_to_sales'] = cash_to_sales(script_data)
    primary_signals['ce_return_on_equity'] = ce_return_on_equity(script_data)
    primary_signals['return_on_invested_capital'] = return_on_invested_capital(script_data)
    primary_signals['free_cash_flows_to_invested_capital'] = free_cash_flows_to_invested_capital(script_data)
    primary_signals['gross_profit_margin'] = gross_profit_margin(script_data)
    primary_signals['inventory_turnover'] = inventory_turnover(script_data)
    primary_signals['minimum_gross_margin'] = minimum_gross_margin(script_data)
    primary_signals['net_profit_margin'] = net_profit_margin(script_data)
    primary_signals['net_profit_margin_estimate'] = net_profit_margin_estimate(script_data)
    primary_signals['operating_cash_flow_margin'] = operating_cash_flow_margin(script_data)
    primary_signals['return_on_assets'] = return_on_assets(script_data)
    primary_signals['roe_coefficient_of_variation'] = roe_coefficient_of_variation(script_data)

    ############# Size #############

    primary_signals['logged_market_cap'] = logged_market_cap(script_data)
    primary_signals['logged_assets'] = logged_assets(script_data)
    primary_signals['logged_revenue'] = logged_revenue(script_data)

    ############# Leverage #############

    primary_signals['net_external_financing'] = net_external_financing(script_data)
    primary_signals['cash_ratio'] = cash_ratio(script_data)
    primary_signals['cash_flow_debt_coverage'] = cash_flow_debt_coverage(script_data)
    primary_signals['percentage_debt_change_one_year'] = percentage_debt_change_one_year(script_data)
    primary_signals['ebitda_interest_coverage'] = ebitda_interest_coverage(script_data)
    primary_signals['total_coverage'] = total_coverage(script_data)
    primary_signals['degree_of_financial_leverage'] = degree_of_financial_leverage(script_data)
    primary_signals['debt_to_assets'] = debt_to_assets(script_data)
    primary_signals['debt_to_equity'] = debt_to_equity(script_data)

    ############# Estimates #############

    primary_signals['rating_revision_100d'] = rating_revision_100d(script_data)
    primary_signals['revenue_growth_estimate'] = revenue_growth_estimate(script_data)
    primary_signals['eps_estimates_range'] = eps_estimates_range(script_data)
    primary_signals['target_price_estimate'] = target_price_estimate(script_data)
    primary_signals['rating_revision_20d'] = rating_revision_20d(script_data)
    primary_signals['earning_to_price_fwd_estimate'] = earning_to_price_fwd_estimate(script_data)
    primary_signals['eps_revisions_one_qtr'] = eps_revisions_one_qtr(script_data)
    primary_signals['eps_revisions_two_qtr'] = eps_revisions_two_qtr(script_data)
    primary_signals['eps_revisions_one_year'] = eps_revisions_one_year(script_data)
    primary_signals['eps_revision_dispersion_one_qtr'] = eps_revision_dispersion_one_qtr(script_data)
    primary_signals['eps_revision_dispersion_two_qtr'] = eps_revision_dispersion_two_qtr(script_data)
    primary_signals['eps_revision_dispersion_one_year'] = eps_revision_dispersion_one_year(script_data)

    ############# Value #############

    primary_signals['book_to_price'] = book_to_price(script_data)
    primary_signals['dividend_yield'] = dividend_yield(script_data)
    primary_signals['earnings_to_price'] = earnings_to_price(script_data)
    primary_signals['earnings_to_price_estimate'] = earnings_to_price_estimate(script_data)
    primary_signals['cash_flows_to_price'] = cash_flows_to_price(script_data)
    primary_signals['free_cash_flows_to_price_estimate'] = free_cash_flows_to_price_estimate(script_data)
    primary_signals['earning_yield'] = earning_yield(script_data)
    primary_signals['earning_yield_estimate'] = earning_yield_estimate(script_data)
    primary_signals['sales_to_price'] = sales_to_price(script_data)
    primary_signals['sales_to_price_estimate'] = sales_to_price_estimate(script_data)

    ############# Momentum #############

    primary_signals['one_month_abs_momentum'] = one_month_abs_momentum(script_data)
    primary_signals['one_month_momentum_vol_adjusted'] = one_month_momentum_vol_adjusted(script_data)
    primary_signals['one_month_residual_momentum'] = one_month_residual_momentum(script_data)
    primary_signals['one_month_residual_momentum_vol_adjusted'] = one_month_residual_momentum_vol_adjusted(script_data)
    primary_signals['abs_price_momentum_one_year'] = abs_price_momentum_one_year(script_data)
    primary_signals['residual_price_momentum_one_year'] = residual_price_momentum_one_year(script_data)
    primary_signals['abs_price_momentum_one_qtr'] = abs_price_momentum_one_qtr(script_data)
    primary_signals['residual_price_momentum_one_qtr'] = residual_price_momentum_one_qtr(script_data)

    ############# Volatility #############

    primary_signals['volatility_one_year'] = volatility_one_year(script_data)
    primary_signals['volatility_one_quarter'] = volatility_one_quarter(script_data)
    primary_signals['volatility_two_year'] = volatility_two_year(script_data)
    primary_signals['stability_one_year'] = stability_one_year(script_data)

    ############# Additional Risk Factors #############

    primary_signals['market_cap'] = market_cap(script_data)
    primary_signals['assets_to_inventory'] = assets_to_inventory(script_data)
    primary_signals['free_cash_flow_to_price'] = free_cash_flow_to_price(script_data)
    primary_signals['operating_income_to_enterprise_value'] = operating_income_to_enterprise_value(script_data)
    primary_signals['de_leveraging'] = de_leveraging(script_data)
    primary_signals['dividend_payout_ratio'] = dividend_payout_ratio(script_data)
    primary_signals['short_term_momentum'] = short_term_momentum(script_data)
    primary_signals['long_term_momentum'] = long_term_momentum(script_data)

    primary_signals = primary_signals.reset_index()
    primary_signals.trade_date = pd.to_datetime(primary_signals.trade_date)
    primary_signals.trade_date = primary_signals.trade_date.dt.date
    primary_signals['equities_hash'] = primary_signals.apply(hash_equities_by_row, axis=1)
    return primary_signals
