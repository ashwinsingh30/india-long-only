import datetime
import hashlib

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from config.PulsePlatformConfig import get_pulse_platform_config
from database.finders.UniverseFinder import get_universe_constituents_between_dates
from utils.DateUtils import get_month_year_of_date, contract_expiry_suffix

config = get_pulse_platform_config()


def options_composite_hash(strike, expiry, trade_date, option_type, script_name):
    strike = '%.5f' % strike * 100
    return hashlib.sha1((script_name + "_" + str(strike) + "_" + str(expiry) + "_" + str(trade_date) + "_" + str(
        option_type)).encode('utf-8')).hexdigest()


def futures_composite_hash(expiry, trade_date, script_name):
    return hashlib.sha1((script_name + "_" + str(expiry) + "_" + str(trade_date)).encode('utf-8')).hexdigest()


def equities_composite_hash(trade_date, script_name):
    return hashlib.sha1((script_name + "_" + str(trade_date)).encode('utf-8')).hexdigest()


def corporate_action_composite_hash(script_name, ex_date, actions_type, action_category):
    return hashlib.sha1((script_name + "_" + str(ex_date) + "_" + str(actions_type) + "_" + str(action_category))
                        .encode('utf-8')).hexdigest()


def cluster_composite_hash(callibration_date, script_name, cluster_name):
    return hashlib.sha1((script_name + "_" + str(callibration_date) + "_" + str(cluster_name)).encode('utf-8')) \
        .hexdigest()


def model_performance_hash(trade_date, model_name):
    return hashlib.sha1((model_name + "_" + str(trade_date)).encode('utf-8')).hexdigest()


def group_returns_hash(trade_date, group_name):
    return hashlib.sha1((group_name + "_" + str(trade_date)).encode('utf-8')).hexdigest()


def alpha_performance_hash(trade_date, alpha_name):
    return hashlib.sha1((alpha_name + "_" + str(trade_date)).encode('utf-8')).hexdigest()


def factor_weights_hash(as_of_date, factor, model):
    return hashlib.sha1((factor + "_" + str(as_of_date) + "_" + str(model)).encode('utf-8')).hexdigest()


def equities_as_of_hash(trade_date, script_name, model_id, return_driver):
    return hashlib.sha1(
        (script_name + "_" + str(trade_date) + str(model_id) + return_driver).encode('utf-8')).hexdigest()


def override_composite_hash(start_date, expiry_date, script_name):
    return hashlib.sha1((script_name + "_" + str(start_date) + "_" + str(expiry_date)).encode('utf-8')).hexdigest()


def securities_master_composite_hash(script_name, end_date):
    return hashlib.sha1((script_name + "_" + str(end_date)).encode('utf-8')).hexdigest()


def universe_composite_hash(script_name, as_of_date, universe_name):
    return hashlib.sha1((script_name + "_" + str(as_of_date) + "_" + str(universe_name)).encode('utf-8')).hexdigest()


def hash_equities_by_as_of_date(row):
    return equities_composite_hash(row['as_of_date'], row['script_name'])


def hash_model_performance_by_row(row):
    return model_performance_hash(row['trade_date'], row['model_name'])


def hash_alpha_performance_by_row(row):
    return alpha_performance_hash(row['trade_date'], row['alpha_name'])


def hash_factor_weights_by_row(row):
    return factor_weights_hash(row['as_of_date'], row['factor'], row['model'])


def hash_equities_by_row(row):
    return equities_composite_hash(row['trade_date'], row['script_name'])


def hash_futures_by_row(row):
    return futures_composite_hash(row['expiry'], row['trade_date'], row['script_name'])


def hash_options_by_row(row):
    return options_composite_hash(row['strike_price'], row['expiry'], row['trade_date'], row['option_type'],
                                  row['script_name'])


def hash_corporate_actions_by_row(row):
    return corporate_action_composite_hash(row['script_name'], row['ex_date'], row['action_type'],
                                           row['action_category'])


def hash_equities_as_of_by_row(row):
    return equities_as_of_hash(row['as_of_date'], row['script_name'], row['model_id'], row['return_driver'])


def hash_overrides_by_row(row):
    return override_composite_hash(row['start_date'], row['expiry_date'], row['script_name'])


def hash_securities_by_row(row):
    return securities_master_composite_hash(row['script_name'], row['end_date'])


def hash_universe_by_row(row):
    return universe_composite_hash(row['script_name'], row['as_of_date'], row['universe_name'])


def hash_group_returns_by_row(row):
    return group_returns_hash(row['trade_date'], row['group_name'])


def equity_hash_list(security_list, date):
    hash_list = []
    for security in security_list:
        hash_list.append(equities_composite_hash(date, security))
    return hash_list


def get_contract_name(row):
    expiry_suffix = contract_expiry_suffix(row['expiry'])
    return row['script_name'] + expiry_suffix + 'FUT'


def find_contract_expiry(row, expiry_suffixes):
    contract_suffix = row['contract'].replace(row['script_name'], '').replace('FUT', '')
    if contract_suffix == expiry_suffixes['near']:
        return 'near'
    if contract_suffix == expiry_suffixes['mid']:
        return 'mid'
    if contract_suffix == expiry_suffixes['far']:
        return 'far'


def parseDate(date):
    return datetime.datetime.strptime(date, '%Y-%m-%d').date()


def parse_excel_date(date):
    date = date.split("/")
    m = date[1]
    d = date[0]
    y = "20" + date[2]
    return datetime.datetime.strptime(m + d + y, "%m%d%Y").date()


def parse_float(value):
    try:
        return float(value)
    except ValueError:
        return np.nan


def parse_int(value):
    try:
        return int(value)
    except ValueError:
        return np.nan


def select_best_underlier(performance_df):
    alphas = performance_df.sort_values('info_ratio', ascending=False).index
    selected_alphas = []
    selected_underliers = []
    for alpha in alphas:
        underlier_alpha = alpha.replace('_conj', '')
        if underlier_alpha not in selected_underliers:
            selected_alphas.append(alpha)
            selected_underliers.append(underlier_alpha)
            selected_underliers = list(set(selected_underliers))
    return selected_alphas


def get_regression_model_weights():
    return pd.read_csv(config.project_directory + '/utils/model_configs/regression_weights.csv')


def get_regression_fundamental_model_weights():
    return pd.read_csv(config.project_directory + '/utils/model_configs/regression_fundamental.csv')


def get_factor_seasonal_weights():
    return pd.read_csv(config.project_directory + '/utils/model_configs/factor_monthly_weights.csv', index_col=[0])


def join_dfs_overlapping_columns(left, right, how=None):
    column_diff = right.columns.difference(left.columns)
    if how is not None:
        return left.join(right[column_diff], how=how)
    else:
        return left.join(right[column_diff])


def process_model_simulations(model_data):
    model_data.trade_date = model_data.trade_date.apply(parse_excel_date)
    return model_data.set_index('trade_date')


def weight_vector_to_model_weights(weight_vector, model_name, trade_date):
    weight_df = pd.DataFrame(index=weight_vector.index)
    weight_df.index.name = 'script_name'
    weight_df['weight'] = weight_vector
    weight_df['model_name'] = model_name
    weight_df['trade_date'] = trade_date
    return weight_df.reset_index()


def expand_universe_df(data, all_dates):
    later_dates = all_dates[np.where(all_dates > data['as_of_date'].max())]
    if len(later_dates) != 0:
        next_update_date = later_dates[0]
        last_entry = data.iloc[-1].copy()
        last_entry['as_of_date'] = next_update_date
        if not (last_entry.empty or last_entry.isnull().all()):
            data = pd.concat([data, last_entry.to_frame().T.dropna(axis=1)], axis=0)
        dates = pd.DataFrame({'trade_date': pd.date_range(data['as_of_date'].min(),
                                                          data['as_of_date'].max(), freq='D', inclusive='left')}) \
            .set_index('trade_date')
        dates = pd.DataFrame(dates, columns=data.script_name.unique())
        return dates.unstack().reset_index().rename(columns={'level_0': 'script_name'})[['trade_date', 'script_name']]
    else:
        return pd.DataFrame()


def expand_trade_date(ser, close_side=None):
    if close_side is None:
        close_side = 'right'
    return pd.DataFrame({'trade_date': pd.date_range(ser['trade_date'].min(),
                                                     ser['trade_date'].max(), freq='D', closed=close_side)})


def expand_as_of_date(ser, close_side=None):
    if close_side is None:
        close_side = 'right'
    return pd.DataFrame({'trade_date': pd.date_range(ser['as_of_date'].min(),
                                                     ser['as_of_date'].max(), freq='D', closed=close_side)})


def hash_equities_df(row):
    return equities_composite_hash(row['trade_date'].date(), row['script_name'])


def get_daily_sampled_nse_100_universe(start_date, end_date):
    universe_data = get_universe_constituents_between_dates('NSE100', start_date, end_date)
    if end_date > universe_data.as_of_date.max():
        current_data = universe_data.loc[universe_data['as_of_date'] == universe_data.as_of_date.max()].copy()
        current_data['as_of_date'] = end_date + relativedelta(days=1)
        universe_data = universe_data.append(current_data, ignore_index=True).reset_index()
    universe_data = universe_data.sort_values('as_of_date')
    dates = universe_data.as_of_date.unique()
    universe_data = universe_data.groupby(['as_of_date']).apply(expand_universe_df, all_dates=dates).reset_index()[
        ['script_name', 'trade_date']]
    universe_data.trade_date = pd.to_datetime(universe_data.trade_date)
    universe_data.trade_date = universe_data.trade_date.dt.date
    universe_data['equities_hash'] = universe_data.apply(hash_equities_by_row, axis=1)
    return universe_data.set_index('equities_hash')


def get_daily_sampled_nse_500_universe(start_date, end_date):
    universe_data = get_universe_constituents_between_dates('NSE500', start_date, end_date)
    if end_date > universe_data.as_of_date.max():
        current_data = universe_data.loc[universe_data['as_of_date'] == universe_data.as_of_date.max()].copy()
        current_data['as_of_date'] = end_date + relativedelta(days=1)
        universe_data = pd.concat([universe_data, current_data], axis=0).reset_index()
    universe_data = universe_data.sort_values('as_of_date')
    dates = universe_data.as_of_date.unique()
    universe_data = universe_data.groupby(['as_of_date']).apply(expand_universe_df, all_dates=dates).reset_index()[
        ['script_name', 'trade_date']]
    universe_data.trade_date = pd.to_datetime(universe_data.trade_date)
    universe_data.trade_date = universe_data.trade_date.dt.date
    universe_data['equities_hash'] = universe_data.apply(hash_equities_by_row, axis=1)
    return universe_data.set_index('equities_hash')


def get_daily_sampled_nse_200_universe(trade_date, start_date=None):
    universe_data = pd.read_csv(config.project_directory + '/utils/NSE200_MonthlySampled.csv', index_col=0)
    universe_data.trade_date = universe_data.trade_date.apply(parseDate)
    if trade_date > universe_data.trade_date.max():
        current_data = universe_data.loc[universe_data['trade_date'] == universe_data.trade_date.max()].copy()
        current_data['trade_date'] = trade_date + relativedelta(days=1)
        universe_data = universe_data.append(current_data, ignore_index=True).reset_index()
    universe_data = universe_data.sort_values('trade_date')
    dates = universe_data.trade_date.unique()
    universe_data = universe_data.rename(columns={'trade_date': 'as_of_date'})
    universe_data = universe_data.groupby(['as_of_date']).apply(expand_universe_df, all_dates=dates).reset_index()[
        ['script_name', 'trade_date']]
    universe_data.trade_date = pd.to_datetime(universe_data.trade_date)
    universe_data.trade_date = universe_data.trade_date.dt.date
    universe_data['equities_hash'] = universe_data.apply(hash_equities_by_row, axis=1)
    if start_date is not None:
        return universe_data[(universe_data.trade_date >= start_date) &
                             (universe_data.trade_date <= trade_date)].set_index('equities_hash')
    else:
        return universe_data[universe_data.trade_date == trade_date].set_index('equities_hash')


def pivot_model_performance(performance_df):
    return performance_df.pivot_table(index='trade_date', columns='model_name', values='model_return')


def pivot_model_turnover(performance_df):
    return performance_df.pivot_table(index='trade_date', columns='model_name', values='model_turnover').mean()


def get_daily_sampled_fno_universe(trade_date, start_date):
    data = pd.read_pickle(config.base_directory + '/meta_data/fno_universe.pkl')
    trade_dates = data[['trade_date']]
    trade_dates['month_year'] = trade_dates.trade_date.apply(get_month_year_of_date)
    sampling_dates = trade_dates.groupby('month_year').min().trade_date.unique()
    universe_data = data[data.trade_date.isin(sampling_dates)].copy()
    if trade_date > universe_data.trade_date.max():
        current_data = universe_data.loc[universe_data['trade_date'] == universe_data.trade_date.max()].copy()
        current_data['trade_date'] = trade_date + relativedelta(days=1)
        universe_data = universe_data.append(current_data, ignore_index=True).reset_index()
    universe_data = universe_data.sort_values('trade_date')
    dates = universe_data.trade_date.unique()
    universe_data = universe_data.rename(columns={'trade_date': 'as_of_date'})
    universe_data = universe_data.groupby(['as_of_date']).apply(expand_universe_df, all_dates=dates).reset_index()[
        ['script_name', 'trade_date']]
    universe_data.trade_date = pd.to_datetime(universe_data.trade_date)
    universe_data.trade_date = universe_data.trade_date.dt.date
    universe_data['equities_hash'] = universe_data.apply(hash_equities_by_row, axis=1)
    if start_date is not None:
        return universe_data[(universe_data.trade_date >= start_date) &
                             (universe_data.trade_date <= trade_date)].set_index('equities_hash')
    else:
        return universe_data[universe_data.trade_date == trade_date].set_index('equities_hash')
