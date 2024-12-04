import concurrent.futures
import os
import time

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from config.ConfiguredLogger import get_logger
from data_load.capitaliq.CapitalIQClient import CapIQClient
from database.finders.SecuritiesMasterFinder import get_all_active_securities
from database.persistence.AnnouncementDatesPersistence import persist_announcement_dates
from database.persistence.PrimaryDataPersistence import persist_primary_data
from utils.Constants import ciq_mnemonics_map
from utils.DateUtils import parse_announcement_date
from utils.TradingPlatformUtils import hash_equities_by_row

log = get_logger(os.path.basename(__file__))


def remove_suffix(value):
    return value.replace(':I', '')


def pull_data_ciq_api(identifiers, fields, date):
    # ciq_client = CapIQClient("apiadmin@alpha-grep.com", "AlphaGrep@1313")
    ciq_client = CapIQClient("Apiadmin@s2adv.com", "Tara1234")
    data = ciq_client.request_point_in_time_ltm_data(ciq_client, identifiers, fields, date)
    data = pd.DataFrame(data).T.reset_index()
    data['trade_date'] = date.date()
    return data


def pull_data_ciq_api_for_period_type(identifiers, fields, date, period_type):
    ciq_client = CapIQClient("Apiadmin@s2adv.com", "Tara1234")
    if period_type == 'NO_PROPS':
        data = ciq_client.request_point_in_time_data_no_props(ciq_client, identifiers, fields, date)
    else:
        data = ciq_client.request_point_in_time_data_for_period_type(ciq_client, identifiers, fields, date, period_type)
    data = pd.DataFrame(data).T.reset_index()
    data['trade_date'] = date.date()
    return data


def run_ciq_data_pull(params):
    return pull_data_ciq_api(*params)


def run_ciq_data_pull_for_period_type(params):
    return pull_data_ciq_api_for_period_type(*params)


def populate_fundamental_data(isin_script_name_map, fields, end_date, start_date=None, persist_parts=False):
    log.info('Starting Fundamental Data Update ......')
    if start_date is not None:
        dates = pd.date_range(start_date, end_date - relativedelta(days=1), freq='D')
    else:
        dates = [pd.to_datetime(end_date)]
    date_splits = np.array_split(dates, np.ceil(len(dates) / 20))
    log.info('Request broken into %s parts', len(date_splits))
    combined_data_df = pd.DataFrame()
    for split in date_splits:
        log.info('Starting update for %s to %s', str(split[0].strftime("%m/%d/%Y")),
                 str(split[-1].strftime("%m/%d/%Y")))
        data_df = pd.DataFrame()
        isins = np.unique(isin_script_name_map.index.values)
        total_securities = len(isins)
        max_scripts_per_request = int(450 / len(fields))
        threads = int(np.ceil(total_securities / max_scripts_per_request))
        request_params = []
        for i in range(0, threads):
            thread_identifiers = isins[i * max_scripts_per_request: (i + 1) * max_scripts_per_request]
            thread_identifiers = [identifier + ":I" for identifier in thread_identifiers]
            for date in split:
                request_params.append((thread_identifiers, fields, date))

        with concurrent.futures.ThreadPoolExecutor(max_workers=threads * len(split)) as executor:
            future = list(executor.map(run_ciq_data_pull, request_params))
            data_df = data_df.append(future)
        data_df = data_df.rename(columns={'index': 'isin'})
        data_df['isin'] = data_df['isin'].apply(remove_suffix)
        data_df = data_df.join(isin_script_name_map['script_name'], on='isin')
        data_df = data_df.drop('isin', axis=1)
        data_df = data_df.replace(to_replace='Data Unavailable', value=np.nan)
        data_df['equities_hash'] = data_df.apply(hash_equities_by_row, axis=1)
        if persist_parts:
            data_df.to_csv('Long Only - Sep.csv')
            persist_primary_data(data_df.set_index('equities_hash'))
        combined_data_df = combined_data_df.append(data_df)
        log.info('Update for %s to %s completed', str(split[0].strftime("%m/%d/%Y")),
                 str(split[-1].strftime("%m/%d/%Y")))
    return combined_data_df.set_index('equities_hash')


def parallel_data_pull_function(split, isin_script_name_map, fields, period_type):
    log.info('Starting update for %s to %s for %s period fields', str(split[0].strftime("%m/%d/%Y")),
             str(split[-1].strftime("%m/%d/%Y")), period_type)
    data_df = pd.DataFrame()
    isins = np.unique(isin_script_name_map.index.values)
    total_securities = len(isins)
    request_params = []
    max_scripts_per_request = int(500 / len(fields))
    threads = int(np.ceil(total_securities / max_scripts_per_request))
    for i in range(0, threads):
        thread_identifiers = isins[i * max_scripts_per_request: (i + 1) * max_scripts_per_request]
        thread_identifiers = [identifier + ":I" for identifier in thread_identifiers]
        for date in split:
            request_params.append((thread_identifiers, fields, date, period_type))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2 * threads * len(split)) as executor:
        future = list(executor.map(run_ciq_data_pull_for_period_type, request_params))
        data_df = pd.concat(future, axis=0)
    data_df = data_df.rename(columns={'index': 'isin'})
    data_df['isin'] = data_df['isin'].apply(remove_suffix)
    data_df = data_df.join(isin_script_name_map['script_name'], on='isin')
    data_df = data_df.drop('isin', axis=1)
    data_df = data_df.replace(to_replace='Data Unavailable', value=np.nan)
    return data_df


def parallel_data_pull_function_retries(split, isin_script_name_map, fields, period_type):
    # try:
        return parallel_data_pull_function(split, isin_script_name_map, fields, period_type)
    # except Exception as e:
    #     log.error('Parallel data pull failing, Retrying...')
    #     log.error(e)
    #     time.sleep(10)
    #     return parallel_data_pull_function_retries(split, isin_script_name_map, fields, period_type)


def populate_primary_data_different_period_type(isin_script_name_map, field_period_map, end_date, start_date=None,
                                                persist_parts=False):
    log.info('Starting Fundamental Data Update ......')
    if start_date is not None:
        dates = pd.date_range(start_date, end_date - relativedelta(days=1), freq='D')
    else:
        dates = [pd.to_datetime(end_date)]
    date_splits = np.array_split(dates, np.ceil(len(dates) / 10))
    combined_data_df = pd.DataFrame()
    groups = field_period_map['group'].unique()
    log.info('Request broken into %s parts', len(date_splits) * len(groups))
    for split in date_splits:
        date_split_df = pd.DataFrame()
        for group in groups:
            group_values = field_period_map[field_period_map['group'] == group]
            period_type_fields = group_values.index.values
            period_type = group_values['period_type'].unique()[0]
            period_type_fields = [a.lower() for a in period_type_fields]
            if '+' in period_type:
                period_suffix = '_' + period_type.replace('IQ_', '').replace('+', '_').lower()
                rename_dict = {a: a + period_suffix for a in period_type_fields}
            else:
                rename_dict = {}
            data_df = parallel_data_pull_function_retries(split, isin_script_name_map, period_type_fields, period_type)
            data_df = data_df.set_index(['script_name', 'trade_date']).rename(columns=rename_dict)
            if date_split_df.empty:
                date_split_df = data_df
            else:
                date_split_df = date_split_df.join(data_df, how='outer', rsuffix='_r')
        date_split_df = date_split_df.reset_index()
        date_split_df.trade_date = pd.to_datetime(date_split_df.trade_date)
        date_split_df.trade_date = date_split_df.trade_date.dt.date
        date_split_df['equities_hash'] = date_split_df.apply(hash_equities_by_row, axis=1)
        if persist_parts:
            persist_primary_data(date_split_df.set_index('equities_hash'))
            continue
        combined_data_df = combined_data_df.append(date_split_df, ignore_index=True)
    if persist_parts:
        return
    return combined_data_df.set_index('equities_hash')


def populate_fundamental_data_retries(isin_script_name_map, fields, end_date, start_date=None, persist_parts=False):
    try:
        return populate_fundamental_data(isin_script_name_map, fields, end_date, start_date, persist_parts)
    except Exception as e:
        log.error('Fundamental data pull failing, Retrying...')
        log.error(e)
        time.sleep(1)
        return populate_fundamental_data(isin_script_name_map, fields, end_date, start_date, persist_parts)


def get_next_earnings_announcements(trade_date):
    isin_script_name_map = get_all_active_securities()[['script_name']]
    announcement_dates = populate_fundamental_data_retries(isin_script_name_map, ['IQ_NEXT_EARNINGS_DATE'], trade_date,
                                                           start_date=None, persist_parts=False)
    announcement_dates = announcement_dates.dropna().rename(columns={'iq_next_earnings_date': 'next_announcement_date'})
    if not announcement_dates.empty:
        announcement_dates['next_announcement_date'] = announcement_dates['next_announcement_date'] \
            .apply(parse_announcement_date)
        announcement_dates = announcement_dates[['script_name', 'next_announcement_date']].reset_index(drop=True)
        announcement_dates['trade_date'] = trade_date
        persist_announcement_dates(announcement_dates, trade_date)


def populate_primary_data_for_date(trade_date, securities=None):
    isin_map = get_all_active_securities()[['script_name']]
    if securities is not None:
        isin_map = isin_map[isin_map['script_name'].isin(securities)]
    mnemonic_map = ciq_mnemonics_map['period_type'].fillna("")
    log.info('Populating fundamental data for %s', trade_date)
    populate_primary_data_different_period_type(isin_map, mnemonic_map, trade_date, persist_parts=True)
    log.info('Fundamental data population for %s completed successfully', trade_date)
