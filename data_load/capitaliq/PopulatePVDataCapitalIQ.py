import concurrent.futures
import os
import time

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from config.ConfiguredLogger import get_logger
from data_load.capitaliq.CapitalIQClient import CapIQClient
from database.persistence.EquitiesPriceDataPersistence import persist_equities_data
from utils.TradingPlatformUtils import hash_equities_by_row

log = get_logger(os.path.basename(__file__))


def remove_suffix(value):
    return value.replace(':I', '')


def pull_pv_data_ciq_api(identifiers, fields, date):
    ciq_client = CapIQClient("Apiadmin@s2adv.com", "Tara1234")
    data = ciq_client.request_pv_data(ciq_client, identifiers, fields, date)
    data = pd.DataFrame(data).T.reset_index()
    data['trade_date'] = date.date()
    return data


def run_ciq_pv_data_pull_for_period_type(params):
    return pull_pv_data_ciq_api(*params)


def run_step(threads, split, request_params):
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads * len(split)) as executor:
        future = list(executor.map(run_ciq_pv_data_pull_for_period_type, request_params))
    return future


def run_step_retries(threads, split, request_params):
    try:
        return run_step(threads, split, request_params)
    except Exception as e:
        log.error('PV Data pull failing, Retrying...')
        log.error(e)
        time.sleep(1)
        return run_step_retries(threads, split, request_params)


def populate_pv_data(isin_script_name_map, fields, end_date, start_date=None, persist_parts=False):
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
        future = run_step_retries(threads, split, request_params)
        data_df = pd.concat(future, axis=0)
        data_df = data_df.rename(columns={'index': 'isin'})
        data_df['isin'] = data_df['isin'].apply(remove_suffix)
        data_df = data_df.join(isin_script_name_map['script_name'], on='isin')
        data_df = data_df.drop('isin', axis=1)
        data_df = data_df.replace(to_replace='Data Unavailable', value=np.nan)
        data_df = data_df.rename(columns={'iq_closeprice': 'close_price',
                                          'iq_openprice': 'open_price',
                                          'iq_highprice': 'high_price',
                                          'iq_lowprice': 'low_price',
                                          'iq_volume': 'volume'})
        data_df = data_df.dropna(subset=['close_price', 'iq_closeprice_adj']).reset_index(drop=True)
        data_df[['open_price', 'high_price', 'low_price', 'close_price', 'volume', 'iq_closeprice_adj']] = \
            data_df[['open_price', 'high_price', 'low_price', 'close_price', 'volume', 'iq_closeprice_adj']].astype(
                float)
        data_df['adjustment_factor'] = data_df['iq_closeprice_adj'] / data_df['close_price']
        data_df[['open_price', 'high_price', 'low_price', 'close_price']] = data_df[['open_price', 'high_price',
                                                                                     'low_price', 'close_price']]. \
            multiply(data_df['adjustment_factor'], axis=0)
        data_df['volume'] = data_df['volume'] * 1E6
        data_df = data_df[
            ['script_name', 'trade_date', 'open_price', 'high_price', 'low_price', 'close_price', 'volume']]
        data_df['equities_hash'] = data_df.apply(hash_equities_by_row, axis=1)
        if persist_parts:
            persist_equities_data(data_df.set_index('equities_hash'))
        combined_data_df = pd.concat([combined_data_df, data_df], axis=0)
        log.info('Update for %s to %s completed', str(split[0].strftime("%m/%d/%Y")),
                 str(split[-1].strftime("%m/%d/%Y")))
    return combined_data_df.set_index('equities_hash')
