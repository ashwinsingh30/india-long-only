import datetime

import numpy as np
import pandas as pd
import pytz


def get_current_date_ist():
    ist = pytz.timezone('Asia/Calcutta')
    return datetime.datetime.now(ist).date()


def get_next_trade_date(trade_date):
    return trade_date + datetime.timedelta(days=7 - trade_date.weekday() if trade_date.weekday() > 3 else 1)


def get_month_year_of_date(trade_date):
    return str(trade_date.month) + '_' + str(trade_date.year)


def get_last_trade_date(trade_date):
    delta = 1
    if (trade_date.weekday() == 6):
        delta = 2
    if (trade_date.weekday() == 0):
        delta = 3
    return trade_date - datetime.timedelta(days=delta)


def parse_date(date):
    return datetime.datetime.strptime(date, "%Y-%m-%d").date()


def parse_text_date(date):
    date = date.split("-")
    if len(date) >= 3:
        day = date[0]
        month = date[1].upper()
        year = date[2]
        return str(datetime.datetime.strptime(month + day + year, "%b%d%Y").date())
    else:
        return None


def parse_expiry_date(date):
    return datetime.datetime.strptime(date, '%d-%b-%Y').date()


def parse_index_date(date):
    return datetime.datetime.strptime(date, '%d-%m-%Y').date()


def date_from_timestamp(timestamp):
    return timestamp.date()


def parse_time(time):
    return pd.Timestamp(time)


def expand_trade_dates(ser, close_side=None, end_date=None):
    if close_side is None:
        close_side = 'right'
    if end_date is None:
        end_date = np.max(ser)
    return pd.DataFrame({'trade_date': pd.date_range(np.min(ser),
                                                     end_date, freq='D', closed=close_side)})


def parse_announcement_date(date):
    date = date[0:11]
    if date[4] == ' ':
        date_list = list(date)
        date_list[4] = '0'
        date = ''.join(date_list)
    date = date.upper().replace(' ', '')
    return datetime.datetime.strptime(date, '%b%d%Y').date()


def get_month(date):
    return date.strftime("%B")


def contract_expiry_suffix(expiry_date):
    expiry_month = expiry_date.strftime('%b')
    expiry_year = int(expiry_date.strftime('%Y'))
    return str(expiry_year - 2000) + expiry_month.upper()
