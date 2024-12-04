import os

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from backtest.config.BacktestConfig import get_pulse_platform_backtest_config
from config.ConfiguredLogger import get_logger
from database.connection.DbConnection import get_pulse_db_connection
from database.domain.EquitiesPriceData import EquitiesPriceData
from database.domain.PrimarySignals import PrimarySignals
from database.domain.TradingConstraints import TradingConstraints
from database.domain.TrendSignals import TrendSignals

from utils.TradingPlatformUtils import equity_hash_list, join_dfs_overlapping_columns

config = get_pulse_platform_backtest_config()
log = get_logger(os.path.basename(__file__), '/back_test.log')
dbConnection = get_pulse_db_connection()


adhoc_signal = pd.read_pickle(r'D:\Project\trading-platform-longonly\backtest\datascratch\TrueBeatsIN.pkl')

def get_equities_price_data_between_dates(data_start_date, end_date, stock_list):
    return pd.read_sql(dbConnection.session.query(EquitiesPriceData)
                       .filter((EquitiesPriceData.trade_date >= data_start_date) &
                               (EquitiesPriceData.trade_date <= end_date) &
                               (EquitiesPriceData.script_name.in_(stock_list)))
                       .statement, dbConnection.session.bind) \
        .set_index("equities_hash")


def get_trading_constraints_between_dates(data_start_date, end_date, stock_list):
    return pd.read_sql(dbConnection.session.query(TradingConstraints)
                       .filter((TradingConstraints.trade_date >= data_start_date) &
                               (TradingConstraints.trade_date <= end_date) &
                               (TradingConstraints.script_name.in_(stock_list)))
                       .statement, dbConnection.session.bind) \
        .set_index("equities_hash")


def get_primary_signals_between_dates(data_start_date, end_date, stock_list):
    return pd.read_sql(dbConnection.session.query(PrimarySignals)
                       .filter((PrimarySignals.trade_date >= data_start_date) &
                               (PrimarySignals.trade_date <= end_date) &
                               (PrimarySignals.script_name.in_(stock_list)))
                       .statement, dbConnection.session.bind) \
        .set_index("equities_hash")


def get_trend_signals_between_dates(data_start_date, end_date, stock_list):
    return pd.read_sql(dbConnection.session.query(TrendSignals)
                       .filter((TrendSignals.trade_date >= data_start_date) &
                               (TrendSignals.trade_date <= end_date) &
                               (TrendSignals.script_name.in_(stock_list)))
                       .statement, dbConnection.session.bind) \
        .set_index("equities_hash")

def get_trade_dates(start_date, end_date):
    return pd.read_sql(dbConnection.session.query(EquitiesPriceData.trade_date)
                       .filter((EquitiesPriceData.trade_date >= start_date) &
                               (EquitiesPriceData.trade_date <= end_date))
                       .distinct()
                       .statement, dbConnection.session.bind)['trade_date'].sort_values().unique()


def get_trade_date_before_X_days(df, trade_date, x):
    trade_dates = df['trade_date']
    return np.min(trade_dates[trade_dates <= trade_date].sort_values(ascending=False).unique()[0:x])


def get_trade_date_before_X_days_from_index(df, trade_date, x):
    trade_dates = df.index
    return np.min(trade_dates[trade_dates <= trade_date].sort_values(ascending=False).unique()[0:x])


def get_latest_trade_date(df, trade_date):
    trade_dates = df['trade_date']
    return np.max(trade_dates[trade_dates <= trade_date].unique())


def populate_close_price_df(data_start_date, end_date, stock_list):
    price_data = get_equities_price_data_between_dates(data_start_date, end_date, stock_list)
    return price_data.pivot_table(index='trade_date', columns='script_name', values='close_price')


def get_trade_date_after_X_days_from_index(df, trade_date, x):
    trade_dates = df.trade_date
    return np.max(trade_dates[trade_dates >= trade_date].sort_values().unique()[0:x])


class EquitiesBacktestDataScratch():
    stock_list = []
    back_test_start_date = config.start_date
    back_test_end_date = config.end_date
    look_back_days = config.look_back_days
    equities_data = pd.DataFrame()
    trade_dates = []
    close_price_df = pd.DataFrame()

    def __init__(self, stock_list, start_date=None, end_date=None, look_back_days=None):
        log.info("Initializing Data Scratch")
        if start_date is None:
            start_date = self.back_test_start_date
        if end_date is None:
            end_date = self.back_test_end_date
        self.stock_list = stock_list
        self.trade_dates = get_trade_dates(start_date, end_date)
        self.refresh_equities_data_scratch(start_date, start_date + relativedelta(months=2), 5)
        self.close_price_df = populate_close_price_df(start_date - relativedelta(years=1), end_date, stock_list)
        log.info("Data Scratch Initialized")

    def refresh_equities_data_scratch(self, start_date, end_date, look_back_days):
        log.info("Refreshing Equities Data Buffer .....")
        data_start_date = start_date - relativedelta(days=2 * look_back_days)
        equities_price_data = get_equities_price_data_between_dates(data_start_date, end_date, self.stock_list)
        trading_constraints = get_trading_constraints_between_dates(data_start_date, end_date, self.stock_list)
        primary_signals = get_primary_signals_between_dates(data_start_date, end_date, self.stock_list)
        trend_signals = get_trend_signals_between_dates(data_start_date, end_date, self.stock_list)
        self.equities_data = join_dfs_overlapping_columns(equities_price_data, trend_signals)
        self.equities_data = join_dfs_overlapping_columns(self.equities_data, trading_constraints)
        self.equities_data = join_dfs_overlapping_columns(self.equities_data, primary_signals)
        self.equities_data = join_dfs_overlapping_columns(self.equities_data, adhoc_signal)
        log.info("Equities Data Buffer Refreshed")

    def get_price_data_for_stock_last_X_days(self, stock, trade_date, x):
        stock_data = self.equities_data[(self.equities_data['script_name'] == stock) &
                                        (self.equities_data['trade_date'] < trade_date)]
        stock_data = stock_data.sort_values('trade_date', ascending=False)
        return stock_data.head(x)

    def get_price_data_for_securities_last_X_days(self, securities, trade_date, x):
        start_date = get_trade_date_before_X_days(self.equities_data, trade_date, x)
        return self.equities_data[((self.equities_data['script_name'].isin(securities)) &
                                   (self.equities_data['trade_date'] <= trade_date) &
                                   (self.equities_data['trade_date'] >= start_date))]

    def get_latest_available_price_securities(self, securities, trade_date):
        latest_date = get_latest_trade_date(self.equities_data, trade_date)
        stock_data = self.equities_data[(self.equities_data['script_name'].isin(securities)) &
                                        (self.equities_data['trade_date'] == latest_date)]
        return stock_data.sort_values('trade_date', ascending=False)

    def get_latest_available_price(self, trade_date):
        latest_date = get_latest_trade_date(self.equities_data, trade_date)
        stock_data = self.equities_data[(self.equities_data['trade_date'] == latest_date)]
        return stock_data.sort_values('trade_date', ascending=False)

    def is_trading_day(self, date):
        if date in self.equities_data['trade_date'].values:
            return True
        else:
            return False

    def next_trading_day(self, date):
        later_dates = self.equities_data[self.equities_data['trade_date'] >= date]['trade_date'].unique()
        return np.min(later_dates)

    def previous_trading_day(self, date):
        later_dates = self.equities_data[self.equities_data['trade_date'] <= date]['trade_date'].unique()
        return np.max(later_dates)

    def get_prices_for_date_security_list(self, security_list, trade_date):
        hash_list = equity_hash_list(security_list, trade_date)
        hash_list = np.intersect1d(hash_list, self.equities_data.index)
        return self.equities_data.loc[hash_list].set_index('script_name')

    def get_close_prices_for_date_security_list(self, security_list, trade_date):
        hash_list = equity_hash_list(security_list, trade_date)
        hash_list = np.intersect1d(hash_list, self.equities_data.index)
        return self.equities_data.loc[hash_list].set_index('script_name')[['close_price']]

    def securities_with_valid_data(self, security_list, trade_date):
        day_data = self.equities_data[self.equities_data['trade_date'] == trade_date]
        return np.intersect1d(day_data['script_name'].values, security_list)

    def last_available_traded_price(self, security, trade_date):
        security_data = self.equities_data[(self.equities_data['script_name'] == security) &
                                           (self.equities_data['trade_date'] <= trade_date)]
        if not security_data.empty:
            return security_data.sort_values('trade_date', ascending=False).iloc[0]['close_price']
        else:
            return None

    def get_price_security_list_between_dates(self, security_list, start_date, end_date):
        return self.equities_data.loc[(self.equities_data.script_name.isin(security_list)) &
                                      (self.equities_data.trade_date >= start_date) &
                                      (self.equities_data.trade_date <= end_date)]

    def get_trading_dates(self, start_date, end_date):
        return self.equities_data[(self.equities_data['trade_date'] <= end_date) &
                                  (self.equities_data['trade_date'] >= start_date)]['trade_date'].sort_values().unique()

    def get_close_price_table(self, security_list, trade_date, look_back):
        start_date = get_trade_date_before_X_days_from_index(self.close_price_df, trade_date, look_back)
        price_table = self.close_price_df[(self.close_price_df.index >= start_date) &
                                          (self.close_price_df.index <= trade_date)]
        return price_table.reindex(security_list, axis=1)

    def get_close_price_table_between_dates(self, security_list, start_date, end_date):
        price_table = self.close_price_df[(self.close_price_df.index >= start_date) &
                                          (self.close_price_df.index <= end_date)]
        return price_table.reindex(security_list, axis=1)

    def get_close_price_before_X_days(self, security_list, trade_date, x):
        start_date = get_trade_date_before_X_days_from_index(self.close_price_df, trade_date, x)
        price_table = self.close_price_df.loc[start_date]
        return price_table.reindex(security_list, axis=1)

    def get_n_day_forward_price(self, securities, trade_date, n):
        forward_date = get_trade_date_after_X_days_from_index(self.equities_data, trade_date, n)
        return self.equities_data[((self.equities_data['script_name'].isin(securities)) &
                                   (self.equities_data['trade_date'] >= trade_date) &
                                   (self.equities_data['trade_date'] <= forward_date))]
