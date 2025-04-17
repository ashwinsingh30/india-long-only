import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sqlalchemy import func

from backtest.config.BacktestConfig import get_pulse_platform_backtest_config
from backtest.datascratch.EquitiesBacktestDataScratch import EquitiesBacktestDataScratch
from config.PulsePlatformConfig import get_pulse_platform_config
from database.connection.DbConnection import get_pulse_db_connection
from database.domain.EquitiesPriceData import EquitiesPriceData
from database.finders.PrimarySignalsFinder import get_primary_signals_security_list_between_dates, \
    get_primary_signals_between_dates, get_primary_signals_neutralised_between_dates
from database.finders.StyleFactorsNSE500Finder import get_style_factors_nse_500_security_list_between_dates, \
    get_style_factors_nse_500_between_dates
from database.finders.TradingConstraintsFinder import get_trading_constraints_security_list_between_dates, \
    get_trading_constraints_between_dates
from database.finders.TrendSignalsFinder import get_trend_signals_security_list_between_dates, \
    get_trend_signals_between_dates
from utils.TradingPlatformUtils import equities_composite_hash, join_dfs_overlapping_columns, \
    get_daily_sampled_nse_500_universe

dbConnection = get_pulse_db_connection()
config = get_pulse_platform_config()

if config.run_mode == "backtest":
    test_config = get_pulse_platform_backtest_config()
    universe = get_daily_sampled_nse_500_universe(test_config.start_date, test_config.end_date)
    universe = universe.script_name.unique()
    universe = np.setdiff1d(universe, ['HEXAWARE'])
    universe = np.append(universe, ['NIFTY', 'NSE500'])
    data_scratch = EquitiesBacktestDataScratch(universe)
    back_test_mode = True
else:
    data_scratch = None
    back_test_mode = False


def get_equities_current_price(trade_date, script_name):
    index = equities_composite_hash(trade_date, script_name)
    return pd.read_sql(dbConnection.session.query(EquitiesPriceData)
                       .filter(EquitiesPriceData.equities_hash == index).statement, dbConnection.session.bind) \
        .set_index("equities_hash")


def get_complete_price_table_between_dates(start_date, end_date):
    return pd.read_sql(dbConnection.session.query(EquitiesPriceData)
                       .filter((EquitiesPriceData.trade_date <= end_date) &
                               (EquitiesPriceData.trade_date >= start_date))
                       .statement, dbConnection.session.bind)


def get_equities_price_for_securities(trade_date, script_list):
    index_list = []
    for script_name in script_list:
        index_list.append(equities_composite_hash(trade_date, script_name))
    return pd.read_sql(dbConnection.session.query(EquitiesPriceData)
                       .filter(EquitiesPriceData.equities_hash.in_(index_list)).statement, dbConnection.session.bind) \
        .set_index("equities_hash")


def get_equities_price_last_year_inclusive(script_name):
    x = 250
    return pd.read_sql(dbConnection.session.query(EquitiesPriceData)
                       .filter(EquitiesPriceData.script_name == script_name)
                       .order_by(EquitiesPriceData.trade_date.desc()).limit(x).statement, dbConnection.session.bind) \
        .set_index("equities_hash")


def get_equities_price_last_year_inclusive_for_date(script_name, trade_date):
    x = 250
    return pd.read_sql(dbConnection.session.query(EquitiesPriceData)
                       .filter((EquitiesPriceData.script_name == script_name) &
                               (EquitiesPriceData.trade_date <= trade_date))
                       .order_by(EquitiesPriceData.trade_date.desc()).limit(x).statement, dbConnection.session.bind) \
        .set_index("equities_hash")


def get_X_day_historical_price_stock(script_name, trade_date, x):
    if back_test_mode:
        return data_scratch.get_price_data_for_stock_last_X_days(script_name, trade_date, x)
    else:
        return pd.read_sql(dbConnection.session.query(EquitiesPriceData)
                           .filter(EquitiesPriceData.script_name == script_name)
                           .filter(EquitiesPriceData.trade_date < trade_date)
                           .order_by(EquitiesPriceData.trade_date.desc())
                           .limit(x).statement, dbConnection.session.bind) \
            .fillna(value=np.nan) \
            .set_index("equities_hash")


def get_date_before_x_days(trade_date, x):
    return pd.read_sql(dbConnection.session.query(EquitiesPriceData.trade_date)
                       .filter(EquitiesPriceData.trade_date < trade_date)
                       .distinct()
                       .order_by(EquitiesPriceData.trade_date.desc())
                       .limit(x)
                       .statement, dbConnection.session.bind).min()['trade_date']


def get_date_before_x_days_inclusive(trade_date, x):
    return pd.read_sql(dbConnection.session.query(EquitiesPriceData.trade_date)
                       .filter(EquitiesPriceData.trade_date <= trade_date)
                       .distinct()
                       .order_by(EquitiesPriceData.trade_date.desc())
                       .limit(x)
                       .statement, dbConnection.session.bind).min()['trade_date']


def get_latest_date(trade_date):
    return pd.read_sql(dbConnection.session.query(EquitiesPriceData.trade_date)
                       .filter(EquitiesPriceData.trade_date <= trade_date)
                       .distinct()
                       .statement, dbConnection.session.bind).max()['trade_date']


def get_X_day_historical_price_securities(securities, trade_date, x):
    if back_test_mode:
        return data_scratch.get_price_data_for_securities_last_X_days(securities, trade_date, x)
    else:
        start_date = get_date_before_x_days(trade_date, x)
        return pd.read_sql(dbConnection.session.query(EquitiesPriceData)
                           .filter(EquitiesPriceData.script_name.in_(securities))
                           .filter(EquitiesPriceData.trade_date < trade_date)
                           .filter(EquitiesPriceData.trade_date >= start_date)
                           .statement, dbConnection.session.bind) \
            .fillna(value=np.nan) \
            .set_index("equities_hash")


def get_X_day_historical_price_table(securities, trade_date, x):
    if back_test_mode:
        return data_scratch.get_close_price_table(securities, trade_date, x)
    else:
        start_date = get_date_before_x_days_inclusive(trade_date, x)
        price_data = pd.read_sql(dbConnection.session.query(EquitiesPriceData)
                                 .filter(EquitiesPriceData.script_name.in_(securities))
                                 .filter(EquitiesPriceData.trade_date <= trade_date)
                                 .filter(EquitiesPriceData.trade_date >= start_date)
                                 .statement, dbConnection.session.bind) \
            .fillna(value=np.nan) \
            .set_index("equities_hash")

        return price_data.pivot_table(index='trade_date', columns='script_name', values='close_price')


def get_historical_price_table_between_dates(securities, start_date, end_date):
    if back_test_mode:
        return data_scratch.get_close_price_table_between_dates(securities, start_date, end_date)
    else:
        price_data = pd.read_sql(dbConnection.session.query(EquitiesPriceData)
                                 .filter(EquitiesPriceData.script_name.in_(securities))
                                 .filter(EquitiesPriceData.trade_date >= start_date)
                                 .filter(EquitiesPriceData.trade_date <= end_date)
                                 .statement, dbConnection.session.bind) \
            .fillna(value=np.nan) \
            .set_index("equities_hash")
        return price_data.pivot_table(index='trade_date', columns='script_name', values='close_price')


def get_latest_price_and_signals_securities(securities, trade_date):
    if back_test_mode:
        return data_scratch.get_latest_available_price_securities(securities, trade_date)
    else:
        latest_date = get_latest_date(trade_date)
        price = pd.read_sql(dbConnection.session.query(EquitiesPriceData)
                            .filter(EquitiesPriceData.script_name.in_(securities))
                            .filter(EquitiesPriceData.trade_date == latest_date)
                            .statement, dbConnection.session.bind) \
            .fillna(value=np.nan) \
            .set_index("equities_hash")
        trading_constraints = get_trading_constraints_security_list_between_dates(securities, latest_date, trade_date)
        primary_signals = get_primary_signals_security_list_between_dates(securities, latest_date, trade_date)
        trend_signals = get_trend_signals_security_list_between_dates(securities, latest_date, trade_date)
        style_factors = get_style_factors_nse_500_security_list_between_dates(securities, latest_date, trade_date)
        price = join_dfs_overlapping_columns(price, trading_constraints)
        price = join_dfs_overlapping_columns(price, primary_signals)
        price = join_dfs_overlapping_columns(price, trend_signals)
        price = join_dfs_overlapping_columns(price, style_factors)
        return price


def get_latest_price_and_signals(trade_date):
    if back_test_mode:
        return data_scratch.get_latest_available_price(trade_date)
    else:
        latest_date = get_latest_date(trade_date)
        price = pd.read_sql(dbConnection.session.query(EquitiesPriceData)
                            .filter(EquitiesPriceData.trade_date == latest_date)
                            .statement, dbConnection.session.bind) \
            .fillna(value=np.nan) \
            .set_index("equities_hash")
        trading_constraints = get_trading_constraints_between_dates(latest_date, trade_date)
        primary_signals = get_primary_signals_between_dates(latest_date, trade_date)
        primary_signals_neutralized = get_primary_signals_neutralised_between_dates(latest_date, trade_date)
        trend_signals = get_trend_signals_between_dates(latest_date, trade_date)
        style_factors = get_style_factors_nse_500_between_dates(latest_date, trade_date)
        price = join_dfs_overlapping_columns(price, trading_constraints)
        price = join_dfs_overlapping_columns(price, primary_signals)
        price = join_dfs_overlapping_columns(price, primary_signals_neutralized)
        price = join_dfs_overlapping_columns(price, trend_signals)
        return join_dfs_overlapping_columns(price, style_factors)


def get_last_year_price_for_securities(stock_list, trade_date):
    if back_test_mode:
        return None
    else:
        return pd.read_sql(dbConnection.session.query(EquitiesPriceData)
                           .filter(EquitiesPriceData.script_name.in_(stock_list))
                           .filter(EquitiesPriceData.trade_date <= trade_date)
                           .filter(EquitiesPriceData.trade_date >= trade_date - relativedelta(years=1))
                           .statement, dbConnection.session.bind) \
            .fillna(value=np.nan) \
            .set_index("equities_hash")


def get_complete_price_for_securities(stock_list):
    if back_test_mode:
        return None
    else:
        return pd.read_sql(dbConnection.session.query(EquitiesPriceData)
                           .filter(EquitiesPriceData.script_name.in_(stock_list))
                           .statement, dbConnection.session.bind) \
            .fillna(value=np.nan) \
            .set_index("equities_hash")


def get_equities_price_for_script(script_name):
    return pd.read_sql(dbConnection.session.query(EquitiesPriceData)
                       .filter(EquitiesPriceData.script_name == script_name)
                       .statement, dbConnection.session.bind) \
        .set_index("equities_hash")


def is_trading_day(date):
    if back_test_mode:
        return data_scratch.is_trading_day(date)
    else:
        return dbConnection.session.query(EquitiesPriceData.trade_date) \
            .filter(EquitiesPriceData.trade_date == date).distinct().scalar() is not None


def next_trading_day(date):
    if back_test_mode:
        return data_scratch.next_trading_day(date)


def previous_trading_day(date):
    if back_test_mode:
        return data_scratch.previous_trading_day(date)
    else:
        return pd.read_sql(dbConnection.session.query(EquitiesPriceData.trade_date)
                           .filter(EquitiesPriceData.trade_date < date)
                           .distinct()
                           .statement, dbConnection.session.bind).max()['trade_date']


def get_prices_for_date_security_list(security_list, date):
    if back_test_mode:
        return data_scratch.get_prices_for_date_security_list(security_list, date)


def get_close_prices_for_date_security_list(security_list, date):
    if back_test_mode:
        return data_scratch.get_close_prices_for_date_security_list(security_list, date)


def get_trading_dates(start_date, end_date):
    if back_test_mode:
        return data_scratch.get_trading_dates(start_date, end_date)
    else:
        return pd.read_sql(dbConnection.session.query(EquitiesPriceData.trade_date)
                           .filter((EquitiesPriceData.trade_date >= start_date) &
                                   (EquitiesPriceData.trade_date <= end_date))
                           .distinct()
                           .statement, dbConnection.session.bind)['trade_date']


def securities_with_data(security_list, date):
    if back_test_mode:
        return data_scratch.securities_with_valid_data(security_list, date)


def get_last_available_price_security(security, date):
    if back_test_mode:
        return data_scratch.last_available_traded_price(security, date)


def get_last_available_price_for_securities(date, securities):
    latest_trade_date = dbConnection.session.query(func.max(EquitiesPriceData.trade_date)) \
        .filter(EquitiesPriceData.trade_date <= date).scalar()
    return get_equities_price_for_securities(latest_trade_date, securities).set_index('script_name')


def get_benchmark_returns_for_dates(benchmark, start_date, end_date):
    return pd.read_sql(dbConnection.session.query(EquitiesPriceData)
                       .filter((EquitiesPriceData.script_name == benchmark) &
                               (EquitiesPriceData.trade_date >= start_date) &
                               (EquitiesPriceData.trade_date <= end_date))
                       .statement, dbConnection.session.bind) \
        .set_index("trade_date").sort_index()['close_price'].pct_change().dropna()


def get_benchmark_returns(benchmark):
    return pd.read_sql(dbConnection.session.query(EquitiesPriceData)
                       .filter(EquitiesPriceData.script_name == benchmark)
                       .statement, dbConnection.session.bind) \
        .set_index("trade_date").sort_index()['close_price'].pct_change().dropna()


def get_monthly_benchmark_returns_for_dates(benchmark, start_date, end_date):
    daily_returns = get_benchmark_returns_for_dates(benchmark, start_date, end_date)
    daily_returns.index = pd.to_datetime(daily_returns.index)
    monthly_returns = daily_returns.resample('1M').apply(lambda x: np.prod(1 + x) - 1)
    monthly_returns.index = monthly_returns.index.shift(1, freq='D').date
    print(monthly_returns)
    return monthly_returns


def get_price_security_list_between_dates(security_list, start_date, end_date):
    return pd.read_sql(dbConnection.session.query(EquitiesPriceData)
                       .filter((EquitiesPriceData.script_name.in_(security_list)) &
                               (EquitiesPriceData.trade_date >= start_date) &
                               (EquitiesPriceData.trade_date <= end_date))
                       .statement, dbConnection.session.bind) \
        .set_index("equities_hash")


def get_price_with_signals_security_list_between_dates(security_list, start_date, end_date):
    if back_test_mode:
        return data_scratch.get_price_security_list_between_dates(security_list, start_date, end_date)
    else:
        price = get_price_security_list_between_dates(security_list, start_date, end_date)
        trading_constraints = get_trading_constraints_security_list_between_dates(security_list, start_date, end_date)
        primary_signals = get_primary_signals_security_list_between_dates(security_list, start_date, end_date)
        trend_signals = get_trend_signals_security_list_between_dates(security_list, start_date, end_date)
        style_factors = get_style_factors_nse_500_security_list_between_dates(security_list, start_date, end_date)
        print(style_factors)
        price = join_dfs_overlapping_columns(price, trading_constraints)
        price = join_dfs_overlapping_columns(price, primary_signals)
        price = join_dfs_overlapping_columns(price, trend_signals)
        price = join_dfs_overlapping_columns(price, style_factors)
        return price


def get_price_for_securities_before_date(stock_list, trade_date):
    return pd.read_sql(dbConnection.session.query(EquitiesPriceData)
                       .filter((EquitiesPriceData.script_name.in_(stock_list)) &
                               (EquitiesPriceData.trade_date < trade_date))
                       .statement, dbConnection.session.bind) \
        .fillna(value=np.nan) \
        .set_index("equities_hash")


def get_price_for_securities_after_date(security_list, start_date):
    return pd.read_sql(dbConnection.session.query(EquitiesPriceData)
                       .filter((EquitiesPriceData.trade_date >= start_date) &
                               (EquitiesPriceData.script_name.in_(security_list)))
                       .statement, dbConnection.session.bind)


def get_price_for_all_securities_after_date(start_date):
    return pd.read_sql(dbConnection.session.query(EquitiesPriceData)
                       .filter(EquitiesPriceData.trade_date >= start_date)
                       .statement, dbConnection.session.bind)


def get_price_for_all_securities(trade_date):
    return pd.read_sql(dbConnection.session.query(EquitiesPriceData)
                       .filter(EquitiesPriceData.trade_date == trade_date)
                       .statement, dbConnection.session.bind).set_index('script_name')


def get_close_price_before_x_days(securities, trade_date, x):
    date_before_x_days = get_date_before_x_days_inclusive(trade_date, x)
    return pd.read_sql(dbConnection.session.query(EquitiesPriceData)
                       .filter((EquitiesPriceData.trade_date == date_before_x_days) &
                               (EquitiesPriceData.script_name.in_(securities)))
                       .statement, dbConnection.session.bind).set_index('script_name')['close_price']


def get_date_after_x_days_inclusive(trade_date, x):
    return pd.read_sql(dbConnection.session.query(EquitiesPriceData.trade_date)
                       .filter(EquitiesPriceData.trade_date >= trade_date)
                       .distinct()
                       .order_by(EquitiesPriceData.trade_date.asc())
                       .limit(x)
                       .statement, dbConnection.session.bind).max()['trade_date']


def get_close_price_after_x_days(securities, trade_date, x):
    date_after_x_days = get_date_after_x_days_inclusive(trade_date, x)
    price_df = pd.read_sql(dbConnection.session.query(EquitiesPriceData)
                           .filter((EquitiesPriceData.trade_date >= trade_date) &
                                   (EquitiesPriceData.trade_date <= date_after_x_days) &
                                   (EquitiesPriceData.script_name.in_(securities)))
                           .statement, dbConnection.session.bind)
    return price_df.pivot_table(index='trade_date', columns='script_name', values='close_price').sort_index()
