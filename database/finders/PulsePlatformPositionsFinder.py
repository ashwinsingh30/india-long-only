import pandas as pd

from database.connection.DbConnection import get_pulse_db_connection
from database.domain.PulsePlatformPositions import PulsePlatformPositions

dbConnection = get_pulse_db_connection()


def get_last_update_date(trade_date):
    return pd.read_sql(dbConnection.session.query(PulsePlatformPositions.trade_date)
                       .filter(PulsePlatformPositions.trade_date < trade_date)
                       .distinct()
                       .statement, dbConnection.session.bind).max().dropna()


def get_last_update_date_model(trade_date, model):
    return pd.read_sql(dbConnection.session.query(PulsePlatformPositions.trade_date)
                       .filter((PulsePlatformPositions.trade_date < trade_date) &
                               (PulsePlatformPositions.model == model))
                       .distinct()
                       .statement, dbConnection.session.bind).max().dropna()


def get_current_update_date_model(trade_date, model):
    return pd.read_sql(dbConnection.session.query(PulsePlatformPositions.trade_date)
                       .filter((PulsePlatformPositions.trade_date <= trade_date) &
                               (PulsePlatformPositions.model == model))
                       .distinct()
                       .statement, dbConnection.session.bind).max().dropna()


def get_latest_positions_for_date(trade_date):
    latest_date = get_last_update_date(trade_date)
    if not latest_date.empty:
        latest_date = latest_date['trade_date']
        return pd.read_sql(dbConnection.session.query(PulsePlatformPositions)
                           .filter((PulsePlatformPositions.trade_date == latest_date))
                           .statement, dbConnection.session.bind)
    else:
        return pd.DataFrame()


def previous_trading_date(date):
    dates = pd.read_sql(dbConnection.session.query(PulsePlatformPositions.trade_date)
                        .filter(PulsePlatformPositions.trade_date < date)
                        .distinct()
                        .statement, dbConnection.session.bind)
    if not dates.empty:
        return dates.max()['trade_date']
    else:
        return None


def get_previous_trade_day_positions_model(trade_date, model):
    previous_trade_date = previous_trading_date(trade_date)
    if previous_trade_date is not None:
        return pd.read_sql(dbConnection.session.query(PulsePlatformPositions)
                           .filter((PulsePlatformPositions.trade_date == previous_trade_date) &
                                   (PulsePlatformPositions.model == model))
                           .statement, dbConnection.session.bind)
    else:
        return pd.DataFrame()


def get_current_positions_for_date_and_model(trade_date, model):
    current_date = get_current_update_date_model(trade_date, model)
    if not current_date.empty:
        current_date = current_date['trade_date']
        return pd.read_sql(dbConnection.session.query(PulsePlatformPositions)
                           .filter((PulsePlatformPositions.trade_date == current_date) &
                                   (PulsePlatformPositions.model == model))
                           .statement, dbConnection.session.bind)
    else:
        return pd.DataFrame()


def get_positions_for_dates_and_model(trade_dates, model):
    return pd.read_sql(dbConnection.session.query(PulsePlatformPositions)
                       .filter((PulsePlatformPositions.trade_date.in_(trade_dates)) &
                               (PulsePlatformPositions.model == model))
                       .statement, dbConnection.session.bind)


def get_positions_for_trade_date_and_model(trade_date, model):
    return pd.read_sql(dbConnection.session.query(PulsePlatformPositions)
                       .filter((PulsePlatformPositions.trade_date == trade_date) &
                               (PulsePlatformPositions.model == model))
                       .statement, dbConnection.session.bind)


def get_all_local_positions():
    return pd.read_sql(dbConnection.session.query(PulsePlatformPositions)
                       .statement, dbConnection.session.bind)


def get_all_local_positions_for_model(model):
    return pd.read_sql(dbConnection.session.query(PulsePlatformPositions)
                       .filter(PulsePlatformPositions.model == model)
                       .statement, dbConnection.session.bind)
