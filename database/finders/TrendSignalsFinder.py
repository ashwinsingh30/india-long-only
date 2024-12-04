import pandas as pd

from database.connection.DbConnection import get_pulse_db_connection
from database.domain.TrendSignals import TrendSignals

dbConnection = get_pulse_db_connection()


def get_trend_signals_security_list_between_dates(security_list, start_date, end_date):
    return pd.read_sql(dbConnection.session.query(TrendSignals)
                       .filter((TrendSignals.script_name.in_(security_list)) &
                               (TrendSignals.trade_date >= start_date) &
                               (TrendSignals.trade_date <= end_date))
                       .statement, dbConnection.session.bind) \
        .set_index("equities_hash")


def get_trend_signals_between_dates(start_date, end_date):
    return pd.read_sql(dbConnection.session.query(TrendSignals)
                       .filter((TrendSignals.trade_date >= start_date) &
                               (TrendSignals.trade_date <= end_date))
                       .statement, dbConnection.session.bind) \
        .set_index("equities_hash")
