import pandas as pd

from database.connection.DbConnection import get_pulse_db_connection
from database.domain.PrimarySignals import PrimarySignals

dbConnection = get_pulse_db_connection()


def get_primary_signals_security_list_between_dates(security_list, start_date, end_date):
    return pd.read_sql(dbConnection.session.query(PrimarySignals)
                       .filter((PrimarySignals.script_name.in_(security_list)) &
                               (PrimarySignals.trade_date >= start_date) &
                               (PrimarySignals.trade_date <= end_date))
                       .statement, dbConnection.session.bind) \
        .set_index("equities_hash")


def get_primary_signals_between_dates(start_date, end_date):
    return pd.read_sql(dbConnection.session.query(PrimarySignals)
                       .filter((PrimarySignals.trade_date >= start_date) &
                               (PrimarySignals.trade_date <= end_date))
                       .statement, dbConnection.session.bind) \
        .set_index("equities_hash")