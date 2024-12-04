import pandas as pd

from database.connection.DbConnection import get_pulse_db_connection
from database.domain.TradingConstraints import TradingConstraints

dbConnection = get_pulse_db_connection()


def get_trading_constraints_security_list_between_dates(security_list, start_date, end_date):
    return pd.read_sql(dbConnection.session.query(TradingConstraints)
                       .filter((TradingConstraints.script_name.in_(security_list)) &
                               (TradingConstraints.trade_date >= start_date) &
                               (TradingConstraints.trade_date <= end_date))
                       .statement, dbConnection.session.bind) \
        .set_index("equities_hash")


def get_trading_constraints_between_dates(start_date, end_date):
    return pd.read_sql(dbConnection.session.query(TradingConstraints)
                       .filter((TradingConstraints.trade_date >= start_date) &
                               (TradingConstraints.trade_date <= end_date))
                       .statement, dbConnection.session.bind) \
        .set_index("equities_hash")
