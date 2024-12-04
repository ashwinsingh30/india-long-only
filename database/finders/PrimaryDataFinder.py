import pandas as pd

from database.connection.DbConnection import get_pulse_db_connection
from database.domain.PrimaryData import PrimaryData

dbConnection = get_pulse_db_connection()


def get_primary_data_between_dates(start_date, end_date):
    return pd.read_sql(dbConnection.session.query(PrimaryData)
                       .filter((PrimaryData.trade_date >= start_date) &
                               (PrimaryData.trade_date <= end_date))
                       .statement, dbConnection.session.bind).set_index('equities_hash')


def get_primary_data_after_date(start_date):
    return pd.read_sql(dbConnection.session.query(PrimaryData)
                       .filter(PrimaryData.trade_date >= start_date)
                       .statement, dbConnection.session.bind)


def get_primary_data_for_securities_after_date(securities, start_date):
    return pd.read_sql(dbConnection.session.query(PrimaryData)
                       .filter((PrimaryData.script_name.in_(securities)) &
                               (PrimaryData.trade_date >= start_date))
                       .statement, dbConnection.session.bind)


def get_primary_data_for_script(script_name):
    return pd.read_sql(dbConnection.session.query(PrimaryData)
                       .filter(PrimaryData.script_name == script_name)
                       .statement, dbConnection.session.bind).set_index('equities_hash')
