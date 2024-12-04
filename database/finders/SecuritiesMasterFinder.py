import pandas as pd
from sqlalchemy import extract

from database.connection.DbConnection import get_pulse_db_connection
from database.domain.SecuritiesMaster import SecuritiesMaster

dbConnection = get_pulse_db_connection()


def get_all_active_securities():
    return pd.read_sql(dbConnection.session.query(SecuritiesMaster)
                       .filter(extract('year', SecuritiesMaster.end_date) == 9999)
                       .statement, dbConnection.session.bind).set_index("isin")


def get_all_active_script_names():
    return pd.read_sql(dbConnection.session.query(SecuritiesMaster)
                       .filter(extract('year', SecuritiesMaster.end_date) == 2099)
                       .statement, dbConnection.session.bind)['script_name'].sort_values().unique()


def get_script_name_isin_map():
    return pd.read_sql(dbConnection.session.query(SecuritiesMaster)
                       .statement, dbConnection.session.bind)[['script_name', 'isin']]\
        .drop_duplicates().set_index('script_name')


def get_all_securities():
    return pd.read_sql(dbConnection.session.query(SecuritiesMaster)
                       .statement, dbConnection.session.bind).set_index("isin")
