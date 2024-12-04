import pandas as pd

from database.connection.DbConnection import get_pulse_db_connection
from database.domain.StyleFactorsNSE500 import StyleFactorsNSE500

dbConnection = get_pulse_db_connection()


def get_style_factors_nse_500_security_list_between_dates(security_list, start_date, end_date):
    return pd.read_sql(dbConnection.session.query(StyleFactorsNSE500)
                       .filter((StyleFactorsNSE500.script_name.in_(security_list)) &
                               (StyleFactorsNSE500.trade_date >= start_date) &
                               (StyleFactorsNSE500.trade_date <= end_date))
                       .statement, dbConnection.session.bind) \
        .set_index("equities_hash")


def get_style_factors_nse_500_between_dates(start_date, end_date):
    return pd.read_sql(dbConnection.session.query(StyleFactorsNSE500)
                       .filter((StyleFactorsNSE500.trade_date >= start_date) &
                               (StyleFactorsNSE500.trade_date <= end_date))
                       .statement, dbConnection.session.bind) \
        .set_index("equities_hash")