import pandas as pd

from database.connection.DbConnection import get_tcp_db_connection
from database.domain.tcp_db.ISINMapping import ISINMapping

dbConnection = get_tcp_db_connection()


def get_latest_tcp_id_for_isin(isin):
    dates = pd.read_sql(dbConnection.session.query(ISINMapping.date)
                        .filter((ISINMapping.isin == isin) &
                                (ISINMapping.equity_series_code == 'EQ'))
                        .statement, dbConnection.session.bind)['date']
    if not dates.empty:
        latest_date = dates.sort_values().max()
        return pd.read_sql(dbConnection.session.query(ISINMapping.tcp_id)
                           .filter((ISINMapping.isin == isin) &
                                   (ISINMapping.equity_series_code == 'EQ') &
                                   (ISINMapping.date == latest_date))
                           .statement, dbConnection.session.bind)['tcp_id'].iloc[0]
    else:
        return None