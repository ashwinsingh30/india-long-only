import pandas as pd

from database.connection.DbConnection import get_tcp_db_connection
from database.domain.tcp_db.EquityPriceHistoryAdjusted import EquityPriceHistoryAdjusted
from utils.DateUtils import parse_date

dbConnection = get_tcp_db_connection()


def get_data_for_id_between_dates(tcp_id, start_date, end_date):
    return pd.read_sql(dbConnection.session.query(EquityPriceHistoryAdjusted)
                       .filter((EquityPriceHistoryAdjusted.tcp_id == tcp_id) &
                               (EquityPriceHistoryAdjusted.equity_series_code == 'EQ') &
                               (EquityPriceHistoryAdjusted.trade_date >= start_date) &
                               (EquityPriceHistoryAdjusted.trade_date <= end_date))
                       .statement, dbConnection.session.bind)