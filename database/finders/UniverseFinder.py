import pandas as pd

from database.connection.DbConnection import get_pulse_db_connection
from database.domain.Universe import Universe
from utils.DateUtils import get_current_date_ist, parse_date

dbConnection = get_pulse_db_connection()


def get_latest_update_date(start_date):
    return pd.read_sql(dbConnection.session.query(Universe.as_of_date)
                       .filter(Universe.as_of_date <= start_date)
                       .distinct()
                       .statement, dbConnection.session.bind).max()['as_of_date']


def get_universe_constituents_between_dates(universe_name, start_date, end_date):
    latest_update = get_latest_update_date(start_date)
    return pd.read_sql(dbConnection.session.query(Universe)
                       .filter((Universe.universe_name == universe_name) &
                               (Universe.as_of_date >= latest_update) &
                               (Universe.as_of_date <= end_date))
                       .statement, dbConnection.session.bind)


def get_universe_for_date(universe_name, trade_date):
    latest_update = get_latest_update_date(trade_date)
    return pd.read_sql(dbConnection.session.query(Universe)
                       .filter((Universe.universe_name == universe_name) &
                               (Universe.as_of_date == latest_update))
                       .statement, dbConnection.session.bind)


def get_latest_universe(universe_name):
    latest_update = get_latest_update_date(get_current_date_ist())
    return pd.read_sql(dbConnection.session.query(Universe)
                       .filter((Universe.universe_name == universe_name) &
                               (Universe.as_of_date == latest_update))
                       .statement, dbConnection.session.bind)


def get_sector_map():
    return pd.read_sql(dbConnection.session.query(Universe)
                       .statement, dbConnection.session.bind)[['script_name', 'sector']] \
        .drop_duplicates().set_index('script_name').dropna()['sector']


def get_sector_industry_map():
    return pd.read_sql(dbConnection.session.query(Universe)
                       .statement, dbConnection.session.bind)[
        ['script_name', 'as_of_date', 'sector', 'industry']].drop_duplicates().rename(columns={'as_of_date': 'trade_date'}).set_index(
        ['script_name', 'trade_date']).dropna()[['sector', 'industry']]


def get_latest_sector_industry_map_universe(universe_name):
    latest_update = get_latest_update_date(get_current_date_ist())
    return pd.read_sql(dbConnection.session.query(Universe).filter((Universe.universe_name == universe_name) &
                               (Universe.as_of_date == latest_update))
                       .statement, dbConnection.session.bind)[
        ['script_name', 'sector', 'industry']].drop_duplicates().set_index('script_name').dropna()[['sector', 'industry']]


def get_super_universe(universe_name):
    return pd.read_sql(dbConnection.session.query(Universe)
                       .filter(Universe.universe_name == universe_name)
                       .statement, dbConnection.session.bind)['script_name'].unique()