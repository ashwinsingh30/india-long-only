import numpy as np
import pandas as pd

from database.connection.DbConnection import get_pulse_db_connection
from database.domain.EquitiesPriceData import EquitiesPriceData
from database.domain.PrimaryData import PrimaryData
from database.domain.TradingConstraints import TradingConstraints
from database.persistence.TradingConstraintsPersistence import persist_trading_constraints
from model.TradingConstraintsModel import trading_constraints
from utils.TradingPlatformUtils import hash_equities_by_row

dbConnection = get_pulse_db_connection()


def populate_trading_constraints(script_name, benchmark_data):
    equities_data = pd.read_sql(dbConnection.session.query(EquitiesPriceData)
                                .filter(EquitiesPriceData.script_name == script_name)
                                .statement, dbConnection.session.bind)
    primary_data = pd.read_sql(dbConnection.session.query(PrimaryData)
                               .filter(PrimaryData.script_name == script_name)
                               .statement, dbConnection.session.bind).set_index('equities_hash')
    equities_data.set_index('trade_date', inplace=True)
    equities_data = equities_data.join(benchmark_data)
    equities_data = equities_data.reset_index().set_index('equities_hash')
    equities_data = equities_data.join(primary_data[['iq_avg_broker_rec_no_ciq', 'iq_marketcap']])
    script_signal = trading_constraints(equities_data, full_refresh=True)
    return script_signal.reset_index()[TradingConstraints.__table__.columns.keys()].set_index('equities_hash')


scripts = pd.read_sql(dbConnection.session.query(EquitiesPriceData.script_name).distinct()
                      .statement, dbConnection.session.bind)['script_name'].values
scripts = sorted(set(scripts))

benchmark_data = pd.read_sql(dbConnection.session.query(EquitiesPriceData)
                             .filter(EquitiesPriceData.script_name == 'NSE500')
                             .statement, dbConnection.session.bind).set_index('trade_date').sort_index()[
    'close_price'].pct_change()
benchmark_data.name = 'benchmark_return'
import warnings

warnings.filterwarnings("ignore")

for script in scripts:
    print(script)
    script_signal = populate_trading_constraints(script, benchmark_data)
    persist_trading_constraints(script_signal)
