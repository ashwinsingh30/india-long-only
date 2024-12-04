import warnings

import pandas as pd

from database.connection.DbConnection import get_pulse_db_connection
from database.domain.EquitiesPriceData import EquitiesPriceData
from database.domain.TrendSignals import TrendSignals
from database.persistence.TrendSignalsPersistence import persist_trend_signals
from model.TrendSignalsModel import calculate_trends_signals

warnings.filterwarnings("ignore")

dbConnection = get_pulse_db_connection()


def populate_equities_alpha_signals(script_name):
    equities_data = pd.read_sql(dbConnection.session.query(EquitiesPriceData)
                                .filter(EquitiesPriceData.script_name == script_name)
                                .statement, dbConnection.session.bind).set_index('equities_hash')
    signal = calculate_trends_signals(equities_data).reset_index()
    return signal[TrendSignals.__table__.columns.keys()].set_index('equities_hash')


scripts = pd.read_sql(dbConnection.session.query(EquitiesPriceData.script_name).distinct()
                      .statement, dbConnection.session.bind)['script_name'].values
scripts = sorted(set(scripts))

for script in scripts:
    print(script)
    script_signal = populate_equities_alpha_signals(script)
    persist_trend_signals(script_signal)
