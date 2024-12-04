import numpy as np
import pandas as pd

from database.connection.DbConnection import get_pulse_db_connection
from database.domain.EquitiesPriceData import EquitiesPriceData
from database.domain.PrimaryData import PrimaryData
from database.finders.EquitiesPriceDataFinder import get_benchmark_returns_for_dates
from database.finders.PrimaryDataFinder import get_primary_data_for_script
from database.persistence.PrimarySignalsPersistence import persist_primary_signals
from model.PrimarySignalsModel import nan_zero_filled, populate_primary_signals
from utils.DateUtils import parse_date
from utils.TradingPlatformUtils import get_daily_sampled_nse_500_universe

identifiers = np.setdiff1d(PrimaryData.__table__.columns.keys(),
                           ['equities_hash', 'script_name', 'trade_date'])

dbConnection = get_pulse_db_connection()


def get_benchmark_value(benchmark, start_date, end_date):
    return pd.read_sql(dbConnection.session.query(EquitiesPriceData)
                       .filter((EquitiesPriceData.script_name == benchmark) &
                               (EquitiesPriceData.trade_date >= start_date) &
                               (EquitiesPriceData.trade_date <= end_date))
                       .statement, dbConnection.session.bind).set_index("trade_date").sort_index()['close_price']


def calculate_primary_signals(script_name, benchmark_returns):
    fundamental_data = get_primary_data_for_script(script_name).reset_index().set_index('trade_date')
    fundamental_data = fundamental_data.dropna(subset=identifiers, how='all')
    fundamental_data[nan_zero_filled] = fundamental_data[nan_zero_filled].fillna(0)
    fundamental_data = fundamental_data.fillna(np.nan)
    if not fundamental_data.empty:
        equities_data = pd.read_sql(dbConnection.session.query(EquitiesPriceData)
                                    .filter(EquitiesPriceData.script_name == script_name)
                                    .statement, dbConnection.session.bind).set_index('trade_date')
        fundamental_data = fundamental_data.join(benchmark_returns)
        fundamental_data = fundamental_data.join(equities_data['close_price'], how='right').sort_index().ffill()
        fundamental_data = fundamental_data.dropna(subset=['script_name'])
        if not fundamental_data.empty:
            signals = populate_primary_signals(fundamental_data.reset_index())
            return signals.set_index('equities_hash')
        else:
            return pd.DataFrame()


start_date = parse_date('2010-01-01')
end_date = parse_date('2024-11-30')
universe = get_daily_sampled_nse_500_universe(start_date, end_date)
scripts = universe.script_name.unique()
scripts = np.array(sorted(set(scripts)))
print(scripts)
benchmark_returns = get_benchmark_value('NSE500', start_date, end_date).pct_change()
benchmark_returns.name = 'benchmark'

for script in scripts:
    print(script)
    script_signal = calculate_primary_signals(script, benchmark_returns)
    if script_signal is not None:
        persist_primary_signals(script_signal)
