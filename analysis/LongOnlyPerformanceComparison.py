from os import listdir

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from empyrical import cum_returns
from hurst import compute_Hc
from matplotlib import pyplot as plt

from backtest.utils.BackTestUtils import get_portfolio_stats_slippage_adjusted, get_portfolio_stats
from database.connection.DbConnection import get_pulse_db_connection
from database.domain.EquitiesPriceData import EquitiesPriceData
from utils.DateUtils import parse_date

dir = 'D:/Project/trading-platform-longonly/analysis/performance_comparison/'
return_files = listdir(dir)

dbConnection = get_pulse_db_connection()


def returns_by_year(returns):
    returns.index = pd.to_datetime(returns.index)
    yearly_returns = returns.resample('YE').apply(lambda x: np.prod(1 + x) - 1)
    return yearly_returns


def get_underwater(returns):
    df_cum_rets = cum_returns(returns, starting_value=1.0)
    running_max = np.maximum.accumulate(df_cum_rets)
    underwater = - ((running_max - df_cum_rets) / running_max)
    return underwater


exposure_return = pd.DataFrame()
active_return = pd.DataFrame()
benchmark_returns = pd.Series()
leverage = pd.DataFrame()
turnover = pd.Series(dtype='float64')
turnover_df = pd.DataFrame()

for return_file in return_files:
    print(return_file)
    if '.csv' in return_file:
        model_name = return_file.replace('ReturnsDF_', '').replace('.csv', '')
        returns_df = pd.read_csv(dir + return_file).fillna(0)
        returns_df['Date'] = returns_df.trade_date.apply(parse_date)
        returns_df.set_index('Date', inplace=True)
        turnover[model_name] = returns_df['Turnover'].mean()
        model_return = returns_df['ExposureReturn']
        model_active = returns_df['ExposureReturn'] - returns_df['BenchmarkReturn']
        model_active.name = model_name
        benchmark_returns = returns_df['BenchmarkReturn']
        model_return.name = model_name
        exposure_return = exposure_return.join(model_return, how='outer')
        active_return = active_return.join(model_active, how='outer')

start_date = parse_date('2018-01-01')
end_date = parse_date('2024-11-30')

models = ['india_long_only', 'small_long_only']
# models = ['stat_arb_opt_baseline', 'stat_arb_opt_new']

exposure_return = exposure_return[models].dropna()
turnover = turnover[models]
active_return = active_return[models]
print(exposure_return)
print(turnover)

exposure_return = exposure_return[exposure_return.index >= start_date]
exposure_return = exposure_return[exposure_return.index <= end_date]
active_return = active_return.reindex(exposure_return.index)
benchmark_returns = benchmark_returns.reindex(exposure_return.index)

exposure_return = exposure_return.sort_index()
exposure_return['Benchmark'] = benchmark_returns

statistics = pd.DataFrame()
yearly_return = pd.DataFrame()
underwater = pd.DataFrame()
for model in exposure_return.columns:
    model_stats = get_portfolio_stats(exposure_return[model], benchmark_returns)
    underwater[model] = get_underwater(exposure_return[model])
    statistics[model] = model_stats
    yearly_return[model] = returns_by_year(exposure_return[model])

statistics['Benchmark'] = get_portfolio_stats(benchmark_returns, benchmark_returns)
statistics = statistics.T
statistics.to_csv('ComparisonStatistics.csv')
yearly_return['Benchmark'] = returns_by_year(benchmark_returns)
yearly_return.to_csv('YearlyReturn.csv')

# turnover_df['prime_raw'].rolling(20).mean().plot()
figure, axis = plt.subplots(3, 1)
exposure_return = exposure_return * 100
active_return_return = active_return * 100
underwater = underwater * 100
exposure_return.cumsum().plot(ax=axis[0], grid=True)
active_return.cumsum().plot(ax=axis[1], grid=True)
underwater.plot(ax=axis[2], grid=True)
axis[0].set_ylabel('Cumulative Returns (Capital)')
axis[0].set_xlabel('Date')
axis[1].set_ylabel('Cumulative Active Return Over Benchmark')
axis[1].set_xlabel('Date')
axis[2].set_ylabel('Draw-down')
axis[2].set_xlabel('Date')
plt.show()
