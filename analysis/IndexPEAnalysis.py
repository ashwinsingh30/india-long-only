import numpy as np
from scipy.stats import stats

from database.finders.EquitiesPriceDataFinder import get_price_security_list_between_dates
from database.finders.PrimaryDataFinder import get_primary_data_between_dates
from utils.DateUtils import parse_date
from utils.TradingPlatformUtils import get_daily_sampled_nse_500_universe

start_date = parse_date('2015-01-01')
end_date = parse_date('2024-11-29')

universe = get_daily_sampled_nse_500_universe(start_date, end_date)
securities = universe['script_name'].unique()
price = get_price_security_list_between_dates(securities, start_date, end_date)
signals = get_primary_data_between_dates(start_date, end_date)
common = np.intersect1d(price.index, universe.index)
common = np.intersect1d(signals.index, common)
signals = signals.loc[common]
price = price.loc[common]
signals = signals.join(price[['close_price']], how='inner')
signals['pe_ratio'] = signals['close_price'] / signals['iq_basic_eps_excl']
signals['pe_ratio'] = signals['pe_ratio'].clip(lower=0, upper=500)
index_pe = signals.groupby('trade_date')[['pe_ratio']].mean()
index_pe['pe_ratio_percentile'] = index_pe['pe_ratio'].rolling(700).apply(
    lambda x: stats.percentileofscore(x, x[-1]))
index_pe = index_pe.shift(1)
index_pe.to_csv('TrailingPE.csv')
index_pe.plot(grid=True)
from matplotlib import pyplot as plt
plt.show()
