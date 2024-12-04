import pandas as pd

from database.finders.EquitiesPriceDataFinder import get_price_with_signals_security_list_between_dates
from database.persistence.StyleFactorsNSE500Persistence import persist_style_factors_nse500
from model.StyleFactorsModel import populate_style_factors
from utils.DateUtils import parse_date
from utils.TradingPlatformUtils import get_daily_sampled_nse_500_universe

start_date = parse_date('2010-01-01')
end_date = parse_date('2015-01-01')

universe = get_daily_sampled_nse_500_universe(start_date, end_date)
style_factor_df = pd.read_csv('D:/Project/trading-platform-longonly/utils/StyleFactorsNSE500.csv', index_col=[0])
securities = universe.script_name.unique()
signals = get_price_with_signals_security_list_between_dates(securities, start_date, end_date)
style_factors = populate_style_factors(signals, universe, style_factor_df)
print(style_factors)
persist_style_factors_nse500(style_factors)
