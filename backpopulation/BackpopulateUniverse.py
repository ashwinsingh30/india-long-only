import pandas as pd

from database.finders.SecuritiesMasterFinder import get_all_active_securities, get_all_securities
from database.persistence.UniversePersistence import persist_universe
from utils.TradingPlatformUtils import hash_universe_by_row

universe_data = pd.read_csv(r"D:\Project\trading-platform-longonly\backpopulation\NSE500.csv"). \
    rename(columns={'ID_ISIN': 'isin',
                    'trade_date': 'as_of_date',
                    'GICS_SECTOR_NAME': 'sector',
                    'GICS_INDUSTRY_NAME': 'industry',
                    'GICS_SUB_INDUSTRY_NAME': 'sub_industry',
                    'CUR_MKT_CAP': 'market_cap',
                    'Weight': 'weight'})

isin_map = get_all_securities().set_index('bbticker')['script_name']
universe_data = universe_data.join(isin_map, on='ticker')
sector_map = pd.read_csv(r'D:\Project\trading-platform-longonly\data\universe\sector_map.csv').set_index('bbticker')
universe_data = universe_data.set_index('ticker')
universe_data['sector'] = universe_data['sector'].fillna(sector_map['sector'])
universe_data = universe_data.reset_index()
universe_data = universe_data[
    ['script_name', 'as_of_date', 'sector', 'industry', 'sub_industry', 'market_cap', 'weight']]
universe_data['universe_name'] = 'NSE500'
universe_data['universe_hash'] = universe_data.apply(hash_universe_by_row, axis=1)
universe_data = universe_data.set_index('universe_hash')
persist_universe(universe_data)
