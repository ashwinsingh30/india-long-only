import numpy as np
import pandas as pd

from database.finders.SecuritiesMasterFinder import get_all_active_securities
from database.finders.UniverseFinder import get_latest_universe

# universe = get_latest_universe('NSE500').set_index('script_name')
# map = get_all_active_securities().set_index('script_name')[['bbticker']]
# universe = universe.join(map)
fno_equities = pd.read_pickle('fno_equities.pkl').set_index('script_name')
borrow_universe = pd.read_csv('SLBM.csv').dropna().set_index('script_name')
# universe.loc[fno_equities.index, 'future_available'] = 'YES'
# universe['future_available'] = universe['future_available'].fillna('NO')
# universe.to_csv('NSE500Universe.csv')
portfolio = pd.read_csv(r'D:\Project\trading-platform-longonly\signalgeneration\Data.csv', index_col=[0])
# portfolio = portfolio.join(borrow_universe[['Security Name']], how='left')
common = np.intersect1d(portfolio.index, fno_equities.index)
portfolio.loc[common, 'future_available'] = 'YES'
portfolio['future_available'] = portfolio['future_available'].fillna('NO')
portfolio_borrows = np.intersect1d(portfolio.index, borrow_universe.index)
portfolio.loc[portfolio_borrows, 'borrow_available'] = 'YES'
portfolio['borrow_available'] = portfolio['borrow_available'].fillna('NO')
portfolio.to_csv('Long Short Signal - Oct 2023.csv')
print(portfolio)
