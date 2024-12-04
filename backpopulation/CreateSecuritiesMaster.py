import pandas as pd

super_universe = pd.read_csv('NSE500SuperUniverse.csv').set_index('isin')
nse_map = pd.read_csv('EQUITY_L.csv').set_index('isin')[['SYMBOL']]


def remove_suffix(data):
    return data.replace(' IN Equity','')


super_universe = super_universe.join(nse_map)
super_universe['script_name'] = super_universe['SYMBOL']
super_universe['delisted'] = super_universe['script_name'].isna()
super_universe['script_name_filler'] = super_universe['bbticker'].apply(remove_suffix)
super_universe['script_name'] = super_universe['script_name'].fillna(super_universe['script_name_filler'])
super_universe['start_date'] = '1900-01-01'
super_universe['end_date'] = '9999-12-31'
super_universe.loc[super_universe['delisted'], 'end_date'] = '2024-08-01'
super_universe = super_universe.reset_index()
super_universe = super_universe[['isin', 'bbticker', 'script_name', 'start_date', 'end_date']]
super_universe.to_csv('NSE500SecuritiesMaster.csv')