from data_load.capitaliq.PopulateFundamentalDataCapIQ import populate_primary_data_different_period_type
from database.finders.SecuritiesMasterFinder import get_all_securities
from utils.Constants import ciq_mnemonics_map
from utils.DateUtils import parse_date

isin_map = get_all_securities()
mnemonic_map = ciq_mnemonics_map[['period_type', 'group']].fillna("")
populate_primary_data_different_period_type(isin_map, mnemonic_map, parse_date('2025-01-06'),
                                            start_date=parse_date('2024-11-01'), persist_parts=True)
