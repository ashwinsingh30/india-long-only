import datetime

import pandas as pd

from database.persistence.SecuritiesMasterPersistence import persist_securities_master_info
from utils.TradingPlatformUtils import hash_securities_by_row


def parsedate(date):
    return datetime.datetime.strptime(date, '%Y-%m-%d').date()


equities_data = pd.read_csv('SecuritiesMaster_increment.csv')

equities_data.start_date = equities_data.start_date.apply(parsedate)
equities_data.end_date = equities_data.end_date.apply(parsedate)
equities_data['securities_master_hash'] = equities_data.apply(hash_securities_by_row, axis=1)
equities_data = equities_data.set_index('securities_master_hash', drop=True)
print(equities_data)
persist_securities_master_info(equities_data)
