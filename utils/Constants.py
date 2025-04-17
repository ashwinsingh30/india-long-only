import numpy as np
import pandas as pd

from config.PulsePlatformConfig import get_pulse_platform_config
from database.finders.UniverseFinder import get_sector_map, get_latest_universe

nse_500_equities = get_latest_universe('NSE500')['script_name'].unique()
nse_500_equities = np.setdiff1d(nse_500_equities, [
    'ADANIENT', 'ATGL', 'ADANIPORTS', 'ADANIGREEN', 'ADANIPOWER', 'ADANIENSOL', 'AWL', "ZEEL"])
config = get_pulse_platform_config()
ciq_mnemonics_map = pd.read_csv(config.project_directory + '/utils/CIQMnemonics.csv', index_col=[0])
sector_map = get_sector_map()
index_under_analysis = ["NIFTY", "BANKNIFTY"]