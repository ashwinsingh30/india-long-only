import warnings

from signalgeneration.PulsePlatformModelBank import india_small_long_only_pulse

warnings.filterwarnings("ignore")

import pandas as pd

old_signal = pd.read_csv('Mar Signal.csv', index_col=[0])
signal = india_small_long_only_pulse(pd.to_datetime('2025-04-15').date(), old_signal=old_signal)
signal.to_csv('Apr Signal.csv')
