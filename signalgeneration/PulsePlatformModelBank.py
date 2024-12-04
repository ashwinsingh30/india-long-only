import pandas as pd
import numpy as np

from database.finders.EquitiesPriceDataFinder import get_latest_price_and_signals
from signalgeneration.LongOnlyFundamentalPulse import get_fundamental_long_only_pulse, get_fundamental_long_short_pulse
from signalgeneration.LongOnlyPulse import get_long_only_pulse, get_long_minus_benchmark_pulse
from signalgeneration.SectorNeutralPulse import get_sector_neutral_factor_pulse
from utils.Constants import sector_map


def india_long_only_pulse(trade_date, old_signal=None, data=None):
    data = get_latest_price_and_signals(trade_date)
    if not old_signal is None:
        momentum_pulse = get_long_only_pulse(trade_date, old_signal=old_signal['momentum_signal'], data=data)['Weight']
        fundamental_pulse = get_fundamental_long_only_pulse(trade_date, old_signal=old_signal['fundamental_signal'],
                                                            data=data)['Weight']
    else:
        momentum_pulse = get_long_only_pulse(trade_date, data=data)['Weight']
        fundamental_pulse = get_fundamental_long_only_pulse(trade_date, data=data)['Weight']
    securities = np.union1d(momentum_pulse.index, fundamental_pulse.index)
    signal_matrix = pd.DataFrame(index=securities)
    signal_matrix['momentum_signal'] = momentum_pulse
    signal_matrix['fundamental_signal'] = fundamental_pulse
    signal_matrix = signal_matrix.fillna(0)
    signal_matrix['Weight'] = (signal_matrix['momentum_signal'] + signal_matrix['fundamental_signal']) / 2
    signal_matrix['Weight'] = signal_matrix['Weight'] / signal_matrix['Weight'].abs().sum()
    signal_matrix = signal_matrix[signal_matrix['Weight'] != 0]
    signal_matrix['Price'] = data.set_index('script_name')['close_price']
    signal_matrix['sector'] = sector_map
    return signal_matrix[['Weight', 'momentum_signal', 'fundamental_signal', 'Price', 'sector']]


def india_lmb_pulse(trade_date, old_signal=None, data=None):
    hedge_ratio = 0.5
    momentum_pulse = get_long_only_pulse(trade_date)
    fundamental_pulse = get_fundamental_long_only_pulse(trade_date)
    securities = np.union1d(momentum_pulse.index, fundamental_pulse.index)
    signal_matrix = pd.DataFrame(index=securities)
    signal_matrix['momentum_signal'] = momentum_pulse
    signal_matrix['fundamental_signal'] = fundamental_pulse
    signal_matrix = signal_matrix.fillna(0)
    signal_matrix['Weight'] = (signal_matrix['momentum_signal'] + signal_matrix['fundamental_signal']) / 2
    signal_matrix['Weight'] = signal_matrix['Weight'] / signal_matrix['Weight'].abs().sum()
    signal_matrix['Weight'] = (1 - hedge_ratio) * signal_matrix['Weight']
    signal_matrix.loc['NIFTY', 'Weight'] = -1 * hedge_ratio
    signal_matrix['Weight'] = signal_matrix['Weight'] / signal_matrix['Weight'].abs().sum()
    return signal_matrix[['Weight']]


def india_long_short_pulse(trade_date, old_signal=None, data=None):
    fundamental_market_neutral = get_fundamental_long_short_pulse(trade_date, old_signal=old_signal['ls_weight'])
    fundamental_sector_neutral = get_sector_neutral_factor_pulse(trade_date, old_signal=old_signal['sn_weight'])
    long_minus_benchmark = get_long_minus_benchmark_pulse(trade_date, old_signal=old_signal['lmb_weight'])
    securities = np.union1d(fundamental_market_neutral.index, fundamental_sector_neutral.index)
    securities = np.union1d(securities, long_minus_benchmark.index)
    signal_matrix = pd.DataFrame(index=securities)
    signal_matrix['ls_weight'] = fundamental_market_neutral['Weight']
    signal_matrix['sn_weight'] = fundamental_market_neutral['Weight']
    signal_matrix['lmb_weight'] = long_minus_benchmark['Weight']
    signal_matrix = signal_matrix.fillna(0)
    signal_matrix['Weight'] = (signal_matrix['ls_weight'] + signal_matrix['sn_weight'] +
                               signal_matrix['lmb_weight']) / 3
    signal_matrix['Weight'] = signal_matrix['Weight'] / signal_matrix['Weight'].abs().sum()
    data = get_latest_price_and_signals(trade_date).set_index('script_name')
    signal_matrix['Close Price'] = data['close_price']
    print(signal_matrix)
    return signal_matrix[['Weight', 'Close Price', 'ls_weight', 'sn_weight', 'lmb_weight']]


# import warnings
# warnings.filterwarnings("ignore")
# old_signal = pd.read_csv('LongShortSignal - Sep 2024.csv', index_col=[0])
# print(old_signal)
# india_long_short_pulse(pd.to_datetime('2024-09-26').date(), old_signal=pd.DataFrame(
#     columns=['ls_weight', 'sn_weight', 'lmb_weight'])).to_csv('LongShortSignal - Oct 2024.csv')
# long_old_signal = pd.read_csv('Long Only - Sep.csv', index_col=[0])
# india_long_only_pulse(pd.to_datetime('2024-11-29').date()).to_csv('Data.csv')