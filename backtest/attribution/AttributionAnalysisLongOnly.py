import warnings

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from config.PulsePlatformConfig import get_pulse_platform_config
from database.finders.EquitiesPriceDataFinder import get_price_with_signals_security_list_between_dates
from database.finders.StyleFactorsNSE500Finder import get_style_factors_nse_500_between_dates
from signalgeneration.AuxxerePulse import get_long_only_factor_pulse
from utils.Constants import sector_map
from utils.DateUtils import parse_date
from utils.TradingPlatformUtils import get_daily_sampled_nse_500_universe

config = get_pulse_platform_config()
warnings.filterwarnings("ignore")


def get_capitalization_bucket(mcap_rank):
    if mcap_rank <= 100:
        return 'Large Cap'
    elif 250 >= mcap_rank > 100:
        return 'Mid Cap'
    else:
        return 'Small Cap'


strategy_name = '_long_only'
signal_df = (pd.read_csv('D:/Project/trading-platform-longonly/backtest/signaltest/SignalDF' + strategy_name + '.csv').
             rename(columns={'index': 'script_name'}))
return_df = pd.read_csv('D:/Project/trading-platform-longonly/backtest/signaltest/ReturnsDF' + strategy_name + '.csv')
return_df['trade_date'] = return_df.trade_date.apply(parse_date)
return_df = return_df.set_index('trade_date')
returns = return_df['ExposureReturn']
signal_df['trade_date'] = signal_df.trade_date.apply(parse_date)
signal_df = signal_df.pivot_table(index='trade_date', values='Weight', columns='script_name').fillna(0)
start_date = return_df.index.min()
end_date = return_df.index.max()
trade_dates = return_df.index.sort_values()
signal_df = signal_df.reindex(np.union1d(signal_df.index, trade_dates)).sort_index().ffill()
turnover = signal_df.diff().abs()
turnover = turnover.sum(axis=1)
turnover.name = 'Turnover'
signal_df.join(turnover).to_csv('Signal.csv')
universe = get_daily_sampled_nse_500_universe(start_date - relativedelta(months=1), end_date)
price_data = get_price_with_signals_security_list_between_dates(universe.script_name.unique(), start_date, end_date)
market_cap_table = price_data.pivot_table(index='trade_date', values='market_cap', columns='script_name')
diff_table = price_data.pivot_table(index='trade_date', values='diff', columns='script_name')

style_factors = ['value', 'quality', 'profitability', 'leverage', 'volatility',
                 'size', 'long_term_trend', 'short_term_trend']

risk_df = pd.DataFrame()
attribution_df = pd.DataFrame()

for i in range(1, len(trade_dates) - 1):
    trade_date = trade_dates[i - 1]
    date_universe = universe[universe['trade_date'] == trade_date].copy().set_index('script_name')
    securities = date_universe.index.values
    market_cap = market_cap_table.loc[trade_date].reindex(securities).fillna(0)
    next_day_diff = diff_table.loc[trade_dates[i]].reindex(securities).fillna(0)
    signal = signal_df.loc[trade_date].reindex(securities).fillna(0)
    market_cap.name = 'market_cap'
    date_universe = date_universe.join(market_cap).dropna()
    date_universe['Portfolio Weight'] = signal
    date_universe = date_universe.fillna(0)
    date_universe['Weight'] = date_universe['market_cap'] / date_universe['market_cap'].sum()
    date_universe['mcap_rank'] = date_universe['market_cap'].rank(ascending=False)
    date_universe['Category'] = date_universe['mcap_rank'].apply(get_capitalization_bucket)
    date_universe['Sector'] = sector_map
    date_universe['Active Weight'] = date_universe['Portfolio Weight'] - date_universe['Weight']
    date_universe['diff'] = next_day_diff
    sector_exposure = date_universe.groupby('Sector')['Active Weight'].sum()
    category_exposure = date_universe.groupby('Category')['Portfolio Weight'].sum()
    category_exposure = category_exposure / category_exposure.sum()
    sector_returns = pd.Series()
    for sector in sector_map.unique():
        sector_universe = date_universe[date_universe['Sector'] == sector]
        sector_universe['Sector Weight'] = sector_universe['Weight'] / sector_universe['Weight'].sum()
        sector_returns[sector] = sector_universe['Sector Weight'].dot(sector_universe['diff'])
    data_for_date = price_data[price_data['trade_date'] == trade_date]
    factor_weights = get_long_only_factor_pulse(trade_date, securities, style_factors, data=data_for_date)
    factor_exposure = signal.dot(factor_weights)
    factor_exposure = factor_exposure / factor_exposure.abs().sum()
    factor_returns = factor_weights.T.dot(next_day_diff)
    model_returns = returns[trade_dates[i]]
    factor_attribution = factor_returns * factor_exposure
    sector_attribution = sector_returns * sector_exposure
    attribution_series = pd.Series(factor_attribution)
    attribution_series['trade_date'] = trade_date
    attribution_series['factor_attribution'] = factor_attribution.sum()
    attribution_series['sector_attribution'] = sector_attribution.sum()
    attribution_series['selection_effect'] = model_returns - factor_attribution.sum()
    attribution_series['total_return'] = model_returns
    risk_series = pd.concat([factor_exposure, sector_exposure, category_exposure], axis=0)
    risk_series['trade_date'] = trade_date
    risk_df = pd.concat([risk_df, risk_series.to_frame().T], axis=0)
    attribution_df = pd.concat([attribution_df, attribution_series.to_frame().T], axis=0)

attribution_df = attribution_df.set_index('trade_date')
risk_df = risk_df.set_index('trade_date')
attribution_df.to_csv('AttributionDF.csv')
risk_df.to_csv('RiskDF.csv')
