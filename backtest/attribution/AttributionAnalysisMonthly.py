import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages

from backtest.config.BacktestConfig import get_pulse_platform_backtest_config
from config.PulsePlatformConfig import get_pulse_platform_config
from database.finders.EquitiesPriceDataFinder import data_scratch, get_price_with_signals_security_list_between_dates
from database.finders.StyleFactorsNSE500Finder import get_style_factors_nse_500_between_dates
from model.EquitiesSignalProcessingModel import get_covariance_matrix_for_securities, norm_zscore
from signalgeneration.AuxxerePulse import get_factor_pulse
from utils.DateUtils import parse_date
from utils.TradingPlatformUtils import get_daily_sampled_fno_universe, get_daily_sampled_nse_100_universe, \
    get_daily_sampled_nse_500_universe

rcParams.update({'figure.autolayout': True})

config = get_pulse_platform_config()
backtest_config = get_pulse_platform_backtest_config()

if config.run_mode == 'backtest':
    data_scratch.refresh_equities_data_scratch(backtest_config.start_date, backtest_config.end_date, look_back_days=200)

universe = get_daily_sampled_nse_500_universe(backtest_config.start_date - relativedelta(months=1),
                                              backtest_config.end_date)

strategy_name = '_long_only'
signal_df = (pd.read_csv('D:/Project/trading-platform-longonly/backtest/signaltest/SignalDF' + strategy_name + '.csv').
             rename(columns={'index': 'script_name'}))
return_df = pd.read_csv('D:/Project/trading-platform-longonly/backtest/signaltest/ReturnsDF' + strategy_name + '.csv')

return_df['trade_date'] = return_df.trade_date.apply(parse_date)
return_df = return_df.set_index('trade_date')
returns = return_df['ExposureReturn']
signal_df['trade_date'] = signal_df.trade_date.apply(parse_date)
start_date = return_df.trade_date.min()
end_date = return_df.trade_date.max()
price_data = get_price_with_signals_security_list_between_dates(universe.script_name.unique(), start_date, end_date)
factor_data_complete = get_style_factors_nse_500_between_dates(start_date, end_date)

style_factors = ['value', 'quality', 'profitability', 'leverage', 'volatility', 'size',
                 'overcrowded_stocks', 'long_term_trend', 'short_term_trend']
diff = price_data.reset_index()[['equities_hash', 'trade_date', 'script_name', 'diff']]
factor_data_complete = factor_data_complete.reset_index()[
    np.append(['equities_hash', 'trade_date', 'script_name'], style_factors)]

trade_dates = return_df.trade_date.sort_values().unique()

risk_df = pd.DataFrame()
attribution_df = pd.DataFrame()
for i in range(1, len(trade_dates) - 1):
    trade_date = trade_dates[i - 1]
    securities = universe[universe['trade_date'] == trade_date].script_name.unique()
    universe[universe['trade_date'] == trade_date].to_csv('Data.csv')
    covariance_matrix = get_covariance_matrix_for_securities(trade_date, securities, 250)
    securities = np.intersect1d(securities, covariance_matrix.index)
    signal = signal_df[signal_df['trade_date'] == trade_date].set_index('script_name')['Weight'] \
        .reindex(securities).fillna(0)
    factor_data = factor_data_complete[factor_data_complete['trade_date'] == trade_date].set_index(
        'script_name')[style_factors].reindex(securities).fillna(0)
    next_day_diff = diff[diff['trade_date'] == trade_dates[i]].set_index(
        'script_name')['diff'].reindex(securities).fillna(0)
    factor_data = factor_data.apply(norm_zscore, axis=0)

    covariance_matrix = covariance_matrix.loc[securities][securities]
    factor_weights = get_factor_pulse(trade_date, securities, style_factors)
    print(signal)
    print(factor_weights)
    factor_exposure = signal.dot(factor_data)
    factor_returns = factor_weights.T.dot(next_day_diff)
    model_returns = returns[trade_dates[i]]
    factor_attribution = factor_returns * factor_exposure
    attribution_series = pd.Series(factor_attribution)
    attribution_series['trade_date'] = trade_date
    attribution_series['factor_attribution'] = factor_attribution.sum()
    attribution_series['selection_effect'] = model_returns - factor_attribution.sum()
    attribution_series['total_return'] = model_returns
    factor_data['Weight'] = signal
    risk_series = pd.Series(factor_exposure)
    risk_series['trade_date'] = trade_date
    factor_covariance = factor_weights.T.dot(covariance_matrix).dot(factor_weights)
    risk_series['stock_dimension_vol'] = np.sqrt(signal.dot(covariance_matrix).dot(signal) * 252)
    risk_series['factor_dimension_vol'] = np.sqrt(factor_exposure.dot(factor_covariance).dot(factor_exposure) * 252)
    risk_df = pd.concat([risk_df, risk_series.to_frame().T], axis=0)
    attribution_df = pd.concat([attribution_df, attribution_series.to_frame().T], axis=0)

attribution_df = attribution_df.set_index('trade_date')
risk_df = risk_df.set_index('trade_date')
attribution_df.to_csv('AttributionDF.csv')
risk_df.to_csv('RiskDF.csv')

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams

rcParams.update({'figure.autolayout': True})
from matplotlib import pyplot as plt

with PdfPages(
        'D:/Project/trading-platform-longonly/backtest/attribution/RiskTearSheet' + strategy_name + '.pdf') as export_pdf:
    return_df['Turnover'].rolling(20).mean().plot()
    plt.title('Strategy Daily Turnover (in percentage of Gross Exposure)', fontsize=8)
    plt.ylabel('Turnover', fontsize=5)
    plt.grid(True)
    plt.autoscale()
    export_pdf.savefig()
    plt.close()

    risk_df[style_factors].rolling(20).mean().plot()
    plt.title('Net Exposure to Style Factors', fontsize=8)
    plt.ylabel('Net Exposure', fontsize=5)
    plt.grid(True)
    plt.autoscale()
    export_pdf.savefig()
    plt.close()

    risk_df[['short_term_trend', 'long_term_trend', 'overcrowded_stocks']].rolling(20).mean().plot()
    plt.title('Net Exposure to Momentum', fontsize=8)
    plt.ylabel('Net Exposure', fontsize=5)
    plt.grid(True)
    plt.autoscale()
    export_pdf.savefig()
    plt.close()

    risk_df[['stock_dimension_vol', 'factor_dimension_vol']].rolling(20).mean().plot()
    plt.title('Stock and Factor Dimension Risk', fontsize=8)
    plt.ylabel('Risk', fontsize=5)
    plt.grid(True)
    plt.autoscale()
    export_pdf.savefig()
    plt.close()

    attribution_df.cumsum().plot()
    plt.title('Return Trend (Total, Factor and Selection)', fontsize=8)
    plt.ylabel('Cumulative Returns.csv', fontsize=5)
    plt.grid(True)
    plt.autoscale()
    export_pdf.savefig()
    plt.close()
