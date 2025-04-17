import empyrical as em
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from backtest.config.BacktestConfig import get_pulse_platform_backtest_config

config = get_pulse_platform_backtest_config()


def update_long_call_portfolio(signal, current_date, previous_date, exposure):
    securities = list(signal.index)
    expiry = options_data_scratch.get_current_month_expiry(previous_date)
    days_to_expiry = (expiry - previous_date).days
    if days_to_expiry < 7:
        option_strikes = options_data_scratch.get_relevant_strike_price_call(previous_date, securities, 0.95)
    else:
        option_strikes = options_data_scratch.get_relevant_strike_price_call(previous_date, securities, 1)
    old_closing_prices = options_data_scratch.get_call_closing_prices(option_strikes, previous_date) \
        .rename(columns={'close_price': 'Old_Prices'})
    new_closing_prices = options_data_scratch.get_call_closing_prices(option_strikes, current_date) \
        .rename(columns={'close_price': 'New_Prices'}).drop('strike_price', axis=1)
    signal = signal.join(old_closing_prices, how='inner')
    signal = signal.join(new_closing_prices, how='inner')
    signal = signal.dropna()
    signal['option_type'] = 'CE'
    signal['Value'] = signal['Weight'].abs() * exposure
    signal['No_of_Shares'] = signal['Value'] / signal['Old_Prices']
    signal.index.name = 'script_name'
    signal = signal.reset_index()
    return signal.rename(columns={'New_Prices': 'Prices'})


def update_long_put_portfolio(signal, current_date, previous_date, exposure):
    securities = list(signal.index)
    expiry = options_data_scratch.get_current_month_expiry(previous_date)
    days_to_expiry = (expiry - previous_date).days
    if days_to_expiry < 7:
        option_strikes = options_data_scratch.get_relevant_strike_price_call(previous_date, securities, 1.05)
    else:
        option_strikes = options_data_scratch.get_relevant_strike_price_call(previous_date, securities, 1)

    old_closing_prices = options_data_scratch.get_put_closing_prices(option_strikes, previous_date) \
        .rename(columns={'close_price': 'Old_Prices'})
    new_closing_prices = options_data_scratch.get_put_closing_prices(option_strikes, current_date) \
        .rename(columns={'close_price': 'New_Prices'}).drop('strike_price', axis=1)
    signal = signal.join(old_closing_prices, how='inner')
    signal = signal.join(new_closing_prices, how='inner')
    signal = signal.dropna()
    signal['option_type'] = 'PE'
    signal['Value'] = signal['Weight'].abs() * exposure
    signal['No_of_Shares'] = signal['Value'] / signal['Old_Prices']
    signal.index.name = 'script_name'
    signal = signal.reset_index()
    return signal.rename(columns={'New_Prices': 'Prices'})


def update_short_call_portfolio(signal, current_date, previous_date, exposure):
    securities = list(signal.index)
    option_strikes = options_data_scratch.get_relevant_strike_price_call(previous_date, securities, 0.95)
    old_closing_prices = options_data_scratch.get_call_closing_prices(option_strikes, previous_date) \
        .rename(columns={'close_price': 'Old_Prices'})
    new_closing_prices = options_data_scratch.get_call_closing_prices(option_strikes, current_date) \
        .rename(columns={'close_price': 'New_Prices'}).drop('strike_price', axis=1)
    signal = signal.join(old_closing_prices, how='inner')
    signal = signal.join(new_closing_prices, how='inner')
    signal = signal.dropna()
    signal['option_type'] = 'CE'
    # signal['Weight'] = signal['Weight'] / signal['Weight'].sum()
    signal['Exposure'] = -1 * signal['Weight'].abs() * exposure
    signal['No_of_Shares'] = signal['Exposure'] / signal['strike_price']
    signal.index.name = 'script_name'
    signal = signal.reset_index()
    return signal.rename(columns={'New_Prices': 'Prices'})


def update_short_put_portfolio(signal, current_date, previous_date, exposure):
    securities = list(signal.index)
    option_strikes = options_data_scratch.get_relevant_strike_price_call(previous_date, securities, 1.05)
    old_closing_prices = options_data_scratch.get_put_closing_prices(option_strikes, previous_date) \
        .rename(columns={'close_price': 'Old_Prices'})
    new_closing_prices = options_data_scratch.get_put_closing_prices(option_strikes, current_date) \
        .rename(columns={'close_price': 'New_Prices'}).drop('strike_price', axis=1)
    signal = signal.join(old_closing_prices, how='inner')
    signal = signal.join(new_closing_prices, how='inner')
    signal = signal.dropna()
    signal['option_type'] = 'PE'
    # signal['Weight'] = signal['Weight'] / signal['Weight'].sum()
    signal['Exposure'] = -1 * signal['Weight'].abs() * exposure
    signal['No_of_Shares'] = signal['Exposure'] / signal['strike_price']
    signal.index.name = 'script_name'
    signal = signal.reset_index()
    return signal.rename(columns={'New_Prices': 'Prices'})


def update_call_bull_spread_portfolio(signal, current_date, previous_date, exposure):
    securities = list(signal.index)
    long_option_strikes = options_data_scratch.get_relevant_strike_price_call(previous_date, securities, 0.95)
    short_option_strikes = options_data_scratch.get_relevant_strike_price_call(previous_date, securities, 1.05)

    old_long_closing_prices = options_data_scratch.get_call_closing_prices(long_option_strikes, previous_date) \
        .rename(columns={'close_price': 'Old_Prices'})
    new_long_closing_prices = options_data_scratch.get_call_closing_prices(long_option_strikes, current_date) \
        .rename(columns={'close_price': 'New_Prices'}).drop('strike_price', axis=1)

    old_short_closing_prices = options_data_scratch.get_call_closing_prices(short_option_strikes, previous_date) \
        .rename(columns={'close_price': 'Old_Prices'})
    new_short_closing_prices = options_data_scratch.get_call_closing_prices(short_option_strikes, current_date) \
        .rename(columns={'close_price': 'New_Prices'}).drop('strike_price', axis=1)

    long_closing_prices = old_long_closing_prices.join(new_long_closing_prices[['New_Prices']], how='inner')
    short_closing_prices = old_short_closing_prices.join(new_short_closing_prices[['New_Prices']], how='inner')
    short_closing_prices['long_cost'] = long_closing_prices['Old_Prices']
    short_closing_prices['spread_cost'] = short_closing_prices['strike_price'] + \
                                          config.leverage * \
                                          (short_closing_prices['long_cost'] - short_closing_prices['Old_Prices'])

    short_signal = signal.join(short_closing_prices, how='inner')
    short_signal = short_signal.dropna()
    short_signal['Exposure'] = -1 * short_signal['Weight'].abs() * exposure
    short_signal['No_of_Shares'] = short_signal['Exposure'] / short_signal['spread_cost']
    short_signal.index.name = 'script_name'
    print(short_signal)

    long_signal = signal.join(long_closing_prices, how='inner')
    long_signal = long_signal.dropna()
    long_signal['Exposure'] = long_signal['Weight'].abs() * exposure
    long_signal['No_of_Shares'] = -1 * short_signal['No_of_Shares']
    long_signal.index.name = 'script_name'

    signal = pd.concat([short_signal.reset_index(), long_signal.reset_index()], axis=0)
    signal['option_type'] = 'CE'
    return signal.rename(columns={'New_Prices': 'Prices'})


def update_put_bear_spread_portfolio(signal, current_date, previous_date, exposure):
    securities = list(signal.index)
    long_option_strikes = options_data_scratch.get_relevant_strike_price_put(previous_date, securities, 1)
    short_option_strikes = options_data_scratch.get_relevant_strike_price_put(previous_date, securities, 1.05)

    old_long_closing_prices = options_data_scratch.get_put_closing_prices(long_option_strikes, previous_date) \
        .rename(columns={'close_price': 'Old_Prices'})
    new_long_closing_prices = options_data_scratch.get_put_closing_prices(long_option_strikes, current_date) \
        .rename(columns={'close_price': 'New_Prices'}).drop('strike_price', axis=1)

    old_short_closing_prices = options_data_scratch.get_put_closing_prices(short_option_strikes, previous_date) \
        .rename(columns={'close_price': 'Old_Prices'})
    new_short_closing_prices = options_data_scratch.get_put_closing_prices(short_option_strikes, current_date) \
        .rename(columns={'close_price': 'New_Prices'}).drop('strike_price', axis=1)

    long_closing_prices = old_long_closing_prices.join(new_long_closing_prices[['New_Prices']], how='inner')
    short_closing_prices = old_short_closing_prices.join(new_short_closing_prices[['New_Prices']], how='inner')
    short_closing_prices['long_cost'] = long_closing_prices['Old_Prices']
    short_closing_prices['spread_cost'] = short_closing_prices['strike_price'] + \
                                          config.leverage * \
                                          (short_closing_prices['long_cost'] - short_closing_prices['Old_Prices'])

    short_signal = signal.join(short_closing_prices, how='inner')
    short_signal = short_signal.dropna()
    short_signal['Exposure'] = -1 * short_signal['Weight'].abs() * exposure
    short_signal['No_of_Shares'] = short_signal['Exposure'] / short_signal['spread_cost']
    short_signal.index.name = 'script_name'
    print(short_signal)

    long_signal = signal.join(long_closing_prices, how='inner')
    long_signal = long_signal.dropna()
    long_signal['Exposure'] = long_signal['Weight'].abs() * exposure
    long_signal['No_of_Shares'] = -1 * short_signal['No_of_Shares']
    long_signal.index.name = 'script_name'

    signal = pd.concat([short_signal.reset_index(), long_signal.reset_index()], axis=0)
    signal['option_type'] = 'PE'
    return signal.rename(columns={'New_Prices': 'Prices'})


def update_short_call_long_put_portfolio(signal, current_date, previous_date, exposure):
    securities = list(signal.index)
    short_call_strikes = options_data_scratch.get_relevant_strike_price_call(previous_date, securities, 0.95)

    old_long_put_closing_prices = options_data_scratch.get_put_closing_prices(short_call_strikes, previous_date) \
        .rename(columns={'close_price': 'Old_Prices'})
    new_long_put_closing_prices = options_data_scratch.get_put_closing_prices(short_call_strikes, current_date) \
        .rename(columns={'close_price': 'New_Prices'}).drop('strike_price', axis=1)

    old_short_call_closing_prices = options_data_scratch.get_call_closing_prices(short_call_strikes, previous_date) \
        .rename(columns={'close_price': 'Old_Prices'})
    new_short_call_closing_prices = options_data_scratch.get_call_closing_prices(short_call_strikes, current_date) \
        .rename(columns={'close_price': 'New_Prices'}).drop('strike_price', axis=1)

    long_closing_prices = old_long_put_closing_prices.join(new_long_put_closing_prices[['New_Prices']], how='inner')
    short_closing_prices = old_short_call_closing_prices.join(new_short_call_closing_prices[['New_Prices']],
                                                              how='inner')

    short_signal = signal.join(short_closing_prices, how='inner')
    short_signal = short_signal.dropna()
    short_signal['Exposure'] = -1 * short_signal['Weight'].abs() * exposure
    short_signal['No_of_Shares'] = short_signal['Exposure'] / short_signal['strike_price']
    short_signal.index.name = 'script_name'
    short_signal['option_type'] = 'CE'

    long_signal = signal.join(long_closing_prices, how='inner')
    long_signal = long_signal.dropna()
    long_signal['Exposure'] = long_signal['Weight'].abs() * exposure
    long_signal['No_of_Shares'] = -1 * short_signal['No_of_Shares']
    long_signal.index.name = 'script_name'
    long_signal['option_type'] = 'PE'
    signal = pd.concat([short_signal.reset_index(), long_signal.reset_index()], axis=0)
    return signal.rename(columns={'New_Prices': 'Prices'})


def update_short_put_long_call_portfolio(signal, current_date, previous_date, exposure):
    securities = list(signal.index)
    short_put_strikes = options_data_scratch.get_relevant_strike_price_put(previous_date, securities, 1.05)

    old_long_call_closing_prices = options_data_scratch.get_call_closing_prices(short_put_strikes, previous_date) \
        .rename(columns={'close_price': 'Old_Prices'})
    new_long_call_closing_prices = options_data_scratch.get_call_closing_prices(short_put_strikes, current_date) \
        .rename(columns={'close_price': 'New_Prices'}).drop('strike_price', axis=1)

    old_short_put_closing_prices = options_data_scratch.get_put_closing_prices(short_put_strikes, previous_date) \
        .rename(columns={'close_price': 'Old_Prices'})
    new_short_put_closing_prices = options_data_scratch.get_put_closing_prices(short_put_strikes, current_date) \
        .rename(columns={'close_price': 'New_Prices'}).drop('strike_price', axis=1)

    long_closing_prices = old_long_call_closing_prices.join(new_long_call_closing_prices[['New_Prices']], how='inner')
    short_closing_prices = old_short_put_closing_prices.join(new_short_put_closing_prices[['New_Prices']], how='inner')

    short_signal = signal.join(short_closing_prices, how='inner')
    short_signal = short_signal.dropna()
    short_signal['Exposure'] = -1 * short_signal['Weight'].abs() * exposure
    short_signal['No_of_Shares'] = short_signal['Exposure'] / short_signal['strike_price']
    short_signal.index.name = 'script_name'
    short_signal['option_type'] = 'PE'
    long_signal = signal.join(long_closing_prices, how='inner')
    long_signal = long_signal.dropna()
    long_signal['Exposure'] = long_signal['Weight'].abs() * exposure
    long_signal['No_of_Shares'] = -1 * short_signal['No_of_Shares']
    long_signal.index.name = 'script_name'
    long_signal['option_type'] = 'CE'
    signal = pd.concat([short_signal.reset_index(), long_signal.reset_index()], axis=0)
    return signal.rename(columns={'New_Prices': 'Prices'})


def get_portfolio_stats(returns, benchmark_returns):
    returns = returns.reindex(benchmark_returns.index).astype('float64')
    active_returns = returns - benchmark_returns
    stats = pd.Series()
    stats['Alpha'] = em.alpha(returns, benchmark_returns, annualization=250)
    stats['Beta'] = em.beta(returns, benchmark_returns)
    stats['CAGR'] = em.cagr(returns, annualization=250)
    stats['Max Drawdown'] = em.max_drawdown(returns)
    stats['Annualised Returns'] = em.annual_return(returns, annualization=250)
    stats['Annualised Volatility'] = em.annual_volatility(returns, annualization=250)
    stats['Calmar Ratio'] = em.calmar_ratio(returns, annualization=250)
    stats['Sharpe Ratio'] = em.sharpe_ratio(returns, annualization=250, risk_free=0.07 / 250)
    stats['Sortino Ratio'] = em.sortino_ratio(returns, annualization=250)
    stats['Stability'] = em.stability_of_timeseries(returns)
    stats['Tail Ratio'] = em.tail_ratio(returns)
    stats['Sharpe - Un-discounted'] = ((returns.sum() / returns.count()) * 250) / (returns.std() * np.sqrt(250))
    stats['Win Ratio Over Benchmark'] = (len(active_returns[active_returns > 0]) / len(active_returns)) * 100
    stats['Tracking Error'] = em.annual_volatility(active_returns, annualization=250)
    stats['Total Return'] = returns.sum()
    return stats


def get_portfolio_stats_slippage_adjusted(returns, benchmark_returns, turnover):
    slippage = (returns.mean() * (turnover / 100) * 0.25)
    returns = returns - slippage
    returns = returns.reindex(benchmark_returns.index).astype('float64')
    stats = pd.Series()
    stats['Alpha'] = em.alpha(returns, benchmark_returns, annualization=250)
    stats['Beta'] = em.beta(returns, benchmark_returns)
    stats['CAGR'] = em.cagr(returns, annualization=250)
    stats['Max Drawdown'] = em.max_drawdown(returns)
    stats['Annualised Returns'] = em.annual_return(returns, annualization=250)
    stats['Annualised Volatility'] = em.annual_volatility(returns, annualization=250)
    stats['Calmar Ratio'] = em.calmar_ratio(returns, annualization=250)
    stats['Sharpe Ratio'] = em.sharpe_ratio(returns, annualization=250, risk_free=0.07 / 250)
    stats['Sortino Ratio'] = em.sortino_ratio(returns, annualization=250)
    stats['Stability'] = em.stability_of_timeseries(returns)
    stats['Tail Ratio'] = em.tail_ratio(returns)
    stats['Sharpe - Un-discounted'] = ((returns.sum() / returns.count()) * 250) / (returns.std() * np.sqrt(250))
    stats['Win Ratio'] = len(returns[returns > 0]) / len(returns)
    stats['Total Return'] = returns.sum()
    stats['Slippage'] = slippage
    stats['Transaction Cost'] = config.transaction_cost
    stats['Turnover'] = turnover
    return stats


def get_portfolio_stats_slippage_adjusted_absolute(returns, turnover):
    slippage = (returns.mean() * (turnover / 100) * 0.25)
    returns = returns - slippage
    stats = pd.Series()
    stats['CAGR'] = em.cagr(returns, annualization=250)
    stats['Max Drawdown'] = em.max_drawdown(returns)
    stats['Annualised Returns'] = em.annual_return(returns, annualization=250)
    stats['Annualised Volatility'] = em.annual_volatility(returns, annualization=250)
    stats['Calmar Ratio'] = em.calmar_ratio(returns, annualization=250)
    stats['Sharpe Ratio'] = em.sharpe_ratio(returns, annualization=250, risk_free=0.07 / 250)
    stats['Sortino Ratio'] = em.sortino_ratio(returns, annualization=250)
    stats['Stability'] = em.stability_of_timeseries(returns)
    stats['Tail Ratio'] = em.tail_ratio(returns)
    stats['Sharpe - Un-discounted'] = ((returns.sum() / returns.count()) * 250) / (returns.std() * np.sqrt(250))
    stats['Win Ratio'] = len(returns[returns > 0]) / len(returns)
    stats['Total Return'] = returns.sum()
    stats['Slippage'] = slippage
    stats['Transaction Cost'] = config.transaction_cost
    stats['Turnover'] = turnover
    return stats


def get_monthly_portfolio_stats(returns, benchmark_returns):
    returns = returns.reindex(benchmark_returns.index).astype('float64')
    stats = pd.Series()
    stats['Alpha'] = em.alpha(returns, benchmark_returns, annualization=12)
    stats['Beta'] = em.beta(returns, benchmark_returns)
    stats['CAGR'] = em.cagr(returns, annualization=12)
    stats['Max Drawdown'] = em.max_drawdown(returns)
    stats['Annualised Returns'] = em.annual_return(returns, annualization=12)
    stats['Annualised Volatility'] = em.annual_volatility(returns, annualization=12)
    stats['Calmar Ratio'] = em.calmar_ratio(returns, annualization=12)
    stats['Sharpe Ratio'] = em.sharpe_ratio(returns, annualization=12, risk_free=0.07 / 12)
    stats['Sortino Ratio'] = em.sortino_ratio(returns, annualization=12)
    stats['Stability'] = em.stability_of_timeseries(returns)
    stats['Tail Ratio'] = em.tail_ratio(returns)
    return stats


def get_capital_for_year(base_capital, date):
    return base_capital * (1 - ((2022 - date.year) * 0.08))


def plot_cumulative_returns(portfolio_returns, benchmark_returns, strategy_name=None, strategy_type='long_short'):
    plot_df = pd.DataFrame()
    if strategy_name is None:
        strategy_name = 'Portfolio'
    plot_df[strategy_name] = portfolio_returns
    plot_df['NIFTY 50'] = benchmark_returns
    plot_df = plot_df * 100
    plot_df['Active Returns Over Benchmark'] = plot_df[strategy_name] - plot_df['NIFTY 50']
    figure, axis = plt.subplots(2)
    axis[0].set_title("Returns")
    axis[1].set_title("Active Returns")
    axis[0].set_ylabel('Cumulative Performance (Percentage Points)')
    axis[1].set_ylabel('Cumulative Performance (Percentage Points)')
    plot_df.dropna(axis=0, inplace=True)
    plot_df[[strategy_name, 'NIFTY 50']].cumsum().plot(ax=axis[0], grid=True)
    plot_df[['Active Returns Over Benchmark']].cumsum().plot(ax=axis[1], grid=True)
    plt.show()
