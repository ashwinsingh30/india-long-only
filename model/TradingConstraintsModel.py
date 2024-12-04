import pandas as pd
import empyrical as em


def rolling_apply(df, window, function):
    return pd.Series([df.iloc[i - window: i].pipe(function)
                      if i >= window else None
                      for i in range(1, len(df) + 1)],
                     index=df.index)


def beta(data):
    return em.beta(data['diff'], data['benchmark_return'])


def trading_constraints(data, full_refresh=False):
    data = data.sort_values('trade_date')
    if not full_refresh:
        data = data.tail(270)
    data['beta'] = rolling_apply(data[['diff', 'benchmark_return']], 250, beta)
    data['turnover'] = data['volume'] * data['close_price']
    data['adt'] = data['turnover'].rolling(100).mean()
    data['brokerage_recommendation'] = 5 - data['iq_avg_broker_rec_no_ciq']
    data['liquidity_momentum'] = data['volume'].rolling(60).mean() / data['volume'].rolling(250).mean()
    return data
