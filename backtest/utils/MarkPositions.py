from database.finders.EquitiesPriceDataFinder import get_prices_for_date_security_list


def mark_long_equity_positions_to_market(portfolio_df, trade_date):
    portfolio_value = 0
    if not portfolio_df.empty:
        prices = get_prices_for_date_security_list(portfolio_df.index, trade_date)
        for index in portfolio_df.index:
            if index in prices.index:
                close_price = prices.loc[index]['close_price']
            portfolio_value += close_price * portfolio_df.loc[index]['position_size']
        return portfolio_value
    else:
        return 0


def mark_short_equity_positions_to_market(portfolio_df):
    pass