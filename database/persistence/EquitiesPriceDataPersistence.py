from pandas import DataFrame as df

from database.connection.DbConnection import get_pulse_db_connection, pulse_db_schema
from database.domain.EquitiesPriceData import EquitiesPriceData

dbConnection = get_pulse_db_connection()


def process_equities(quote_list):
    equities_quotes = df()
    for quote in quote_list:
        equities_quotes = equities_quotes.append(quote, ignore_index=True)
    equities_quotes.set_index('equities_hash', inplace=True)
    persist_equities_data(equities_quotes)


def persist_equities_data(transformed_df):
    dbConnection.session.query(EquitiesPriceData) \
        .filter(EquitiesPriceData.equities_hash.in_(transformed_df.index)) \
        .delete(synchronize_session='fetch')
    dbConnection.session.commit()
    transformed_df.to_sql('equities_data', con=dbConnection.engine, if_exists='append', schema=pulse_db_schema,
                          chunksize=1000)
    dbConnection.session.commit()
