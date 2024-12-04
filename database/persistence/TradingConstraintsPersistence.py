from database.connection.DbConnection import get_pulse_db_connection, pulse_db_schema
from database.domain.TradingConstraints import TradingConstraints

dbConnection = get_pulse_db_connection()


def persist_trading_constraints(transformed_df):
    dbConnection.session.query(TradingConstraints) \
        .filter(TradingConstraints.equities_hash.in_(transformed_df.index)) \
        .delete(synchronize_session='fetch')
    dbConnection.session.commit()
    transformed_df.to_sql('trading_constraints', con=dbConnection.engine, if_exists='append', schema=pulse_db_schema,
                          chunksize=1000)
    dbConnection.session.commit()
