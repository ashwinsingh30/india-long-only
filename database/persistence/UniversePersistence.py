from database.connection.DbConnection import get_pulse_db_connection, pulse_db_schema
from database.domain.Universe import Universe

dbConnection = get_pulse_db_connection()


def persist_universe(universe_df):
    dbConnection.session.query(Universe) \
        .filter(Universe.universe_hash.in_(universe_df.index)) \
        .delete(synchronize_session='fetch')
    dbConnection.session.commit()
    universe_df.\
        to_sql('universe', con=dbConnection.engine, if_exists='append', schema=pulse_db_schema, chunksize=1000)
    dbConnection.session.commit()
