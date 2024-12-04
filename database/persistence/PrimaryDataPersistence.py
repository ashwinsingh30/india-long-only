from database.connection.DbConnection import get_pulse_db_connection, pulse_db_schema
from database.domain.PrimaryData import PrimaryData

dbConnection = get_pulse_db_connection()


def persist_primary_data(data_df):
    dbConnection.session.query(PrimaryData) \
        .filter(PrimaryData.equities_hash.in_(data_df.index)) \
        .delete(synchronize_session='fetch')
    dbConnection.session.commit()
    data_df.to_sql('primary_data', con=dbConnection.engine, if_exists='append', schema=pulse_db_schema,
                   chunksize=1000)
    dbConnection.session.commit()
