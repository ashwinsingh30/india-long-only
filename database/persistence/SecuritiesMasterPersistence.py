from database.connection.DbConnection import get_pulse_db_connection, pulse_db_schema
from database.domain.SecuritiesMaster import SecuritiesMaster

dbConnection = get_pulse_db_connection()


def persist_securities_master_info(securities_df):
    dbConnection.session.query(SecuritiesMaster) \
        .filter(SecuritiesMaster.securities_master_hash.in_(securities_df.index)) \
        .delete(synchronize_session='fetch')
    dbConnection.session.commit()
    securities_df.\
        to_sql('securities_master', con=dbConnection.engine, if_exists='append', schema=pulse_db_schema, chunksize=1000)
    dbConnection.session.commit()
