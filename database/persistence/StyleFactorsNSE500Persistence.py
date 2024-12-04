from database.connection.DbConnection import get_pulse_db_connection, pulse_db_schema
from database.domain.StyleFactorsNSE500 import StyleFactorsNSE500

dbConnection = get_pulse_db_connection()


def persist_style_factors_nse500(data_df):
    dbConnection.session.query(StyleFactorsNSE500) \
        .filter(StyleFactorsNSE500.equities_hash.in_(data_df.index)) \
        .delete(synchronize_session='fetch')
    dbConnection.session.commit()
    data_df.to_sql('style_factors_nse500', con=dbConnection.engine, if_exists='append', schema=pulse_db_schema,
                   chunksize=1000)
    dbConnection.session.commit()
