from database.connection.DbConnection import get_pulse_db_connection, pulse_db_schema
from database.domain.PulsePlatformPositions import PulsePlatformPositions

dbConnection = get_pulse_db_connection()

def persist_positions_for_date(positions, trade_date, models):
    dbConnection.session.query(PulsePlatformPositions) \
        .filter((PulsePlatformPositions.trade_date.in_(trade_date)) &
                (PulsePlatformPositions.model.in_(models))) \
        .delete(synchronize_session='fetch')
    dbConnection.session.commit()
    positions.to_sql('pulse_platform_positions', con=dbConnection.engine, if_exists='append', schema=pulse_db_schema,
                          chunksize=1000, index=False)
    dbConnection.session.commit()