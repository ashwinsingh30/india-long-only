from database.connection.DbConnection import get_pulse_db_connection, pulse_db_schema
from database.domain.AnnouncementDates import AnnouncementDates

dbConnection = get_pulse_db_connection()


def persist_announcement_dates(announcement_dates, trade_date):
    scripts = announcement_dates.script_name.unique()
    dbConnection.session.query(AnnouncementDates) \
        .filter(AnnouncementDates.trade_date == trade_date) \
        .delete(synchronize_session='fetch')
    dbConnection.session.commit()
    dbConnection.session.query(AnnouncementDates) \
        .filter((AnnouncementDates.next_announcement_date > trade_date) &
                (AnnouncementDates.script_name.in_(scripts))) \
        .delete(synchronize_session='fetch')
    dbConnection.session.commit()
    announcement_dates.to_sql('announcement_dates', con=dbConnection.engine, if_exists='append', schema=pulse_db_schema,
                              chunksize=1000, index=False)
    dbConnection.session.commit()
