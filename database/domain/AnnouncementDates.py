from sqlalchemy import *
from sqlalchemy.ext.declarative import declarative_base

from database.connection.DbConnection import pulse_db_schema

metadata = MetaData(schema=pulse_db_schema)
Base = declarative_base(metadata=metadata)


class AnnouncementDates(Base):
    __tablename__ = 'announcement_dates'
    __schemaname__ = pulse_db_schema
    script_name = Column(String, nullable=False, primary_key=True)
    next_announcement_date = Column(Date, nullable=False, primary_key=True)
    trade_date = Column(Date, nullable=False, primary_key=True)
