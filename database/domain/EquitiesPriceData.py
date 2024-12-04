from sqlalchemy import *
from sqlalchemy.ext.declarative import declarative_base

from database.connection.DbConnection import pulse_db_schema

metadata = MetaData(schema=pulse_db_schema)
Base = declarative_base(metadata=metadata)


class EquitiesPriceData(Base):
    __tablename__ = 'equities_data'
    __schemaname__ = pulse_db_schema
    equities_hash = Column(String, primary_key=True, nullable=False)
    script_name = Column(String, nullable=False)
    trade_date = Column(Date, nullable=False)
    diff = Column(Float, nullable=True)
    previous_close = Column(Float, nullable=True)
    open_price = Column(Float, nullable=True)
    high_price = Column(Float, nullable=True)
    low_price = Column(Float, nullable=True)
    close_price = Column(Float, nullable=True)
    last_price = Column(Float, nullable=True)
    volume = Column(Float, nullable=True)
    vwap_price = Column(Float, nullable=True)
    delivered_to_traded = Column(Float, nullable=True)
