from sqlalchemy import *
from sqlalchemy.ext.declarative import declarative_base

from database.connection.DbConnection import pulse_db_schema

metadata = MetaData(schema=pulse_db_schema)
Base = declarative_base(metadata=metadata)


class StyleFactorsNSE500(Base):
    __tablename__ = 'style_factors_nse100'
    __schemaname__ = pulse_db_schema
    equities_hash = Column(String, primary_key=True, nullable=False)
    script_name = Column(String, nullable=False)
    trade_date = Column(Date, nullable=False)
    value = Column(Float, nullable=True)
    volatility = Column(Float, nullable=True)
    quality = Column(Float, nullable=True)
    leverage = Column(Float, nullable=True)
    profitability = Column(Float, nullable=True)
    analyst_rating = Column(Float, nullable=True)
    overcrowded_stocks = Column(Float, nullable=True)
    short_term_trend = Column(Float, nullable=True)
    long_term_trend = Column(Float, nullable=True)
    size = Column(Float, nullable=True)
