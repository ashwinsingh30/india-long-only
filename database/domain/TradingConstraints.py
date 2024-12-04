from sqlalchemy import *
from sqlalchemy.ext.declarative import declarative_base

from database.connection.DbConnection import pulse_db_schema

metadata = MetaData(schema=pulse_db_schema)
Base = declarative_base(metadata=metadata)


class TradingConstraints(Base):
    __tablename__ = 'trading_constraints'
    __schemaname__ = pulse_db_schema
    equities_hash = Column(String, primary_key=True, nullable=False)
    script_name = Column(String, nullable=False)
    trade_date = Column(Date, nullable=False)
    beta = Column(Float, nullable=True)
    adt = Column(Float, nullable=True)
    brokerage_recommendation = Column(Float, nullable=True)
    liquidity_momentum = Column(Float, nullable=True)
