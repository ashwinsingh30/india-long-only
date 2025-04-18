from sqlalchemy import *
from sqlalchemy.ext.declarative import declarative_base

from database.connection.DbConnection import pulse_db_schema

metadata = MetaData(schema=pulse_db_schema)
Base = declarative_base(metadata=metadata)


class TrendSignals(Base):
    __tablename__ = 'trend_signals'
    __schemaname__ = pulse_db_schema
    equities_hash = Column(String, primary_key=True, nullable=False)
    script_name = Column(String, nullable=False)
    trade_date = Column(Date, nullable=False)
    trend_signal_1 = Column(Float, nullable=True)
    trend_signal_2 = Column(Float, nullable=True)
    trend_signal_3 = Column(Float, nullable=True)
    trend_signal_4 = Column(Float, nullable=True)
    trend_signal_5 = Column(Float, nullable=True)
    trend_signal_6 = Column(Float, nullable=True)
    trend_signal_7 = Column(Float, nullable=True)
    trend_signal_8 = Column(Float, nullable=True)
    trend_signal_9 = Column(Float, nullable=True)
    trend_signal_10 = Column(Float, nullable=True)
    trend_signal_11 = Column(Float, nullable=True)
    trend_signal_12 = Column(Float, nullable=True)
    trend_signal_13 = Column(Float, nullable=True)
    trend_signal_14 = Column(Float, nullable=True)
    trend_signal_15 = Column(Float, nullable=True)
    trend_signal_16 = Column(Float, nullable=True)
    trend_signal_17 = Column(Float, nullable=True)
    trend_signal_18 = Column(Float, nullable=True)
    trend_signal_19 = Column(Float, nullable=True)
    trend_signal_20 = Column(Float, nullable=True)
    trend_signal_21 = Column(Float, nullable=True)
    trend_signal_22 = Column(Float, nullable=True)
    trend_signal_23 = Column(Float, nullable=True)
    trend_signal_24 = Column(Float, nullable=True)
    trend_signal_25 = Column(Float, nullable=True)
    trend_signal_26 = Column(Float, nullable=True)
    trend_signal_27 = Column(Float, nullable=True)
    trend_signal_28 = Column(Float, nullable=True)
    trend_signal_29 = Column(Float, nullable=True)
    long_term_momentum = Column(Float, nullable=True)
    short_term_momentum = Column(Float, nullable=True)
    long_term_volatility = Column(Float, nullable=True)
    short_term_volatility = Column(Float, nullable=True)
    mom_100 = Column(Float, nullable=True)
    mom_250 = Column(Float, nullable=True)
    mom_500 = Column(Float, nullable=True)
    vol_250 = Column(Float, nullable=True)
    vol_500 = Column(Float, nullable=True)
    hurst500 = Column(Float, nullable=True)
    hurst250 = Column(Float, nullable=True)