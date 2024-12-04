from sqlalchemy import *
from sqlalchemy.ext.declarative import declarative_base

metadata = MetaData(schema='marketdata')
Base = declarative_base(metadata=metadata)


class EquityPriceHistoryAdjusted(Base):
    __tablename__ = 'equity_price_history_adjusted'
    __schemaname__ = 'marketdata'
    trade_date = Column(Date, nullable=False, primary_key=True)
    tcp_id = Column(Integer, nullable=False, primary_key=True)
    equity_series_code = Column(String, nullable=False, primary_key=True)
    open = Column(Float, nullable=True)
    high = Column(Float, nullable=True)
    low = Column(Float, nullable=True)
    close = Column(Float, nullable=True)
    ltp = Column(Float, nullable=True)
    total_traded_quantity = Column(Float, nullable=True)
