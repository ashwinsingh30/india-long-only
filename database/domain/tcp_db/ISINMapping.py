from sqlalchemy import *
from sqlalchemy.ext.declarative import declarative_base

metadata = MetaData(schema='marketdata')
Base = declarative_base(metadata=metadata)


class ISINMapping(Base):
    __tablename__ = 'isin_mapping'
    __schemaname__ = 'marketdata'
    date = Column(Date, nullable=False, primary_key=True)
    tcp_id = Column(String, nullable=False, primary_key=True)
    exchange_ticker = Column(String, nullable=True)
    isin = Column(String, nullable=True)
    equity_series_code = Column(Float, nullable=True)