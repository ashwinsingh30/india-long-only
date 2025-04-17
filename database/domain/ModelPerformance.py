from sqlalchemy import *
from sqlalchemy.ext.declarative import declarative_base

metadata = MetaData(schema='pulse')
Base = declarative_base(metadata=metadata)


class ModelPerformance(Base):
    __tablename__ = 'model_performance'
    __schemaname__ = 'pulse'
    model_performance_hash = Column(String, primary_key=True, nullable=False)
    trade_date = Column(Date, nullable=False)
    model_name = Column(String, nullable=False)
    universe = Column(String, nullable=False)
    model_return = Column(Float, nullable=False)
    model_turnover = Column(Float, nullable=False)
