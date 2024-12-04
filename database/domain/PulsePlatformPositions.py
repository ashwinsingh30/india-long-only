from sqlalchemy import *
from sqlalchemy.ext.declarative import declarative_base

from database.connection.DbConnection import pulse_db_schema

metadata = MetaData(schema=pulse_db_schema)
Base = declarative_base(metadata=metadata)


class PulsePlatformPositions(Base):
    __tablename__ = 'pulse_platform_positions'
    __schemaname__ = pulse_db_schema
    script_name  = Column(String, nullable=False)
    contract = Column(String,primary_key=True,nullable=False)
    trade_date = Column(Date, primary_key=True, nullable=False)
    model = Column(String,primary_key=True,nullable=False)
    position_size = Column(Float)
    model_weight = Column(Float)
    exposure_weight = Column(Float)
    exposure = Column(Float)