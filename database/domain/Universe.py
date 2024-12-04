from sqlalchemy import *
from sqlalchemy.ext.declarative import declarative_base

from database.connection.DbConnection import pulse_db_schema

metadata = MetaData(schema=pulse_db_schema)
Base = declarative_base(metadata=metadata)


class Universe(Base):
    __tablename__ = 'universe'
    __schemaname__ = pulse_db_schema
    universe_hash = Column(String, primary_key=True, nullable=False)
    script_name = Column(String)
    as_of_date = Column(Date)
    universe_name = Column(String)
    sector = Column(String)
    industry = Column(String)
    sub_industry = Column(String)
    market_cap = Column(Float)
    weight = Column(Float)
