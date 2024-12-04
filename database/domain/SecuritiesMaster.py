from sqlalchemy import *
from sqlalchemy.ext.declarative import declarative_base

from database.connection.DbConnection import pulse_db_schema

metadata = MetaData(schema=pulse_db_schema)
Base = declarative_base(metadata=metadata)


class SecuritiesMaster(Base):
    __tablename__ = 'securities_master'
    __schemaname__ = pulse_db_schema
    securities_master_hash = Column(String, primary_key=True, nullable=False)
    script_name = Column(String)
    isin = Column(String)
    bbticker = Column(String)
    start_date = Column(Date)
    end_date = Column(Date)
