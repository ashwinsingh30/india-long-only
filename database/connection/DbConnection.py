from sqlalchemy import *
from sqlalchemy.orm import sessionmaker

from config.PulsePlatformConfig import get_db_config, get_local_pulse_db_schema

pulse_db_config = get_db_config('pulse_db')


class DbConnection:
    def __init__(self, config):
        self.config = config
        if config.port is None:
            self.url = config.dialect + "://" + config.username + ':' + config.password + '@' + config.host + '/' + config.dbname
        else:
            self.url = config.dialect + "://" + config.username + ':' + config.password + '@' + config.host + ':' \
                       + str(config.port) + '/' + config.dbname
        self.engine = create_engine(self.url, echo=config.echo)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()


pulse_db_connection = DbConnection(pulse_db_config)


def get_pulse_db_connection():
    return pulse_db_connection


pulse_db_schema = get_local_pulse_db_schema()
