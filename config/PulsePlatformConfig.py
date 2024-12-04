import os.path

import yaml

from utils.DateUtils import parse_time

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(dir_path, os.pardir))

with open(parent_path + "/pulseplatformconfig.yaml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)


class Configuration:
    project_directory = cfg["project_directory"]
    ui_styles = project_directory + cfg["ui"]["base_directory"] + cfg["ui"]["styles_file"]
    ui_logo = project_directory + cfg["ui"]["base_directory"] + cfg["ui"]["logo"]
    ui_base_directory = project_directory + cfg["ui"]["base_directory"]
    base_directory = project_directory + cfg["data_directory"]
    daily_analysis = project_directory + cfg["reports"]["base_directory"] + cfg["reports"]["daily_analysis"]
    log_directory = cfg["project_directory"] + cfg["logs"]["log_directory"]
    log_name = log_directory + cfg["logs"]["name"]
    autosys_log = log_directory + cfg["logs"]["autosys_log"]
    log_level = cfg["logs"]["level"]
    run_mode = cfg['run_mode']
    chrome_binary = cfg['data_ingest']['chrome_binary']
    chrome_driver = cfg['data_ingest']['chrome_driver']
    user_agent = cfg['data_ingest']['user_agent']
    portfolios = cfg['model_params']['portfolios']
    broker = cfg['model_params']['broker']
    post_orders = cfg['model_params']['post_orders']
    rollover = cfg['model_params']['rollover']
    trade_type = cfg['model_params']['trade_type']
    twap_start = parse_time(cfg['model_params']['twap_start'])
    twap_end = parse_time(cfg['model_params']['twap_end'])
    constrain_turnover = cfg['model_params']['constrain_turnover']
    turnover_multiplier = cfg['model_params']['turnover_multiplier']
    kite_access_token = cfg['kite_api']['access_token']
    kite_api_key = cfg['kite_api']['api_key']
    kite_api_secret = cfg['kite_api']['api_secret']
    kite_request_token = cfg['kite_api']['request_token']
    autosys_db_table = cfg['autosys']['db_table']
    autosys_pool_executors = cfg['autosys']['pool_executors']
    autosys_coalesce = cfg['autosys']['coalesce']
    autosys_max_instances = cfg['autosys']['max_instances']
    autosys_jobs = cfg['autosys']['jobs']


class DbConfiguration():
    def __init__(self, dbname):
        self.dbname = cfg["database"][dbname]["dbname"]
        self.schema = cfg["database"][dbname]["schema"]
        self.username = cfg["database"][dbname]["username"]
        self.password = cfg["database"][dbname]["password"]
        self.dialect = cfg["database"][dbname]["dialect"]
        self.host = cfg["database"][dbname]["host"]
        self.port = cfg["database"][dbname]["port"]
        self.echo = cfg["database"][dbname]["logging"]


def get_pulse_platform_config():
    return Configuration()


def get_db_config(dbname):
    return DbConfiguration(dbname)


def get_local_pulse_db_schema():
    return DbConfiguration('pulse_db').schema