import os.path

import yaml

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(dir_path, os.pardir))

with open(parent_path + "/backtestconfig.yaml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

class BacktestConfig():
    start_date = cfg['run_parameters']['start_date']
    end_date = cfg['run_parameters']['end_date']
    training_days = cfg['run_parameters']['training_days']
    look_back_days = cfg['run_parameters']['look_back_days']
    portfolio = cfg['portfolio']
    strategy = cfg['strategy']
    short_exposure = cfg['model_parameters']['short_exposure']
    margin_rate = cfg['model_parameters']['margin_rate']
    maintainance_margin = cfg['model_parameters']['maintainance_margin']
    leverage = cfg['model_parameters']['leverage']
    result_path = cfg['result_path']
    transaction_cost = cfg['transaction_cost']
    option_moneyness = cfg['model_parameters']['option_moneyness']



def get_pulse_platform_backtest_config():
    return BacktestConfig()