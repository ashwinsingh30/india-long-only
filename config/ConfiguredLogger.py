import logging

from config.PulsePlatformConfig import get_pulse_platform_config

config = get_pulse_platform_config()


class Logger:
    def __init__(self, script, filename=None, log_name=None):
        if log_name is None:
            log_name = config.log_name
        self.log = logging.getLogger(script)
        if (config.log_level == "DEBUG"):
            self.log.setLevel(logging.DEBUG)
        if (config.log_level == "INFO"):
            self.log.setLevel(logging.INFO)
        if (config.log_level == "ERROR"):
            self.log.setLevel(logging.ERROR)
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(format)
        if filename is None:
            fh = logging.FileHandler(log_name, 'w+')
        else:
            fh = logging.FileHandler(config.log_directory + filename, 'w+')
        fh.setFormatter(formatter)
        self.log.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.log.addHandler(ch)


def get_logger(script, filename=None):
    return Logger(script, filename).log


def get_autosys_logger(script, log_name):
    return Logger(script, log_name=log_name).log
