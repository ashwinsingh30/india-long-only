import pandas as pd

from database.connection.DbConnection import get_pulse_db_connection
from database.domain.ModelPerformance import ModelPerformance
from utils.TradingPlatformUtils import pivot_model_performance, pivot_model_turnover

dbConnection = get_pulse_db_connection()


def get_performance_for_models(models):
    performance = pd.read_sql(dbConnection.session.query(ModelPerformance)
                              .filter(ModelPerformance.model_name.in_(models)).statement,
                              dbConnection.session.bind)
    return pivot_model_performance(performance), pivot_model_turnover(performance)


def get_performance_for_models_unpivot(models, universe):
    performance = pd.read_sql(dbConnection.session.query(ModelPerformance)
                              .filter((ModelPerformance.model_name.in_(models)) &
                                      (ModelPerformance.universe == universe)).statement,
                              dbConnection.session.bind)
    return performance


def get_performance_for_models_and_universe(models, universe):
    performance = pd.read_sql(dbConnection.session.query(ModelPerformance)
                              .filter((ModelPerformance.model_name.in_(models)) &
                                      (ModelPerformance.universe == universe)).statement,
                              dbConnection.session.bind)
    return pivot_model_performance(performance), pivot_model_turnover(performance)




def get_performance_for_models_from_date(models, start_date):
    performance = pd.read_sql(dbConnection.session.query(ModelPerformance)
                              .filter(ModelPerformance.model_name.in_(models) &
                                      (ModelPerformance.trade_date >= start_date)).statement,
                              dbConnection.session.bind)
    return pivot_model_performance(performance)


def get_all_performance_for_models_from_date(start_date):
    performance = pd.read_sql(dbConnection.session.query(ModelPerformance)
                              .filter(ModelPerformance.trade_date >= start_date).statement,
                              dbConnection.session.bind)
    return pivot_model_performance(performance)


def get_date_before_x_days(trade_date, x):
    return pd.read_sql(dbConnection.session.query(ModelPerformance.trade_date)
                       .filter(ModelPerformance.trade_date <= trade_date)
                       .distinct()
                       .order_by(ModelPerformance.trade_date.desc())
                       .limit(x)
                       .statement, dbConnection.session.bind).min()['trade_date']


def get_model_trailing_performances(models, trade_date, x):
    start_date = get_date_before_x_days(trade_date, x)
    return get_performance_for_models_from_date(models, start_date)


def get_model_performance_between_dates(models, start_date, end_date):
    performance = pd.read_sql(dbConnection.session.query(ModelPerformance)
                              .filter(ModelPerformance.model_name.in_(models) &
                                      (ModelPerformance.trade_date >= start_date) &
                                      (ModelPerformance.trade_date <= end_date)).statement,
                              dbConnection.session.bind)
    return pivot_model_performance(performance)


def get_model_performance_between_dates_no_pivot(models, start_date, end_date, universe):
    return pd.read_sql(dbConnection.session.query(ModelPerformance)
                       .filter(ModelPerformance.model_name.in_(models) &
                               (ModelPerformance.universe.in_(universe)) &
                               (ModelPerformance.trade_date >= start_date) &
                               (ModelPerformance.trade_date <= end_date)).statement,
                       dbConnection.session.bind)
