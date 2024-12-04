import os

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.linalg import sqrtm

from config.ConfiguredLogger import get_logger
from markowitzoptimization.efficient_frontier import EfficientFrontier

log = get_logger(os.path.basename(__file__))


class LongOnlyOptimizerConstraints:
    single_stock_bound = (0, 1)
    beta_constraint = (0, 1)
    gross_sector_constraint = 1
    turnover_constraint = 2
    adt_constraint = 1
    liquidity_constraint = 1

    def __init__(self, single_stock_bound=(0, 1), beta_constraint=(0, 1), gross_sector_constraint=1,
                 turnover_constraint=2, adt_constraint=1, liquidity_constraint=1):
        self.single_stock_bound = single_stock_bound
        self.beta_constraint = beta_constraint
        self.gross_sector_constraint = gross_sector_constraint
        self.turnover_constraint = turnover_constraint
        self.adt_constraint = adt_constraint
        self.liquidity_constraint = liquidity_constraint


class LongOnlyPortfolioOptimization:
    optimization_matrix = pd.DataFrame()
    covariance_matrix = pd.DataFrame()
    market_volatility = 0.15 / np.sqrt(250)
    portfolio_value = 0
    constraints = LongOnlyOptimizerConstraints()
    old_signal = pd.Series()

    def __init__(self, optimization_matrix, covariance_matrix, market_volatility, portfolio_value, constraints,
                 old_signal):
        self.optimization_matrix = optimization_matrix
        self.covariance_matrix = covariance_matrix
        self.market_volatility = market_volatility
        self.portfolio_value = portfolio_value
        self.constraints = constraints
        self.old_signal = old_signal
        if self.old_signal is None:
            self.old_signal = pd.Series(0, index=self.optimization_matrix.index, dtype='float64')
        self.old_signal = self.old_signal.reindex(np.union1d(
            self.optimization_matrix.index, self.old_signal.index)).fillna(0)

    def add_long_only_sector_constraints(self, model, variables, constraints):
        sectors = self.optimization_matrix.sector_map.unique()
        sector_map = self.optimization_matrix[['sector_map']].copy()
        sector_map.reset_index(inplace=True)
        sector_map.index.name = 'security_position'
        sector_map.reset_index(inplace=True)
        for sector in sectors:
            sector_constituents = sector_map[sector_map['sector_map'] == sector]['security_position'].values
            constraints.append(
                model.sum(variables[sector_constituents]) <= self.constraints.gross_sector_constraint)

    def optimize_number_of_shares(self):
        variables = cp.Variable(len(self.optimization_matrix.index), integer=True)
        constraints = []
        for i in range(0, len(self.optimization_matrix.index)):
            stock_params = self.optimization_matrix.iloc[i]
            script_name = self.optimization_matrix.index[i]
            if not np.isnan(stock_params['adt']):
                adt_constraint = self.constraints.adt_constraint * stock_params['adt'] / stock_params['close_price']
                adt_positive_bound = self.old_signal.loc[script_name] + adt_constraint
            else:
                adt_positive_bound = (self.constraints.single_stock_bound[1] * self.portfolio_value) / stock_params[
                    'close_price']
            max_shares = (self.constraints.single_stock_bound[1] * self.portfolio_value) / stock_params['close_price']
            positive_bound = min(max_shares, adt_positive_bound)
            constraints.append(variables[i] >= 0)
            constraints.append(variables[i] <= positive_bound)
        variable_value = np.array([variables[i] * self.optimization_matrix['close_price'][i]
                                   for i in range(0, len(self.optimization_matrix.index))])

        old_signal = self.optimization_matrix['old_signal']
        if not (old_signal == 0).all():
            old_signal = old_signal.values
            constraints.append(cp.sum([cp.abs(variable_value[i] - (old_signal[i] * self.portfolio_value))
                                       for i in range(0, len(self.optimization_matrix.index))]) <=
                               self.constraints.turnover_constraint * self.portfolio_value)

        constraints.append(variables @ self.optimization_matrix['close_price'] <= self.portfolio_value)
        constraints.append(variables @ self.optimization_matrix['close_price'] >= 0.9 * self.portfolio_value)
        objective = cp.Minimize(-1 * (variable_value @ (self.optimization_matrix['Weight'] * self.portfolio_value)))
        problem = cp.Problem(objective, constraints)
        problem.solve(solver='SCIPY', verbose=True, scipy_options={'maxiter': 200})
        self.optimization_matrix['no_of_shares'] = variables.value
        self.optimization_matrix['Weight'] = variables.value * self.optimization_matrix['close_price']
        self.optimization_matrix['Weight'] = self.optimization_matrix['Weight'] / self.optimization_matrix[
            'Weight'].sum()

    def optimize_long_only_portfolio(self):
        ef = EfficientFrontier(self.optimization_matrix['expected_returns'], self.covariance_matrix,
                               weight_bounds=(0, self.constraints.single_stock_bound[1]))
        ef.max_sharpe(risk_free_rate=0.05)
        self.optimization_matrix['initial_weight'] = pd.Series(ef.clean_weights(),
                                                               index=self.optimization_matrix.index).fillna(0)
        self.optimization_matrix.loc[self.optimization_matrix['initial_weight'] == 0, 'Weight'] = 0
        self.optimization_matrix['Weight'] = self.optimization_matrix['Weight'] / self.optimization_matrix['Weight'].sum()
        self.covariance_matrix = self.covariance_matrix.loc[self.optimization_matrix.index][
            self.optimization_matrix.index]
        self.optimize_number_of_shares()
        return self.optimization_matrix[['Weight', 'no_of_shares']]
