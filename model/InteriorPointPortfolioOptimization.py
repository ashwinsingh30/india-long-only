import os

import numpy as np
import pandas as pd
from gekko import GEKKO
from scipy.linalg import sqrtm

from config.ConfiguredLogger import get_logger
from markowitzoptimization.efficient_frontier import EfficientFrontier
import cvxpy as cp

log = get_logger(os.path.basename(__file__))


class OptimizerConstraints:
    single_stock_bound = (-1, 1)
    beta_constraint = (-1, 1)
    gross_sector_constraint = 1
    net_sector_constraint = (-1, 1)
    turnover_constraint = 2
    adt_constraint = 1
    liquidity_constraint = 1
    net_exposure = (-1, 1)

    def __init__(self, single_stock_bound=(-1, 1), beta_constraint=(-1, 1), gross_sector_constraint=1,
                 net_sector_constraint=(-1, 1), turnover_constraint=2, adt_constraint=1, liquidity_constraint=1,
                 net_exposure=(-1, 1)):
        self.single_stock_bound = single_stock_bound
        self.beta_constraint = beta_constraint
        self.gross_sector_constraint = gross_sector_constraint
        self.net_sector_constraint = net_sector_constraint
        self.turnover_constraint = turnover_constraint
        self.adt_constraint = adt_constraint
        self.liquidity_constraint = liquidity_constraint
        self.net_exposure = net_exposure


def output_constraints(weight_vector, optimization_matrix, covariance_matrix, vol_limit):
    net_beta = weight_vector @ optimization_matrix['beta']
    gross_beta = np.array([np.abs(a) for a in weight_vector]) @ optimization_matrix['beta']
    net_exposure = np.sum(weight_vector)
    gross_exposure = np.sum([np.abs(a) for a in weight_vector])
    rel_vol = np.sqrt(weight_vector @ covariance_matrix @ weight_vector.T * 250)
    log.info('Net Beta: %s', net_beta)
    log.info('Gross Beta: %s', gross_beta)
    log.info('Net Exposure: %s', net_exposure)
    log.info('Gross Exposure: %s', gross_exposure)
    log.info('Vol Limit: %s, Realised Vol: %s', vol_limit, rel_vol)
    sectors = optimization_matrix.sector_map.unique()
    sector_map = optimization_matrix[['sector_map']].copy()
    sector_map.reset_index(inplace=True)
    sector_map.index.name = 'security_position'
    sector_map.reset_index(inplace=True)
    for sector in sectors:
        sector_constituents = sector_map[sector_map['sector_map'] == sector]['security_position'].values
        net_sector_exposure = np.sum(weight_vector[sector_constituents])
        gross_sector_exposure = np.sum([np.abs(a) for a in weight_vector[sector_constituents]])
        log.info('%s | Net Exposure: %s, Gross Exposure: %s', sector, net_sector_exposure, gross_sector_exposure)


def get_interior_point_optimizer():
    model = GEKKO(remote=False)
    model.options.IMODE = 3
    model.options.SOLVER = 1
    model.options.MAX_ITER = 300
    model.options.MAX_MEMORY = 6
    model.options.RTOL = 0.001
    return model


class InteriorPointPortfolioOptimization:
    optimization_matrix = pd.DataFrame()
    covariance_matrix = pd.DataFrame()
    market_volatility = 0.15 / np.sqrt(250)
    portfolio_value = 0
    constraints = OptimizerConstraints()

    def __init__(self, optimization_matrix, covariance_matrix, market_volatility, portfolio_value, constraints,
                 selected_stocks):
        self.optimization_matrix = optimization_matrix
        self.selected_stocks = selected_stocks
        self.covariance_matrix = covariance_matrix
        self.market_volatility = market_volatility
        self.portfolio_value = portfolio_value
        self.constraints = constraints
        self.var_count = len(optimization_matrix.index)

    def add_sector_constraints(self, model, variables):
        sectors = self.optimization_matrix.sector_map.unique()
        sector_map = self.optimization_matrix[['sector_map']].copy()
        sector_map.reset_index(inplace=True)
        sector_map.index.name = 'security_position'
        sector_map.reset_index(inplace=True)
        for sector in sectors:
            sector_constituents = sector_map[sector_map['sector_map'] == sector]['security_position'].values
            model.Equation(model.sum(variables[sector_constituents]) <= self.constraints.net_sector_constraint[1])
            model.Equation(model.sum(variables[sector_constituents]) >= self.constraints.net_sector_constraint[0])
            model.Equation(model.sum(
                [model.abs2(variables[i]) for i in sector_constituents]) <= self.constraints.gross_sector_constraint)

    def add_sector_constraints_cvxopt(self, model, variables, abs_variables, constraints):
        sectors = self.optimization_matrix.sector_map.unique()
        sector_map = self.optimization_matrix[['sector_map']].copy()
        sector_map.reset_index(inplace=True)
        sector_map.index.name = 'security_position'
        sector_map.reset_index(inplace=True)
        for sector in sectors:
            sector_constituents = sector_map[sector_map['sector_map'] == sector]['security_position'].values
            constraints.append(model.sum(variables[sector_constituents]) <= self.constraints.net_sector_constraint[1])
            constraints.append(model.sum(variables[sector_constituents]) >= self.constraints.net_sector_constraint[0])
            constraints.append(
                model.sum(abs_variables[sector_constituents]) <= self.constraints.gross_sector_constraint)

    def optimize_long_only_portfolio(self):
        optimizer = get_interior_point_optimizer()
        variables = np.array([])
        old_signal = self.optimization_matrix['old_signal']
        ef = EfficientFrontier(self.optimization_matrix['expected_returns'], self.covariance_matrix,
                               weight_bounds=(0, self.constraints.single_stock_bound[1]))
        ef.max_sharpe(risk_free_rate=0.10)
        self.optimization_matrix['initial_weight'] = pd.Series(ef.clean_weights())
        for i in range(0, len(self.optimization_matrix.index)):
            stock_params = self.optimization_matrix.iloc[i]
            if not np.isnan(stock_params['adt']):
                adt_constraint = self.constraints.adt_constraint * stock_params['adt'] / self.portfolio_value
                adt_bound = old_signal.iloc[i] + adt_constraint
            else:
                adt_bound = self.constraints.single_stock_bound[1]
            if not np.isnan(stock_params['adt']):
                liquidity_bound = self.constraints.liquidity_constraint * stock_params['adt'] / self.portfolio_value
            else:
                liquidity_bound = self.constraints.single_stock_bound[1]
            positive_bound = min(self.constraints.single_stock_bound[1], adt_bound, liquidity_bound)
            var = optimizer.Var(stock_params['initial_weight'], lb=0, ub=positive_bound)
            variables = np.append(variables, var)

        optimizer.Equation(
            np.dot(variables, self.optimization_matrix['beta'].values) <= self.constraints.beta_constraint[1])
        optimizer.Equation(
            np.dot(variables, self.optimization_matrix['beta'].values) >= self.constraints.beta_constraint[0])

        optimizer.Equation(optimizer.sum([optimizer.abs2(x) for x in variables]) == optimizer.Param(value=1))
        self.add_sector_constraints(optimizer, variables)

        vol_vector = [optimizer.Intermediate(np.dot(variables, self.covariance_matrix.values)[i])
                      for i in range(0, len(variables))]
        market_volatility = np.sqrt(self.market_volatility * 250)
        optimizer.Equation(optimizer.sqrt(np.dot(vol_vector, variables) * 250) <= market_volatility)
        if not (old_signal == 0).all():
            old_signal = old_signal.values
            optimizer.Equation(optimizer.sum([optimizer.abs2(variables[i] - old_signal[i])
                                              for i in range(0, len(variables))]) <=
                               self.constraints.turnover_constraint)
        optimizer.Obj(-1 * (np.dot(variables, self.optimization_matrix['expected_returns']) - 0.1) /
                      optimizer.sqrt(np.dot(vol_vector, variables) * 250))
        try:
            optimizer.solve(disp=True, debug=0)
        except Exception:
            log.error('Optimization Convergence Failed. Update Parameters !!!!!!!!!!!!!!!!!!')
        self.optimization_matrix['Weight'] = [x.value[0] for x in variables]
        output_constraints(self.optimization_matrix['Weight'], self.optimization_matrix, self.covariance_matrix,
                           market_volatility)
        return self.optimization_matrix['Weight']


    def optimize_portfolio_interior_point_method(self):
        optimizer = get_interior_point_optimizer()
        variables = np.array([])
        old_signal = self.optimization_matrix['old_signal']
        for i in range(0, len(self.optimization_matrix.index)):
            stock_params = self.optimization_matrix.iloc[i]
            script_name = self.optimization_matrix.index[i]
            if not np.isnan(stock_params['adt']):
                adt_constraint = self.constraints.adt_constraint * stock_params['adt'] / self.portfolio_value
                adt_positive_bound = old_signal.iloc[i] + adt_constraint
                adt_negative_bound = old_signal.iloc[i] - adt_constraint
            else:
                adt_positive_bound = self.constraints.single_stock_bound[1]
                adt_negative_bound = self.constraints.single_stock_bound[0]
            positive_bound = min(self.constraints.single_stock_bound[1], adt_positive_bound)
            negative_bound = max(self.constraints.single_stock_bound[0], adt_negative_bound)
            var = optimizer.Var(stock_params['centroid_vector'], lb=negative_bound, ub=positive_bound)
            variables = np.append(variables, var)
        optimizer.Equation(optimizer.sum(variables) <= self.constraints.net_exposure[1])
        optimizer.Equation(optimizer.sum(variables) >= self.constraints.net_exposure[0])
        mid = int(np.ceil(len(variables) / 2))
        net_beta_1 = optimizer.Intermediate(np.dot(variables[:mid], self.optimization_matrix['beta'].values[:mid]))
        net_beta_2 = optimizer.Intermediate(np.dot(variables[mid:], self.optimization_matrix['beta'].values[mid:]))
        optimizer.Equation((net_beta_1 + net_beta_2) >= self.constraints.beta_constraint[0])
        optimizer.Equation((net_beta_1 + net_beta_2) <= self.constraints.beta_constraint[1])
        gross_beta_1 = optimizer.Intermediate(np.dot([optimizer.abs2(x) for x in variables[:mid]],
                                                     self.optimization_matrix['beta'].values[:mid]))
        gross_beta_2 = optimizer.Intermediate(np.dot([optimizer.abs2(x) for x in variables[mid:]],
                                                     self.optimization_matrix['beta'].values[mid:]))
        optimizer.Equation((gross_beta_1 + gross_beta_2) <= optimizer.Param(value=1))
        optimizer.Equation(optimizer.sum([optimizer.abs2(x) for x in variables]) == optimizer.Param(value=1))
        self.add_sector_constraints(optimizer, variables)

        # vol_vector = [optimizer.Intermediate(np.dot(variables, self.covariance_matrix.values)[i])
        #               for i in range(0, len(variables))]
        # market_volatility = np.sqrt(self.market_volatility * 250) * 0.3
        # vol_limit = np.minimum(market_volatility, 0.05)
        # optimizer.Equation(optimizer.sqrt(np.dot(vol_vector, variables) * 250) <= vol_limit)

        if not (old_signal == 0).all():
            old_signal = old_signal.values
            optimizer.Equation(optimizer.sum([optimizer.abs2(variables[i] - old_signal[i])
                                              for i in range(0, len(variables))]) <=
                               self.constraints.turnover_constraint)
        var1 = optimizer.Intermediate(np.dot(variables[:mid], self.optimization_matrix['centroid_vector'][:mid]))
        var2 = optimizer.Intermediate(np.dot(variables[mid:], self.optimization_matrix['centroid_vector'][mid:]))
        optimizer.Obj(-1 * (var1 + var2))
        try:
            optimizer.solve(disp=True, debug=0)
        except Exception:
            log.error('Optimization Convergence Failed. Update Parameters !!!!!!!!!!!!!!!!!!')
        self.optimization_matrix['Weight'] = [x.value[0] for x in variables]
        return self.optimization_matrix['Weight']

    def optimize_portfolio_cvxopt(self):
        old_signal = self.optimization_matrix['old_signal']
        variables = cp.Variable(len(self.optimization_matrix.index))
        abs_variables = cp.Variable(len(self.optimization_matrix.index))
        constraints = []
        for i in range(0, len(self.optimization_matrix.index)):
            stock_params = self.optimization_matrix.iloc[i]
            if not np.isnan(stock_params['adt']):
                adt_constraint = self.constraints.adt_constraint * stock_params['adt'] / self.portfolio_value
                adt_positive_bound = old_signal.iloc[i] + adt_constraint
                adt_negative_bound = old_signal.iloc[i] - adt_constraint
            else:
                adt_positive_bound = self.constraints.single_stock_bound[1]
                adt_negative_bound = self.constraints.single_stock_bound[0]
            positive_bound = min(self.constraints.single_stock_bound[1], adt_positive_bound)
            negative_bound = max(self.constraints.single_stock_bound[0], adt_negative_bound)
            constraints.append(variables[i] >= negative_bound)
            constraints.append(variables[i] <= positive_bound)
        constraints.append(cp.sum(abs_variables) == 1)
        constraints.append(-abs_variables <= variables)
        constraints.append(variables <= abs_variables)
        constraints.append(cp.sum(variables) >= self.constraints.net_exposure[0])
        constraints.append(cp.sum(variables) <= self.constraints.net_exposure[1])
        constraints.append(variables @ self.optimization_matrix['beta'] >= self.constraints.beta_constraint[0])
        constraints.append(variables @ self.optimization_matrix['beta'] <= self.constraints.beta_constraint[1])
        constraints.append(abs_variables @ self.optimization_matrix['beta'] <= 1)
        self.add_sector_constraints_cvxopt(cp, variables, abs_variables, constraints)
        vol_matrix = sqrtm(self.covariance_matrix)
        market_volatility = np.sqrt(self.market_volatility * 250) * 0.3
        vol_limit = np.minimum(market_volatility, 0.04)
        constraints.append(cp.sum_squares(vol_matrix @ variables) * 2500 <= (vol_limit ** 2) * 10)
        if not (old_signal == 0).all():
            old_signal = old_signal.values
            constraints.append(cp.sum([cp.abs(variables[i] - old_signal[i])
                                       for i in range(0, len(self.optimization_matrix.index))]) <=
                               self.constraints.turnover_constraint)
        objective = cp.Minimize(-100 * (variables @ self.optimization_matrix['centroid_vector']))
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.CLARABEL, verbose=True, tol_feas=1.0e-5, tol_gap_abs=1.0e-4, tol_gap_rel=1.0e-4,
                      max_iter=200)
        self.optimization_matrix['Weight'] = variables.value
        output_constraints(self.optimization_matrix['Weight'], self.optimization_matrix, self.covariance_matrix,
                           vol_limit)
        return self.optimization_matrix['Weight']
