import numpy
import pandas
import math

from abc import ABC, abstractmethod
from scipy.optimize import minimize


class MainMetricsForSolution:
    """
    Methods for assists in defining of constraint
    """
    def __init__(self, tb, risk_func):
        self.tb = tb
        self.risk_func = risk_func

        self.mean = tb.mean()
        self.cov = tb.cov()

        self.list_with_opt_volatility = []
        self.bound = (0, 1)

    @property
    def define_number_of_assets(self) -> int:
        return len(self.tb.columns)

    @property
    def creating_weights(self) -> list:
        return [1 / self.define_number_of_assets] * self.define_number_of_assets

    @property
    def define_len_of_matrix(self) -> int:
        return len(self.tb)

    @property
    def define_all_bounds(self) -> tuple:
        return tuple(self.bound for asset in range(self.define_number_of_assets))

    def calculating_return(self, w) -> float:
        return numpy.sum(w * self.mean)


class BetaCalculationForCAPMStrategy(MainMetricsForSolution):
    """
    Methods for defining beta in CAPM model
    """
    def __init__(self, tb, risk_func, index, risk_free_asset):
        MainMetricsForSolution.__init__(self, tb, risk_func)
        self.index = index
        self.risk_free_asset = risk_free_asset

    def getting_index_return(self):

        index_return = self.index.iloc[:, 0]

        return index_return

    def calculating_difference_between_index_and_risk_free_returns(self):

        index_return = self.getting_index_return()

        risk_free_asset_return = self.risk_free_asset.iloc[:, 0]

        return index_return - risk_free_asset_return

    def calculating_cov_between_stock_and_difference(self, stock: pandas.Series) -> float:

        difference = self.calculating_difference_between_index_and_risk_free_returns()
        dimension_of_stock_series = len(stock)

        if len(difference) != len(stock):
            raise ValueError('Mismatched dimension')

        mean_return_of_stock = numpy.mean(stock)
        mean_value_of_difference = numpy.mean(difference)

        sum_of_residuals = 0

        for j in range(1, dimension_of_stock_series):
            sum_of_residuals += (stock[j] - mean_return_of_stock) * (difference[j] - mean_value_of_difference)

        cov = sum_of_residuals / (dimension_of_stock_series - 1)

        return cov

    def calculating_variance_of_difference(self) -> float:

        difference = self.calculating_difference_between_index_and_risk_free_returns()

        return numpy.var(difference)

    def calculating_beta_for_stock(self, stock):

        cov = self.calculating_cov_between_stock_and_difference(stock)

        var = self.calculating_variance_of_difference()

        beta = cov / var

        return beta

    def creating_1d_array_of_beta(self):

        list_with_beta = []

        list_with_tickers = [j for j in self.tb.columns]

        for ticker in list_with_tickers:
            beta = self.calculating_beta_for_stock(self.tb[ticker])

            list_with_beta.append(beta)

        return numpy.array(list_with_beta)


def default_risk_fun(w: numpy.ndarray, kwargs):
    cov = kwargs['cov']
    return numpy.dot(w.T, numpy.dot(cov, w))


def ridge_risk_fun(w: numpy.ndarray, kwargs):
    cov, lam = kwargs['cov'], kwargs['lam']
    return numpy.dot(w.T, numpy.dot(cov, w)) + lam * math.sqrt(numpy.dot(w.T, w)) * 5_000_000  # lambda [0, 1]


def lasso_risk_fun(w: numpy.ndarray, kwargs):
    cov, lam = kwargs['cov'], kwargs['lam']
    return numpy.dot(w.T, numpy.dot(cov, w)) + lam * numpy.sum(abs(w)) * 5_000_000  # lambda [0, 1]


class ConstructingOptProblem(ABC, MainMetricsForSolution):

    @abstractmethod
    def set_lhs_for_opt_problem(self, w: numpy.ndarray):
        pass

    @abstractmethod
    def set_rhs_for_opt_problem(self):
        pass

    @abstractmethod
    def getting_opt_weights(self, **kwargs):
        pass


class CustomReturnPortfolioStrategy(ConstructingOptProblem):
    """
    Classical portfolio optimization problem
    with constraint on minimum of portfolio return
    """
    def __init__(self, tb, risk_func, r_min):
        MainMetricsForSolution.__init__(self, tb, risk_func)
        self.r_min = r_min

    def set_lhs_for_opt_problem(self, w: numpy.ndarray):
        return numpy.sum(w * self.mean)

    @property
    def set_rhs_for_opt_problem(self):
        return self.r_min

    def getting_opt_weights(self, pbr=False, **kwargs):
        constraints = [{'type': 'eq', 'fun': lambda x: self.set_lhs_for_opt_problem(x) - self.set_rhs_for_opt_problem},
                       {'type': 'eq', 'fun': lambda y: numpy.sum(y) - 1}]

        optimal_weight = minimize(self.risk_func, kwargs['weights'], kwargs, bounds=self.define_all_bounds,
                                  constraints=constraints)

        return optimal_weight.x


class CAPMPortfolioStrategy(ConstructingOptProblem, BetaCalculationForCAPMStrategy):
    """
    Classical Portfolio optimization problem
    with constraint on beta of portfolio.
    """

    def __init__(self, tb, risk_func, beta_min, index, risk_free_asset):
        BetaCalculationForCAPMStrategy.__init__(self, tb, risk_func, index, risk_free_asset)
        MainMetricsForSolution.__init__(self, tb, risk_func)

        self.beta_min = beta_min
        self.array_with_beta = self.creating_1d_array_of_beta()

    def set_lhs_for_opt_problem(self, w):
        return numpy.dot(w, self.array_with_beta.T)

    @property
    def set_rhs_for_opt_problem(self):
        return self.beta_min

    def getting_opt_weights(self, **kwargs):

        constraints = [{'type': 'eq', 'fun': lambda x: self.set_lhs_for_opt_problem(x) - self.beta_min},
                       {'type': 'eq', 'fun': lambda y: numpy.sum(y) - 1}]

        optimal_weight = minimize(self.risk_func, kwargs['weights'], kwargs, bounds=self.define_all_bounds,
                                  method='SLSQP', constraints=constraints)

        return optimal_weight.x