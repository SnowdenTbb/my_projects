from sklearn.model_selection import TimeSeriesSplit
from dataclasses import dataclass
from abc import ABC, abstractmethod
import math
import pandas
import time
import datetime
import numpy
import matplotlib.pyplot as plt
from scipy.optimize import minimize


class Matrix:

    def __init__(self, list_of_ticker, start_date, end_date, interval):
        self.list_of_ticker = list_of_ticker
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.number_of_assets = len(list_of_ticker)
        self.my_dict = {}
        self.start_date_unix_epoch = int(time.mktime(datetime.date(*self.start_date).timetuple()))
        self.end_date_unix_epoch = int(time.mktime(datetime.date(*self.end_date).timetuple()))

    def getting_price_table(self):
        for ticker in range(self.number_of_assets):
            url = f'https://query1.finance.yahoo.com/v7/finance/download/{self.list_of_ticker[ticker]}' \
                  f'?period1={self.start_date_unix_epoch}' \
                  f'&period2={self.end_date_unix_epoch}&interval=' \
                  f'{self.interval}&events=history&includeAdjustedClose=true'
            if ticker == 0:
                self.my_dict['Date'] = pandas.read_csv(url)['Date']
                self.my_dict[self.list_of_ticker[ticker]] = pandas.read_csv(url)['Close']
            else:
                self.my_dict[self.list_of_ticker[ticker]] = pandas.read_csv(url)['Close']

        price_table = pandas.DataFrame.from_dict(self.my_dict).set_index('Date')

        return price_table

    def getting_table_of_log_returns(self):
        price_table = self.getting_price_table()

        return_of_stocks = numpy.log(price_table / price_table.shift(1))

        return return_of_stocks


class MainMetricsForSolution:
    def __init__(self, tb, risk_func):
        self.tb = tb
        self.risk_func = risk_func
        self.mean = tb.mean()
        self.cov = tb.cov()
        self.list_with_opt_volatility = []
        self.bound = (0, 1)

    @property
    def define_number_of_assets(self):
        return len(self.tb.columns)

    @property
    def creating_weights(self):
        return [1 / self.define_number_of_assets] * self.define_number_of_assets

    @property
    def define_len_of_matrix(self):
        return len(self.tb)

    @property
    def define_all_bounds(self):
        return tuple(self.bound for asset in range(self.define_number_of_assets))

    def calculating_return(self, w):
        return numpy.sum(w * self.mean)

    def calculating_condition_number(self):
        return max(numpy.linalg.eig(self.cov)[0]) / min(numpy.linalg.eig(self.cov)[0])


def cleaning_table_from_nan(table: pandas.Series):

    table = table.iloc[:, 0]
    table_dimension = len(table)

    i = 1
    j = 2

    while j < table_dimension + 1:

        if table[i:j].values.size > 0 and numpy.isnan(table[i:j].values):
            table = table.drop([table[i:j].index[0]])

            i -= 1
            j -= 1

        i += 1
        j += 1

    return table


class BetaCalculationForCAPMStrategy(MainMetricsForSolution):

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


@dataclass
class EstimatingMetricsForPBR:
    metrics: MainMetricsForSolution

    @staticmethod
    def calculating_fourth_central_moment(vector: pandas.Series) -> float or int:
        mean_value_of_vector = vector.mean()
        size_of_vector = len(vector)

        if vector[1:].values.size > 0 and any(numpy.isnan(vector[1:])):
            raise ValueError('NaN values in vector')

        if vector[0:1].values.size > 0 and numpy.isnan(vector[0:1].values):
            vector = vector[1:]

        sum = 0

        for value in vector:
            sum += (value - mean_value_of_vector) ** 4

        return sum / size_of_vector

    def estimate_variance_of_estimated_variance(self, vector: pandas.Series) -> float or int:
        fourth_central_moment_estimate = self.calculating_fourth_central_moment(vector)
        dispersion_estimate = numpy.var(vector)

        size_of_vector = len(vector)

        big_q = fourth_central_moment_estimate / size_of_vector - \
            ((size_of_vector - 3) * (dispersion_estimate ** 2)) / (size_of_vector * (size_of_vector - 1))

        return big_q

    def estimating_var_of_estimated_variance_for_stock_array(self) -> float or int:
        list_with_tickers = [asset for asset in self.metrics.tb.columns]
        list_with_estimate = []

        for asset in list_with_tickers:
            estimate = self.estimate_variance_of_estimated_variance(self.metrics.tb[asset])
            list_with_estimate.append(estimate)

        return list_with_estimate

    def calculating_approximation_var_for_pbr(self) -> float or int:
        var = self.estimating_var_of_estimated_variance_for_stock_array()
        approximation_RHS = [element**(1/4) for element in var]

        return approximation_RHS

    def creating_LHS_for_pbr_opt_problem(self, w: numpy.ndarray, scaling_parameter: int = 5_000_000) -> float or int:
        approximation_var = self.calculating_approximation_var_for_pbr()
        return (numpy.dot(w.T, approximation_var)) * scaling_parameter


class ConstructingOptProblem(ABC, MainMetricsForSolution):

    @abstractmethod
    def set_LHS_for_opt_problem(self, w: numpy.ndarray):
        pass

    @abstractmethod
    def set_RHS_for_opt_problem(self):
        pass

    @abstractmethod
    def getting_opt_weights(self, **kwargs):
        pass


class CustomReturnPortfolioStrategy(ConstructingOptProblem):

    def __init__(self, tb, risk_func, r_min):
        MainMetricsForSolution.__init__(self, tb, risk_func)
        self.r_min = r_min

    def set_LHS_for_opt_problem(self, w: numpy.ndarray):
        return numpy.sum(w * self.mean)

    @property
    def set_RHS_for_opt_problem(self):
        return self.r_min

    def calculating_lower_bound_for_pbr(self):
        estimate_metrics_for_pbr = EstimatingMetricsForPBR(metrics=MainMetricsForSolution(self.tb, self.risk_func))

        constraints = [
                       {'type': 'eq', 'fun': lambda y: numpy.sum(y) - 1}]

        opt_fun = minimize(
            estimate_metrics_for_pbr.creating_LHS_for_pbr_opt_problem,
            numpy.array(self.creating_weights),
            bounds=self.define_all_bounds,
            method='SLSQP',
            constraints=constraints)

        return opt_fun.fun

    def calculating_upper_bound_for_pbr(self, **kwargs):
        analytical_solution = AnalyticalSolutionOfEfficientFrontier(self, 'beta')
        random_combination = analytical_solution.getting_combination_of_values_for_rw().sort_values('Volatility')
        combination_with_min_variance = random_combination[0:1].iloc[:, 1:]

        opt_weights = numpy.array(combination_with_min_variance.iloc[:, 1:].values[0])

        estimate_metrics_for_pbr = EstimatingMetricsForPBR(metrics=MainMetricsForSolution(self.tb, self.risk_func))

        return estimate_metrics_for_pbr.creating_LHS_for_pbr_opt_problem(opt_weights)

    def getting_opt_weights(self, lam=None, performance_based_regularization=False,  **kwargs):

        if performance_based_regularization:
            estimate_metrics = EstimatingMetricsForPBR(metrics=MainMetricsForSolution(self.tb, self.risk_func))

            upper_bound = self.calculating_upper_bound_for_pbr(weights=self.creating_weights, cov=self.cov)
            lower_bound = self.calculating_lower_bound_for_pbr()

            constraints = [{'type': 'eq', 'fun': lambda x: self.set_LHS_for_opt_problem(x) - self.set_RHS_for_opt_problem},
                           {'type': 'eq', 'fun': lambda y: numpy.sum(y) - 1},
                           {'type': 'eq', 'fun': lambda z: estimate_metrics.creating_LHS_for_pbr_opt_problem(z) - (
                                       (upper_bound / lower_bound) * (1 - lam) + lam) * lower_bound}]

            optimal_weight = minimize(
                self.risk_func,
                kwargs['weights'],
                kwargs,
                bounds=self.define_all_bounds,
                method='SLSQP',
                constraints=constraints)

            return optimal_weight.x

        constraints = [{'type': 'eq', 'fun': lambda x: self.set_LHS_for_opt_problem(x) - self.set_RHS_for_opt_problem},
                       {'type': 'eq', 'fun': lambda y: numpy.sum(y) - 1}]

        optimal_weight = minimize(self.risk_func, kwargs['weights'], kwargs, bounds=self.define_all_bounds,
                                  constraints=constraints)

        return optimal_weight.x


class CAPMPortfolioStrategy(ConstructingOptProblem, BetaCalculationForCAPMStrategy):

    def __init__(self, tb, risk_func, beta_min, index, risk_free_asset):
        BetaCalculationForCAPMStrategy.__init__(self, tb, risk_func, index, risk_free_asset)
        MainMetricsForSolution.__init__(self, tb, risk_func)
        self.beta_min = beta_min
        self.array_with_beta = self.creating_1d_array_of_beta()

    def set_LHS_for_opt_problem(self, w):
        return numpy.dot(w, self.array_with_beta.T)

    @property
    def set_RHS_for_opt_problem(self):
        return self.beta_min

    def calculating_lower_bound_for_pbr(self):
        estimate_metrics_for_pbr = EstimatingMetricsForPBR(metrics=MainMetricsForSolution(self.tb, self.risk_func))
        beta_for_gmvp = self.calculating_upper_bound_for_pbr()[1]

        constraints = [{'type': 'eq', 'fun': lambda y: numpy.sum(y) - 1},
                       {'type': 'eq', 'fun': lambda x: self.set_LHS_for_opt_problem(x) - beta_for_gmvp}]

        opt_fun = minimize(
            estimate_metrics_for_pbr.creating_LHS_for_pbr_opt_problem,
            numpy.array(self.creating_weights),
            bounds=self.define_all_bounds,
            method='SLSQP',
            constraints=constraints)

        return opt_fun.fun

    def calculating_upper_bound_for_pbr(self):

        analytical_solution = AnalyticalSolutionOfEfficientFrontier(self, 'beta')
        random_combination = analytical_solution.getting_combination_of_values_for_rw().sort_values('Volatility')
        combination_with_min_variance = random_combination[0:1].iloc[:, 1:]

        beta = random_combination[0:1].iloc[:, 0].values[0]
        opt_weights = numpy.array(combination_with_min_variance.iloc[:, 1:].values[0])

        estimate_metrics_for_pbr = EstimatingMetricsForPBR(metrics=MainMetricsForSolution(self.tb, self.risk_func))

        return estimate_metrics_for_pbr.creating_LHS_for_pbr_opt_problem(opt_weights), beta

    def getting_opt_weights(self, lam=None, performance_based_regularization=False, **kwargs):

        if performance_based_regularization:
            estimate_metrics = EstimatingMetricsForPBR(metrics=MainMetricsForSolution(self.tb, self.risk_func))

            beta = self.calculating_upper_bound_for_pbr()[1]
            upper_bound = self.calculating_upper_bound_for_pbr()[0]
            lower_bound = self.calculating_lower_bound_for_pbr()

            constraints = [
                {'type': 'eq', 'fun': lambda x: self.set_LHS_for_opt_problem(x) - beta},
                {'type': 'eq', 'fun': lambda y: numpy.sum(y) - 1},
                {'type': 'eq', 'fun': lambda z: estimate_metrics.creating_LHS_for_pbr_opt_problem(z) - (
                        (upper_bound / lower_bound) * (1 - lam) + lam) * lower_bound}]

            optimal_weight = minimize(
                self.risk_func,
                kwargs['weights'],
                kwargs,
                bounds=self.define_all_bounds,
                method='SLSQP',
                constraints=constraints)

            return optimal_weight.x

        constraints = [{'type': 'eq', 'fun': lambda x: self.set_LHS_for_opt_problem(x) - self.set_RHS_for_opt_problem},
                       {'type': 'eq', 'fun': lambda y: numpy.sum(y) - 1}]

        optimal_weight = minimize(self.risk_func, kwargs['weights'], kwargs, bounds=self.define_all_bounds,
                                  method='SLSQP', constraints=constraints)

        return optimal_weight.x


@dataclass
class AnalyticalSolutionOfEfficientFrontier:
    opt_problem: ConstructingOptProblem
    name_of_second_parameter: str

    def getting_combination_of_values_for_rw(self, counting_of_iterations=10000):

        all_weight = numpy.zeros((counting_of_iterations, self.opt_problem.define_number_of_assets))

        value_array = numpy.zeros(counting_of_iterations)
        volatility_array = numpy.zeros(counting_of_iterations)

        for iteration in range(counting_of_iterations):
            random_value = numpy.random.random(self.opt_problem.define_number_of_assets)
            weights = random_value / numpy.sum(random_value)

            all_weight[iteration, :] = weights
            value_array[iteration] = self.opt_problem.set_LHS_for_opt_problem(weights)
            volatility_array[iteration] = numpy.dot(weights.T, numpy.dot(self.opt_problem.cov, weights))
        table_of_combination_return_volatility = {self.name_of_second_parameter: value_array,
                                                  'Volatility': volatility_array}

        for quantity_of_tickers, ticker in enumerate(self.opt_problem.tb):
            table_of_combination_return_volatility[str(ticker) + 'Weight'] = [w[quantity_of_tickers] for w in all_weight]

        table_of_combination_return_volatility = pandas.DataFrame(table_of_combination_return_volatility)

        return table_of_combination_return_volatility

    def calculating_limit_for_efficient_frontier(self):
        high_limit = numpy.max(self.getting_combination_of_values_for_rw()[self.name_of_second_parameter])
        low_limit = numpy.min(self.getting_combination_of_values_for_rw()[self.name_of_second_parameter])
        return low_limit, high_limit

    def solving_opt_problem_for_sequence(self, **kwargs):

        set = numpy.linspace(*self.calculating_limit_for_efficient_frontier(), 400)

        for step in set:
            constraints = [{'type': 'eq', 'fun': lambda x: self.opt_problem.set_LHS_for_opt_problem(x) - step},
                           {'type': 'eq', 'fun': lambda y: numpy.sum(y) - 1}]

            w_opt = minimize(self.opt_problem.risk_func, kwargs['weights'], kwargs, method='SLSQP',
                             bounds=self.opt_problem.define_all_bounds, constraints=constraints)
            self.opt_problem.list_with_opt_volatility.append(w_opt['fun'])
        return self.opt_problem.list_with_opt_volatility


@dataclass
class PlottingEfficientFrontier:
    analytical_solution: AnalyticalSolutionOfEfficientFrontier

    def plotting_efficient_frontier(self, **kwargs):
        set = numpy.linspace(*self.analytical_solution.calculating_limit_for_efficient_frontier(), 400)
        plot = plt.plot(self.analytical_solution.solving_opt_problem_for_sequence(**kwargs), set)
        return plot

    def plotting_of_all_weights_combination(self):
        all_combination = self.analytical_solution.getting_combination_of_values_for_rw()
        graph = all_combination.plot.scatter(x='Volatility', y='Portfolio_Beta', marker='o', s=10, alpha=0.3,
                                             grid=True, figsize=[10, 10])
        return graph


def default_risk_fun(w: numpy.ndarray, kwargs):
    cov = kwargs['cov']
    return numpy.dot(w.T, numpy.dot(cov, w))


class CVInfo:

    def __init__(self, matrix):
        self.matrix = matrix
        self.name_of_assets = matrix.columns
        self.matrix_for_calculations = numpy.array(matrix)
        self.train_index_dictionary = {}
        self.validation_index_dictionary = {}
        self.train_set_dictionary = {}
        self.valid_set_dictionary = {}

    @property
    def calculating_len_of_matrix(self):
        return len(self.matrix)

    @property
    def define_training_dimension(self):
        len_of_matrix = self.calculating_len_of_matrix
        return math.ceil(len_of_matrix * 0.8)

    @property
    def define_total_training_data_set(self):
        training_dimension = self.define_training_dimension
        return self.matrix_for_calculations[1:training_dimension]

    @property
    def calculating_len_of_total_training_data_set(self):
        total_training_data_set = self.define_total_training_data_set
        return len(total_training_data_set)


class CrossValStrategy(ABC, CVInfo):

    @abstractmethod
    def creating_index_for_train_set(self):
        pass

    @abstractmethod
    def creating_index_for_validation_set(self):
        pass

    @abstractmethod
    def create_set_of_train(self):
        pass

    @abstractmethod
    def create_set_of_validation(self):
        pass


class LimitedTimeSeriesSplitStrategy(CrossValStrategy):

    def __init__(self, matrix, window: int, step: int):
        CVInfo.__init__(self, matrix)
        self.window = window
        self.step = step

    @property
    def creating_list_index(self):
        index_list = [j for j in range(self.calculating_len_of_total_training_data_set)]
        return index_list

    def creating_index_for_train_set(self):

        j = 0
        i = 0

        validation_window = math.ceil(0.02 * self.window)

        while j < self.calculating_len_of_total_training_data_set - self.window + 1:
            self.train_index_dictionary[f'train set № {i}'] = self.creating_list_index[j:j + self.window]
            if len(self.train_index_dictionary[f'train set № {i}']) < self.window \
                    or len(
                self.creating_list_index[j + self.window:j + self.window + validation_window]) < validation_window:
                del self.train_index_dictionary[f'train set № {i}']
                break
            else:
                j += self.step
                i += 1

        return self.train_index_dictionary

    def creating_index_for_validation_set(self):

        j = 0
        i = 0

        validation_window = math.ceil(0.02 * self.window)

        while j < self.calculating_len_of_total_training_data_set - self.window + 1:
            self.validation_index_dictionary[f'validation set № {i}'] = \
                self.creating_list_index[j + self.window:j + self.window + validation_window]
            if len(self.validation_index_dictionary[f'validation set № {i}']) < validation_window \
                    or len(self.creating_list_index[j:j + self.window]) < self.window:
                del self.validation_index_dictionary[f'validation set № {i}']
                break
            else:
                j += self.step
                i += 1
        return self.validation_index_dictionary

    def create_set_of_train(self):
        for key, value in self.creating_index_for_train_set().items():
            self.train_set_dictionary[key] = pandas.DataFrame(self.define_total_training_data_set[value])
        return self.train_set_dictionary

    def create_set_of_validation(self):
        for key, value in self.creating_index_for_validation_set().items():
            self.valid_set_dictionary[key] = pandas.DataFrame(self.define_total_training_data_set[value])
        return self.valid_set_dictionary


class TimeSeriesSplitStrategy(CrossValStrategy):

    @staticmethod
    def creating_split_method():
        tscv = TimeSeriesSplit()
        return tscv

    def creating_index_for_train_set(self):
        for index, values in enumerate(self.creating_split_method().split(self.define_total_training_data_set)):
            self.train_index_dictionary[f'train set № {index}'] = values[0]
        return self.train_index_dictionary

    def creating_index_for_validation_set(self):
        for index, values in enumerate(self.creating_split_method().split(self.define_total_training_data_set)):
            self.validation_index_dictionary[f'validation set № {index}'] = values[1]
        return self.validation_index_dictionary

    def create_set_of_train(self):
        for key, value in self.creating_index_for_train_set().items():
            self.train_set_dictionary[key] = pandas.DataFrame(self.define_total_training_data_set[value])
        return self.train_set_dictionary

    def create_set_of_validation(self):
        for key, value in self.creating_index_for_validation_set().items():
            self.valid_set_dictionary[key] = pandas.DataFrame(self.define_total_training_data_set[value])
        return self.valid_set_dictionary


class PreparingDateForCV:
    cv_strategy: CrossValStrategy

    def __init__(self, cv_strategy, risk_func, CAPM=False, r_min=None, beta_min=None, r_f=None, index=None):
        self.cv_strategy = cv_strategy
        self.risk_func = risk_func
        self.r_f = r_f
        self.index = index
        self.CAPM = CAPM
        self.r_min = r_min
        self.beta_min = beta_min
        self.dict_opt_weight = {}
        self.dict_mean_value = {}
        self.dict_portfolio_return_in_each_val_period = {}
        self.dict_return_aggregate_stock = {}
        self.dict_dispersion = {}
        self.dict_sharp_ratio = {}

    def creating_dict_with_train_data(self):
        return self.cv_strategy.create_set_of_train()

    def calculating_quantity_of_set(self):
        return range(len(self.creating_dict_with_train_data()))

    def creating_dict_with_valid_data(self):
        return self.cv_strategy.create_set_of_validation()

    def creating_array_with_r_f_for_train(self):
        index_for_train_set = self.cv_strategy.creating_index_for_train_set()
        dict_with_rf_data = {}

        r_f_without_nan = cleaning_table_from_nan(self.r_f)
        r_f_without_nan = numpy.array(r_f_without_nan)

        for key, value in index_for_train_set.items():
            dict_with_rf_data[key] = pandas.DataFrame(r_f_without_nan[value])
        return dict_with_rf_data

    def creating_array_with_rf_return_for_validation(self):
        index_for_validation_set = self.cv_strategy.creating_index_for_validation_set()
        dict_with_rf_data = {}

        r_f_without_nan = cleaning_table_from_nan(self.r_f)
        r_f_without_nan = numpy.array(r_f_without_nan)

        for key, value in index_for_validation_set.items():
            dict_with_rf_data[key] = pandas.DataFrame(r_f_without_nan[value])
        return dict_with_rf_data

    def creating_array_with_index_return_for_train(self):

        index_for_train_set = self.cv_strategy.creating_index_for_train_set()
        dict_with_index_data = {}

        index_return = numpy.array(self.index.iloc[:, 0])
        for key, value in index_for_train_set.items():
            dict_with_index_data[key] = pandas.DataFrame(index_return[value])
        return dict_with_index_data

    def creating_array_with_index_return_for_validation(self):
        index_for_validation_set = self.cv_strategy.creating_index_for_validation_set()
        dict_with_index_data = {}

        index_return = numpy.array(self.index.iloc[:, 0])

        for key, value in index_for_validation_set.items():
            dict_with_index_data[key] = pandas.DataFrame(index_return[value])
        return dict_with_index_data

    def define_final_validation_set(self):
        training_dimension = self.cv_strategy.define_training_dimension
        len_of_matrix = self.cv_strategy.calculating_len_of_matrix
        return self.cv_strategy.matrix_for_calculations[training_dimension:len_of_matrix]

    def define_train_data_for_final_validation(self, window: int):
        training_dimension = self.cv_strategy.define_training_dimension
        return self.cv_strategy.matrix_for_calculations[training_dimension - window:training_dimension]


class DisaggregatedMetricsForCV:

    def __init__(self,
                 preparing_date: PreparingDateForCV,
                 lam: int or float,
                 performance_based_regularization: bool = False):
        self.preparing_date = preparing_date
        self.lam = lam
        self.performance_based_regularization = performance_based_regularization

    @property
    def define_dimension(self):
        return self.preparing_date.calculating_quantity_of_set()

    def getting_sequences_of_opt_weight(self):

        dict_with_train_data = self.preparing_date.creating_dict_with_train_data()

        for period_number in self.define_dimension:
            data = dict_with_train_data[f'train set № {period_number}']

            if not self.preparing_date.CAPM:

                solution = CustomReturnPortfolioStrategy(data, self.preparing_date.risk_func, self.preparing_date.r_min)

                self.preparing_date.dict_opt_weight[f'opt weights in train period {period_number}'] = \
                    solution.getting_opt_weights(weights=solution.creating_weights,
                                                 cov=solution.cov,
                                                 lam=self.lam,
                                                 performance_based_regularization=self.performance_based_regularization)

            else:
                index_value = self.preparing_date.creating_array_with_index_return_for_train()[
                    f'train set № {period_number}']
                r_f_without_nan = self.preparing_date.creating_array_with_r_f_for_train()[
                    f'train set № {period_number}']

                solution = CAPMPortfolioStrategy(
                    tb=data,
                    risk_func=self.preparing_date.risk_func,
                    beta_min=self.preparing_date.beta_min,
                    index=index_value,
                    risk_free_asset=r_f_without_nan)

                self.preparing_date.dict_opt_weight[f'opt weights in train period {period_number}'] = \
                    solution.getting_opt_weights(weights=solution.creating_weights,
                                                 cov=solution.cov,
                                                 lam=self.lam,
                                                 performance_based_regularization=self.performance_based_regularization)

        return self.preparing_date.dict_opt_weight

    def calculating_mean_return_of_stock_in_each_val_period(self):
        dict_with_valid_data = self.preparing_date.creating_dict_with_valid_data()

        for period_number in self.define_dimension:
            metrics = MainMetricsForSolution(
                dict_with_valid_data[f'validation set № {period_number}'],
                self.preparing_date.risk_func)
            self.preparing_date.dict_mean_value[f'mean value in validation set № {period_number}'] = metrics.mean

        return self.preparing_date.dict_mean_value

    def calculating_dispersion_for_each_val_period(self):
        excess_cum_return = self.cumulative_excess_portfolio_return()

        sum_of_excess = sum(excess_cum_return)
        average_excess = sum_of_excess / len(excess_cum_return)

        deviation = 0
        for i, j in enumerate(excess_cum_return):
            deviation += (j - average_excess) ** 2

        dispersion_of_excess_return = (deviation / (len(excess_cum_return) - 1))

        return dispersion_of_excess_return

    def calculating_average_portfolio_return_for_each_val_period(self):
        opt_weights = self.getting_sequences_of_opt_weight()
        mean_value_for_all_val_period = self.calculating_mean_return_of_stock_in_each_val_period()

        for period_number in self.define_dimension:
            self.preparing_date.dict_portfolio_return_in_each_val_period[
                f'return in validation period {period_number}'] \
                = numpy.sum(
                opt_weights[f'opt weights in train period {period_number}'] *
                mean_value_for_all_val_period[f'mean value in validation set № {period_number}'])
        return self.preparing_date.dict_portfolio_return_in_each_val_period

    def cumulative_excess_of_index_return(self):
        list_with_cumulative_return = []

        daily_rf_return = self.preparing_date.creating_array_with_rf_return_for_validation()
        daily_index_return = self.preparing_date.creating_array_with_index_return_for_validation()

        for period_number in self.define_dimension:

            index_return_in_period = daily_index_return[f'validation set № {period_number}']
            rf_return = daily_rf_return[f'validation set № {period_number}']

            size_of_array_with_rf = len(rf_return)

            cumulative_return = 1

            i = 0
            j = 1

            while i < len(index_return_in_period):
                index_return = index_return_in_period[i:j].values[0]

                cumulative_return = cumulative_return * (1 + index_return)

                i += 1
                j += 1

            list_with_cumulative_return.append(
                (cumulative_return - 1 -
                rf_return[size_of_array_with_rf-1:size_of_array_with_rf].values[0][0] * size_of_array_with_rf)[0])

        return list_with_cumulative_return

    def sharp_ratio_of_index(self):
        excess_cum_return = self.cumulative_excess_of_index_return()

        sum_of_excess = sum(excess_cum_return)
        average_excess = sum_of_excess / len(excess_cum_return)

        deviation = 0
        for i, j in enumerate(excess_cum_return):
            deviation += (j - average_excess) ** 2

        dispersion_of_excess_return = (deviation / (len(excess_cum_return) - 1))

        return average_excess / dispersion_of_excess_return

    def cumulative_excess_portfolio_return(self):
        list_with_cumulative_return = []

        opt_weight = self.getting_sequences_of_opt_weight()
        stock_fluctuation = self.preparing_date.creating_dict_with_valid_data()
        daily_rf_return = self.preparing_date.creating_array_with_rf_return_for_validation()

        for period_number in self.define_dimension:

            stock_return_in_period = stock_fluctuation[f'validation set № {period_number}']
            opt_weights_in_period = opt_weight[f'opt weights in train period {period_number}']
            rf_return = daily_rf_return[f'validation set № {period_number}']

            size_of_array_with_rf = len(rf_return)

            cumulative_return = 1

            i = 0
            j = 1

            while i < len(stock_return_in_period):
                stock_return = stock_return_in_period[i:j].values[0]
                portfolio_return = numpy.dot(opt_weights_in_period, stock_return)

                cumulative_return = cumulative_return * (1 + portfolio_return)

                i += 1
                j += 1

            list_with_cumulative_return.append(
                cumulative_return - 1 -
                rf_return[size_of_array_with_rf-1:size_of_array_with_rf].values[0][0] * size_of_array_with_rf)

        return list_with_cumulative_return

    def sharp_ratio_of_portfolio(self):
        excess_cum_return = self.cumulative_excess_portfolio_return()


        sum_of_excess = sum(excess_cum_return)
        average_excess = sum_of_excess / len(excess_cum_return)

        deviation = 0

        for i,j in enumerate(excess_cum_return):
            deviation += (j - average_excess) ** 2

        dispersion_of_excess_return = (deviation / (len(excess_cum_return) - 1))

        return average_excess / dispersion_of_excess_return



@dataclass
class AggregatedMetricsForCV:
    disaggregated_metrics: DisaggregatedMetricsForCV

    def calculating_average_aggregate_return_for_portfolio_in_val_period(self):
        return_for_each_lambda = self.disaggregated_metrics.cumulative_excess_portfolio_return()

        sum_of_excess = sum(return_for_each_lambda)
        average_excess = sum_of_excess / len(return_for_each_lambda)

        return average_excess

    def calculating_average_sharp_ratio_for_all_val_period(self):
        sharp_ratio_in_all_period = self.disaggregated_metrics.sharp_ratio_of_portfolio()

        return sharp_ratio_in_all_period

    def calculating_sharp_ratio_of_index(self):
        sharp_ratio_in_all_period = self.disaggregated_metrics.sharp_ratio_of_index()

        return sharp_ratio_in_all_period

    def calculating_average_aggregate_dispersion_in_all_val_period(self):
        set_of_dispersion = self.disaggregated_metrics.calculating_dispersion_for_each_val_period()

        return set_of_dispersion

@dataclass
class InfoForComparingLambda:
    preparing_date: PreparingDateForCV
    start_for_lambda: float or int
    end_for_lambda: float or int
    step_for_lambda: float or int
    pbr: bool = False

    def __post_init__(self):
        self.inverse_step = int(1 / self.step_for_lambda)
        self.list_with_lambda_values = \
            [j / self.inverse_step for j in range(int(self.start_for_lambda * self.inverse_step),
                                                  int(self.end_for_lambda * self.inverse_step))]
        self.name_of_assets = [asset for asset in self.preparing_date.cv_strategy.name_of_assets]
        self.number_of_assets = len(self.name_of_assets)


@dataclass
class ComparingLambdaWithAggregateMetrics:
    info_for_comparing: InfoForComparingLambda

    def __post_init__(self):
        self.dict_aggregate_metrics_for_lambda = {'lambda_index': [],
                                                  'profitability': [],
                                                  'sd': [],
                                                  'sr_portf': [],
                                                  'sr_index': []}

    @property
    def creating_dictionary_with_aggregate_values(self):
        for lambda_element in self.info_for_comparing.list_with_lambda_values:
            disaggregated = DisaggregatedMetricsForCV(preparing_date=self.info_for_comparing.preparing_date,
                                                      lam=lambda_element,
                                                      performance_based_regularization=self.info_for_comparing.pbr)

            aggregated = AggregatedMetricsForCV(disaggregated)

            self.dict_aggregate_metrics_for_lambda['profitability'].append(
                aggregated.calculating_average_aggregate_return_for_portfolio_in_val_period())

            self.dict_aggregate_metrics_for_lambda['sd'].append(
                aggregated.calculating_average_aggregate_dispersion_in_all_val_period())

            self.dict_aggregate_metrics_for_lambda['sr_portf'].append(
                aggregated.calculating_average_sharp_ratio_for_all_val_period())

            self.dict_aggregate_metrics_for_lambda['sr_index'].append(
                aggregated.calculating_sharp_ratio_of_index()
            )

            self.dict_aggregate_metrics_for_lambda['lambda_index'].append(lambda_element)

        return self.dict_aggregate_metrics_for_lambda

    def creating_pandas_dataframe_with_aggregate_metrics(self):
        dict_with_aggregate_metrics = self.creating_dictionary_with_aggregate_values

        creating_pandas_dataframe = pandas.DataFrame(dict_with_aggregate_metrics).set_index('lambda_index')

        return creating_pandas_dataframe


@dataclass
class ComparingLambdaWithDisaggregatedMetrics:
    table_info: InfoForComparingLambda

    def __post_init__(self):
        self.dis_weights_dict = {'lambda': [],
                                 'name_of_assets': []}
        self.dis_dispersion_dict = {'lambda': []}
        self.dis_sharp_ratio_dict = {'lambda': []}

    def merger_dictionary_with_opt_weights(self):

        for index, value in enumerate(self.table_info.list_with_lambda_values):

            opt_weights = DisaggregatedMetricsForCV(preparing_date=self.table_info.preparing_date,
                                                    lam=value,
                                                    performance_based_regularization=self.table_info.pbr)\
                .getting_sequences_of_opt_weight()

            self.dis_weights_dict['lambda'] += self.table_info.number_of_assets * [value]
            self.dis_weights_dict['name_of_assets'] = (
                    len(self.table_info.list_with_lambda_values) * self.table_info.name_of_assets)

            if index == 0:

                for key in opt_weights.keys():
                    self.dis_weights_dict.setdefault(key, [])
                    self.dis_weights_dict[key].extend(opt_weights[key])

            else:

                for key in opt_weights.keys():
                    self.dis_weights_dict[key].extend(opt_weights[key])

        return self.dis_weights_dict

    def creating_pandas_dataframe_with_opt_weights(self):

        pandas_table = pandas.DataFrame(
            self.merger_dictionary_with_opt_weights()).set_index(['lambda', 'name_of_assets'])
        return pandas_table

    def merger_dictionary_with_functional(self):

        for index, value in enumerate(self.table_info.list_with_lambda_values):

            dispersion = DisaggregatedMetricsForCV(
                preparing_date=self.table_info.preparing_date,
                lam=value,
                performance_based_regularization=self.table_info.pbr).calculating_dispersion_for_each_val_period()

            self.dis_dispersion_dict['lambda'].append(value)

            if index == 0:

                for key in dispersion.keys():
                    self.dis_dispersion_dict.setdefault(key, [])
                    self.dis_dispersion_dict[key].append(dispersion[key])

            else:
                for key in dispersion.keys():
                    self.dis_dispersion_dict[key].append(dispersion[key])
        return self.dis_dispersion_dict

    def creating_pandas_dataframe_with_dispersion(self):

        return pandas.DataFrame(self.merger_dictionary_with_functional()).set_index('lambda').T

    def merger_dictionary_with_sharp_ratio(self):

        for index, value in enumerate(self.table_info.list_with_lambda_values):

            sharp_ratio = DisaggregatedMetricsForCV(preparing_date=self.table_info.preparing_date,
                                                    lam=value,
                                                    performance_based_regularization=self.table_info.pbr)\
                .sharp_ratio_of_portfolio()

            self.dis_sharp_ratio_dict['lambda'].append(value)

            if index == 0:

                for key in sharp_ratio.keys():
                    self.dis_sharp_ratio_dict.setdefault(key, [])
                    self.dis_sharp_ratio_dict[key].append(sharp_ratio[key])

            else:

                for key in sharp_ratio.keys():
                    self.dis_sharp_ratio_dict[key].append(sharp_ratio[key])

        return self.dis_sharp_ratio_dict

    def creating_pandas_dataframe_with_sharp_ratio(self):
        return pandas.DataFrame(self.merger_dictionary_with_sharp_ratio()).set_index('lambda').T


def calculating_dimension_of_train_set(table: pandas.Series) -> int:
    len_of_table = len(table)
    len_of_totally_training_set = len_of_table - 504

    return len_of_totally_training_set


def define_final_validation_window(table: pandas.Series) -> pandas.Series:
    len_of_totally_training_set = calculating_dimension_of_train_set(table)
    len_of_table = len(table)

    return table[len_of_totally_training_set:len_of_table]


def define_train_data_for_final_validation_window(table: pandas.Series, window: int, step: int) -> pandas.Series:
    len_of_totally_training_set = calculating_dimension_of_train_set(table)

    return table[len_of_totally_training_set - window + step: len_of_totally_training_set + step]


def define_r_f_in_train_period_for_fin_validation(risk_free: pandas.Series, window: int, step: int):
    len_of_totally_training_set = calculating_dimension_of_train_set(risk_free)

    return risk_free[len_of_totally_training_set - window + step: len_of_totally_training_set + step]


def define_index_return_in_train_period_for_fin_validation(index: pandas.Series, window: int, step: int):
    len_of_totally_training_set = calculating_dimension_of_train_set(index)

    return index[len_of_totally_training_set - window + step: len_of_totally_training_set + step]


def getting_opt_weights_in_train_period(opt_problem: ConstructingOptProblem, lam: int or float, pbr=False) -> list:
    opt_weight = opt_problem.getting_opt_weights(weights=opt_problem.creating_weights, cov=opt_problem.cov, lam=lam, perfomance_based_regularization=pbr)
    return opt_weight


def calculating_excess_return_of_assets(asset_return: pandas.DataFrame,
                                        risk_free_return: pandas.DataFrame,
                                        weights_of_portfolio: numpy.ndarray,
                                        portfolio=True) -> float:

    i = 0
    j = 1

    cummulative_return = 1

    if not portfolio:

        while i < len(asset_return):
            stock_return = asset_return[i:j].values[0][0]

            cummulative_return = cummulative_return * (1 + stock_return)

            i += 1
            j += 1

        spot_r_f = risk_free_return[len(risk_free_return) - 1:len(risk_free_return)].values[0][0] * len(
            risk_free_return)

        excess_return = cummulative_return - 1 - spot_r_f

        return excess_return

    while i < len(asset_return):
        stock_return = asset_return[i:j].values[0]
        stock_return = numpy.array(stock_return)

        return_in_period = numpy.dot(weights_of_portfolio, stock_return.T)

        cummulative_return = cummulative_return * (1 + return_in_period)

        i += 1
        j += 1

    spot_r_f = risk_free_return[len(risk_free_return)-1:len(risk_free_return)].values[0][0] * len(risk_free_return)

    excess_return = cummulative_return - 1 - spot_r_f

    return excess_return

def calculating_sharp_ration(list_with_sr_sequences: list) -> float:

    sum_of_excess = sum(list_with_sr_sequences)
    average_excess = sum_of_excess / len(list_with_sr_sequences)

    deviation = 0
    for i, j in enumerate(list_with_sr_sequences):
        deviation += (j - average_excess) ** 2

    dispersion_of_excess_return = (deviation / (len(list_with_sr_sequences) - 1))

    return average_excess / dispersion_of_excess_return


def creating_cumulative_return_of_portfolio1(
                                            lam: float,
                                            table_stock: pandas.Series,
                                            table_index: pandas.Series,
                                            table_r_f: pandas.Series,
                                            table_stock_final: pandas.Series,
                                            portfolio=True) -> list:



    i = 0
    j = 1

    train_data = define_train_data_for_final_validation_window(table_stock, window=250, step = 0)
    r_f_for_fv = define_r_f_in_train_period_for_fin_validation(table_r_f, window=250, step = 0)
    index_for_fv = define_index_return_in_train_period_for_fin_validation(table_index, window=250, step = 0)

    opt_problem = CAPMPortfolioStrategy(train_data,
                                                    default_risk_fun,
                                                    beta_min=0,
                                                    index=index_for_fv,
                                                    risk_free_asset=r_f_for_fv)

    weights = opt_problem.getting_opt_weights(weights=opt_problem.creating_weights, cov=opt_problem.cov, lam=lam,
                                              performance_based_regularization=True)

    excess_list=[]

    while i < len(table_stock_final) + 1:

        if (i % 5 == 0) and i > 0:

            train_data = define_train_data_for_final_validation_window(table_stock, window=250, step=i)
            r_f_for_fv = define_r_f_in_train_period_for_fin_validation(table_r_f, window=250, step=i)
            index_for_fv = define_index_return_in_train_period_for_fin_validation(table_index, window=250, step=i)

            train_for_sr = train_data[len(train_data) - 5: len(train_data)]

            r_f_for_sr = r_f_for_fv[len(r_f_for_fv) - 5: len(r_f_for_fv)]

            index_for_sr = index_for_fv[len(index_for_fv) - 5: len(index_for_fv)]


            excess = calculating_excess_return_of_assets(train_for_sr, r_f_for_sr, weights)

            if not portfolio:
                excess = calculating_excess_return_of_assets(index_for_sr, r_f_for_sr, weights, portfolio=False)


            excess_list.append(excess)

            opt_problem = CAPMPortfolioStrategy(train_data,
                                                default_risk_fun,
                                                beta_min=0,
                                                index=index_for_fv,
                                                risk_free_asset=r_f_for_fv)

            weights = opt_problem.getting_opt_weights(weights=opt_problem.creating_weights, cov=opt_problem.cov,
                                                      lam=lam, performance_based_regularization=True)


        i += 1
        j += 1

    return excess_list

def creating_cumulative_return_of_portfolio(
                                            lam: float,
                                            table_stock: pandas.Series,
                                            table_index: pandas.Series,
                                            table_r_f: pandas.Series,
                                            table_stock_final: pandas.Series,
                                            portfolio=True) -> dict:
    cumulative_return = 1

    if not portfolio:
        dict_with_main_data = {'Date': [],
                               'cumulative_return_of_index': []}

        i = 0
        j = 1

        while j < len(table_stock_final) + 1:
            date = table_stock_final[i:j].index[0]
            dict_with_main_data['Date'].append(date)

            return_of_asset = table_stock_final[i:j].values[0][0]
            cumulative_return = cumulative_return * (1 + return_of_asset)
            dict_with_main_data['cumulative_return_of_index'].append(cumulative_return)

            i += 1
            j += 1
        return dict_with_main_data

    dict_with_portfolio_return_info = {'Date': [],
                                       'portfolio_return': []}

    i = 0
    j = 1

    train_data = define_train_data_for_final_validation_window(table_stock, window=250, step = 0)
    r_f_for_fv = define_r_f_in_train_period_for_fin_validation(table_r_f, window=250, step = 0)
    index_for_fv = define_index_return_in_train_period_for_fin_validation(table_index, window=250, step = 0)

    opt_problem = CAPMPortfolioStrategy(train_data,
                                                    default_risk_fun,
                                                    beta_min=0,
                                                    index=index_for_fv,
                                                    risk_free_asset=r_f_for_fv)

    weights = opt_problem.getting_opt_weights(weights=opt_problem.creating_weights, cov=opt_problem.cov, lam=lam, performance_based_regularization=True)


    while i < len(table_stock_final):

        if (i % 5 == 0):

            train_data = define_train_data_for_final_validation_window(table_stock, window=250, step=i)
            r_f_for_fv = define_r_f_in_train_period_for_fin_validation(table_r_f, window=250, step=i)
            index_for_fv = define_index_return_in_train_period_for_fin_validation(table_index, window=250, step=i)

            opt_problem = CAPMPortfolioStrategy(train_data,
                                                default_risk_fun,
                                                beta_min=0,
                                                index=index_for_fv,
                                                risk_free_asset=r_f_for_fv)

            weights = opt_problem.getting_opt_weights(weights=opt_problem.creating_weights, cov=opt_problem.cov,
                                                      lam=lam, performance_based_regularization=True)


        date = table_stock_final[i:j].index[0]
        dict_with_portfolio_return_info['Date'].append(date)

        return_of_stock = table_stock_final[i:j].values[0]
        cumulative_return = cumulative_return * (1 + numpy.dot(weights, return_of_stock))
        dict_with_portfolio_return_info['portfolio_return'].append(cumulative_return)

        i += 1
        j += 1

    return dict_with_portfolio_return_info


def plotting_and_comparing_cumulative_return_of_portfolio_and_index(portfolio_return:
                                                                    creating_cumulative_return_of_portfolio,
                                                                    index_return:
                                                                    creating_cumulative_return_of_portfolio) -> None:

    test_dict = {**portfolio_return, **index_return}
    test_dict1 = {**portfolio_return, **index_return}

    test_dict1['portfolio_return'] = [x / numpy.std(portfolio_return['portfolio_return']) for x in portfolio_return['portfolio_return']]
    test_dict1['cumulative_return_of_index'] = [x / numpy.std(index_return['cumulative_return_of_index']) for x in index_return['cumulative_return_of_index']]

    transform_to_pandas = pandas.DataFrame(test_dict)
    transform_to_pandas.plot(x='Date', y=['portfolio_return', 'cumulative_return_of_index'])
    plt.show()

    transform_to_pandas = pandas.DataFrame(test_dict1)
    transform_to_pandas.plot(x='Date', y=['portfolio_return', 'cumulative_return_of_index'])
    plt.show()


def regularization_in_opt_function():

    """Getting table with aggregate metrics, which help us to choose better lambda parameter"""

    stock_return = pandas.read_excel('stock_return.xlsx').set_index('Date')
    index = pandas.read_excel('index.xlsx').set_index('Date')
    r_f = pandas.read_excel('risk_free.xlsx').set_index('DT')

    # preparing_date_for_cv = PreparingDateForCV(LimitedTimeSeriesSplitStrategy(stock_return, window=250, step=5),
    #                                            default_risk_fun,
    #                                            CAPM=True,
    #                                            beta_min=0,
    #                                            index=index,
    #                                            r_f=r_f)
    #
    # info_for_comparing = InfoForComparingLambda(preparing_date_for_cv,
    #                                             pbr=True,
    #                                             start_for_lambda=0,
    #                                             end_for_lambda=1.1,
    #                                             step_for_lambda=0.05)
    #
    #
    #
    # aggregate = ComparingLambdaWithAggregateMetrics(info_for_comparing)
    #
    # print('AGGREGATE VALUE OF PERFORMANCE FOR PERFORMANCE BASED REGULARIZATION ')
    # lambda_table = aggregate.creating_pandas_dataframe_with_aggregate_metrics()
    # print(lambda_table)
    # print('==========================================================')
    # best_lambda = lambda_table.sort_values('sr_portf', ascending=False).index.values[0]
    # print(f'CHOOSE BEST LAMBDA WITH EQUAL TO {best_lambda}')

    "Calculating optimal weights for final validation window"
    index_data =  define_final_validation_window(index)

    stock_data = define_final_validation_window(stock_return)

    index_return = creating_cumulative_return_of_portfolio1(lam=0, table_stock=stock_return,
                                                               table_index=index, table_r_f=r_f,
                                                               table_stock_final=index_data, portfolio=False)

    portfolio_return = creating_cumulative_return_of_portfolio1(lam=0.1, table_stock=stock_return,
                                                               table_index=index, table_r_f=r_f,
                                                               table_stock_final=stock_data)



    portfolio_markowitz = creating_cumulative_return_of_portfolio1(lam=0, table_stock=stock_return,
                                                               table_index=index, table_r_f=r_f,
                                                               table_stock_final=stock_data)


    print('====PORTFOLIO SR======')
    print(calculating_sharp_ration(portfolio_return))
    print(calculating_sharp_ration(portfolio_markowitz))
    print('====INDEX SR=====')
    print(calculating_sharp_ration(index_return))



if __name__ == '__main__':
    regularization_in_opt_function()

