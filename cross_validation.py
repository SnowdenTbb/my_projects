import numpy
import pandas

import math

from abc import ABC, abstractmethod
from sklearn.model_selection import TimeSeriesSplit


class CVInfo:
    """"
    Properties for assists in cross validation
    """
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
    """"
    Cross Validation strategy,
    which meaning of which is to build a sliding window
    """
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
    """"
    Cross Validation strategy,
    which meaning of which is to build a extending window
    """

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

    def __init__(self, cv_strategy, risk_func):
        self.cv_strategy = cv_strategy
        self.risk_func = risk_func

    def creating_dict_with_train_data(self):
        return self.cv_strategy.create_set_of_train()

    def calculating_quantity_of_set(self):
        return range(len(self.creating_dict_with_train_data()))

    def creating_dict_with_valid_data(self):
        return self.cv_strategy.create_set_of_validation()
