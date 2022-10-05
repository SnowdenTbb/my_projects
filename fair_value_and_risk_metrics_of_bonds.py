from typing import Dict, List
from scipy.optimize import minimize

import numpy

def discounting_cash_flow_with_constant_yield(yield_rate: float, kwargs) -> float:
    """
    Input parameters: yield rate and pandas.DataFrame
    with cash_flow and term of this cash flow
    columns must be named: 'total_cf', 'term'
    """

    cash_flow = kwargs['table_with_cf']

    discount_cf = 0
    i = 0

    while i < len(cash_flow):

        total_cf_in_period = cash_flow['total_cf'].values[i]
        term = cash_flow['term'].values[i]

        discount_cf += total_cf_in_period / (1 + yield_rate) ** term

        i +=  1

    return discount_cf


def discounting_cash_flow_with_non_constant_yield(z_spread, kwargs) -> float:
    """
    Input parameters: yield rate, pandas.DataFrame
    with cash_flow and term. columns must be named: 'total_cf', 'term'
    Also need a list of yield.
    """

    table_with_cash_flow = kwargs['table_with_cf']
    list_with_bond_term = list(table_with_cash_flow['term'])

    list_with_yield = kwargs['list_with_yield']

    column_with_cf = table_with_cash_flow['total_cf']

    fair_value = 0

    for index, value in enumerate(column_with_cf.values):
        cash_flow = value

        term = list_with_bond_term[index]
        spot_rate = list_with_yield[index]

        discounting_cf = cash_flow / (1 + spot_rate) ** term

        fair_value += discounting_cf

    return fair_value




class InformationAboutBond:

    def __init__(self, table_with_cf, dict_with_yield_and_term, bond_dirty_price, isin_of_bond):

        self.table_with_cf = table_with_cf

        self.dict_with_yield_and_term = dict_with_yield_and_term

        self.bond_dirty_price = bond_dirty_price

        self.isin_of_bond = isin_of_bond
        self.list_with_bond_term = list(self.table_with_cf['term'])

    def __repr__(self):
        return '{self.__class__.__name__}(ISIN = {self.isin_of_bond})'.format(self=self)

    def extracting_yield_for_bond_term(self) -> List[float]:
        list_with_bond_yield = []

        for term in self.list_with_bond_term:
            term_yield = self.dict_with_yield_and_term[term]
            list_with_bond_yield.append(term_yield)

        return list_with_bond_yield


class FairValueOfBond(InformationAboutBond):
    """
    Methods, which calculating
    fair value of bond, ytm
    """

    def __init__(self, table_with_cf, dict_with_yield_term_pairs, bond_dirty_price, face_value, isin):
        InformationAboutBond.__init__(self,
                                      table_with_cf,
                                      dict_with_yield_term_pairs,
                                      bond_dirty_price,
                                      isin)
        self.face_value = face_value


    def calculating_fv_of_bond(self, z_spread: float) -> float:
        """ Discounting all cash flows.
        For discounting parameters we take point from
        zero coupon rate curve
        """
        list_with_bond_yield = self.extracting_yield_for_bond_term()

        kwargs = dict(table_with_cf=self.table_with_cf,
                      list_with_yield=list_with_bond_yield)

        discounting_cash_flow = discounting_cash_flow_with_non_constant_yield(z_spread, kwargs)

        return discounting_cash_flow / self.face_value

    def calculating_z_spread(self):
        """Solve optimization problem,
        which find a such additional number in denominator
        where bond dirty price will be equal
        to all discount cash flow
        """

        kwargs = dict(list_with_yield=self.extracting_yield_for_bond_term(),
                      table_with_cf=self.table_with_cf)

        constraints = [{
            'type': 'eq', 'fun': lambda y: discounting_cash_flow_with_non_constant_yield(y, kwargs) / self.face_value}]

        solution = minimize(discounting_cash_flow_with_non_constant_yield,
                            numpy.array(0),
                            kwargs,
                            method='SLSQP',
                            constraints=constraints)

        return solution.x[0]

    def calculating_ytm_of_bond(self):
        """
        Solve optimization problem,
        which find a such yield
        where bond dirty price
        will be equal to all discount cash flows
        """

        kwargs = dict(yield_rate=0,
                      table_with_cf=self.table_with_cf)

        constraints = [{
            'type': 'eq', 'fun': lambda y: discounting_cash_flow_with_constant_yield(y, kwargs) / self.face_value}]

        solution = minimize(discounting_cash_flow_with_constant_yield,
                            numpy.array(0),
                            kwargs,
                            method='SLSQP',
                            constraints=constraints)

        return solution.x[0]


class BondPropertiesForScenarioAnalysis(InformationAboutBond):
    """
    Methods for scenario changes of bond metrics
    """

    def __init__(self, table_with_cf, dict_with_yield_and_term, bond_price, isin_of_bond, z_spread):
        InformationAboutBond.__init__(self,
        table_with_cf,
        dict_with_yield_and_term,
        bond_price,
        isin_of_bond)

        self.z_spread = z_spread
        self.list_with_yield = self.extracting_yield_for_bond_term()

    def calculating_cash_flow_with_yield_changing(self, yield_changing: float) -> Dict[str, float]:

        dict_with_cf_with_changing_yield = dict()

        dict_with_yield = dict(
        yield_plus_one_pp = [yield_rate + yield_changing for yield_rate in self.list_with_yield],
        yield_minus_one_pp = [yield_rate - yield_changing for yield_rate in self.list_with_yield],
        yield_without_change = self.list_with_yield)

        for key,value in dict_with_yield.items():

            key_for_cf_dict = key.split('_')[1]

            kwargs = dict(table_with_cf = self.table_with_cf,
            list_with_yield=value)

            discount_total_cf = discount_cash_flow_with_non_constant_yield(self.z_spread, kwargs)

            dict_with_cf_with_changing_yield[key_for_cf_dict] = discount_total_cf * 1_000

        return dict_with_cf_with_changing_yield


    def calculating_cash_flow_with_z_spread_changing(self, yield_changing: float) -> dict:
        dict_with_cf_with_changing_yield = dict()

        dict_with_yield = dict(
        zspread_plus_one_pp = self.z_spread + yield_changing,
        zspread_minus_one_pp = self.z_spread - yield_changing)

        for key,value in dict_with_yield.items():

            key_for_cf_dict = key.split('_')[1]

            kwargs = dict(table_with_cf = self.table_with_cf,
            list_with_yield=self.list_with_yield)

            discount_total_cf = discount_cash_flow_with_non_constant_yield(value, kwargs)

            dict_with_cf_with_changing_yield[key_for_cf_dict] = discount_total_cf * 1_000

        return dict_with_cf_with_changing_yield


    class RiskMetricsOfBond(BondPropertiesForScenarioAnalysis):
        """
        information about risk metrics of bond:
        1. effective duration
        2. dollar duration
        4. convexity
        """


        def __init__(self, table_with_cf, dict_with_yield_and_term, bond_price, isin_of_bond, z_spread):

            BondPropertiesForScenarioAnalysis.__init__(self,
            table_with_cf,
            dict_with_yield_and_term,
            bond_price,
            isin_of_bond,
            z_spread)

    def calculating_dv01(self) -> float:
        effective_duration = self.calculating_effective_duration()

        kwargs = dict(table_with_cf = self.table_with_cf,
        list_with_yield=self.list_with_yield)

        total_discount_cash_flow = discount_cash_flow_with_non_constant_yield(self.z_spread, kwargs)

        dv01 = (effective_duration / 10_000) * total_discount_cash_flow

        return dv01

    def calculating_effective_duration(self) -> float:
        dict_with_cash_flow = self.calculating_cash_flow_with_yield_changing(0.0001)

        discount_cash_yield_growth = dict_with_cash_flow['plus']
        discount_cash_yield_decline = dict_with_cash_flow['minus']

        initial_cash_flow = dict_with_cash_flow['without']

        effective_duration = (discount_cash_yield_decline - discount_cash_yield_growth) / (2 *
        initial_cash_flow * 0.0001)

        return effective_duration

    def calculating_convexity_of_bond(self) -> float:
        dict_with_cash_flow = self.calculating_cash_flow_with_yield_changing(0.0001)

        discount_cash_yield_growth = dict_with_cash_flow['plus']
        discount_cash_yield_decline = dict_with_cash_flow['minus']
        initial_cash_flow = dict_with_cash_flow['without']

        convexity = (discount_cash_yield_decline + discount_cash_yield_growth - 2 * initial_cash_flow) / (
                initial_cash_flow * 0.0001 ** 2)

        return convexity

