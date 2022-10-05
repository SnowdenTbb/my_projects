import pandas
import numpy

import time
import datetime


class MatrixWithPriceOfStocks:
    """"
    Data loader, which will be called by user
    for downloading dynamic of stocks price changing from SP&500
    """
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


def cleaning_column_from_nan(table: pandas.Series) -> pandas.Series:

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


def getting_ruonia_rate_as_risk_free(start_date: tuple, end_date: tuple) -> pandas.DataFrame:
    start_day = start_date[0]
    start_mounth = start_date[1]
    start_year = start_date[2]

    end_day = end_date[0]
    end_mounth = end_date[1]
    end_year = end_date[2]

    URL = f'https://www.cbr.ru/Queries/UniDbQuery/DownloadExcel/115850?Posted=True&From=' \
          f'{start_day}.{start_mounth}.{start_year}&' \
          f'To={end_day}.{end_mounth}.{end_year}&FromDate={start_mounth}%2F{start_day}%2F{start_year}' \
          f'&ToDate={end_mounth}%2F{end_day}%2F{end_year}'


    pandas_dataframe = pandas.read_excel(URL, engine="openpyxl")[::-1].set_index('DT')

    ruonia_dataframe = pandas_dataframe['ruo'] / (100 * 365)

    return ruonia_dataframe

