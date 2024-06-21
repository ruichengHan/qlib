import numpy as np
import pandas as pd


class SeriesProcessor:
    def fit(self, series: pd.Series):
        raise NotImplementedError()


class MinMaxSProcessor(SeriesProcessor):

    def __init__(self, min=None, max=None):
        self.min_value = min
        self.max_value = max

    def fit(self, series: pd.Series):
        if self.min_value is not None:
            series[series < self.min_value] = self.min_value
        if self.max_value is not None:
            series[series > self.max_value] = self.max_value

        return series

    def __str__(self):
        return f"MinMaxProcessor min = {self.min_value} max = {self.max_value}"


class ZScoreSProcessor(SeriesProcessor):
    def __init__(self, fit_start_time, fit_end_time):
        self.start_time = fit_start_time
        self.end_time = fit_end_time
        pass

    def fit(self, series: pd.Series):
        part_series = series.loc[slice(pd.Timestamp(self.start_time), pd.Timestamp(self.end_time))]
        mean = np.nanmean(part_series, axis=0)
        std = np.nanstd(part_series, axis=0)
        part_series = (series - mean) / std
        return part_series

    def __str__(self):
        return f"ZScoreProcessor start time = f{self.start_time} end time = f{self.end_time}"


class Fillna(SeriesProcessor):
    """Process NaN"""

    def __init__(self, fields_group=None, fill_value=0):
        self.fields_group = fields_group
        self.fill_value = fill_value

    def fit(self, series: pd.Series):
        series.fillna(self.fill_value, inplace=True)
        return series
