from typing import Callable, Union

import pandas as pd

from qlib.contrib.data.handler import check_transform_proc
from qlib.typehint import Literal
from ...data.dataset.handler import DataHandlerLP, DataHandler

_DEFAULT_LEARN_PROCESSORS = [
    {"class": "DropnaLabel"},
    {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
]
_DEFAULT_INFER_PROCESSORS = [
    {"class": "ProcessInf", "kwargs": {}},
    {"class": "ZScoreNorm", "kwargs": {}},
    {"class": "Fillna", "kwargs": {}},
]

DATA_KEY_TYPE = Literal["raw", "infer", "learn"]
DK_I: DATA_KEY_TYPE = "infer"


def alpha004():
    return "alpha004", "-1 * Rank($low, 9)"


def alpha008():
    return "alpha008", "Sum($close, 5) * Sum(Delta($close, 1), 5) - Sum(Ref($close, 10), 5) * Sum(Delta(Ref($close, 10), 1), 5)"


def alpha012():
    return "alpha012", "Sign(Delta($volume, 1)) * (-1 * Delta($close, 1))"


def alpha014_a():
    return "alpha014_a", "Corr($open, $volume, 10)"


def alpha014_b():
    return "alpha014_b", "Delta(Delta($close, 1) / Ref($close, 1), 3)"


def alpha019_a():
    # 这个地方我实现的不一样，我总觉得 close - delay(close,7) 和 Delta(close, 7)是一个东西
    return "alpha019_a", "-1 * Sign(Delta($close, 7))"


def alpha019_b():
    return "alpha019_b", "1 + Delta($close, 250)"


def alpha020_a():
    return "alpha020_a", "$open - Ref($high, 1)"


def alpha020_b():
    return "alpha020_b", "open - Ref($close, 1)"


def alpha020_c():
    return "alpha020_c", "open - Ref($low, 1)"


def alpha101():
    return "alpha101", "($close - $open) / ($high - $low) + 0.001"


class Formula101(DataHandlerLP):
    def __init__(self,
                 instruments="csi500",
                 start_time=None,
                 end_time=None,
                 freq="day",
                 infer_processors=None,
                 learn_processors=_DEFAULT_LEARN_PROCESSORS,
                 fit_start_time=None,
                 fit_end_time=None,
                 process_type=DataHandlerLP.PTYPE_A,
                 filter_pipe=None,
                 inst_processors=None,
                 **kwargs):
        infer_processors = []
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": self.get_feature_config(),
                    "label": kwargs.pop("label", self.get_label_config()),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processors,
            },
        }
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
            **kwargs
        )
        pass

    def get_feature_config(self):
        base = [("$open", "$open"), ("$close", "$close"), ("$volume", "$volume"),
                ("$high", "$high"), ("$low", "$low")]
        funcs = [alpha004(), alpha008(), alpha014_a(), alpha014_b(),
                 alpha019_b(), alpha019_a(),
                 alpha020_a(), alpha020_b(), alpha020_c(),
                 alpha101(),
                 ] + base
        names = list(map(lambda x: x[0], funcs))
        fields = list(map(lambda x: x[1], funcs))

        print(fields)
        print(names)

        return fields, names

    def get_label_config(self):
        return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]

    def fetch(
            self,
            selector: Union[pd.Timestamp, slice, str] = slice(None, None),
            level: Union[str, int] = "datetime",
            col_set=DataHandler.CS_ALL,
            data_key: DATA_KEY_TYPE = DK_I,
            squeeze: bool = False,
            proc_func: Callable = None,
    ) -> pd.DataFrame:
        df = self._fetch_data(
            data_storage=self._get_df_by_key(data_key),
            selector=selector,
            level=level,
            col_set=col_set,
            squeeze=squeeze,
            proc_func=proc_func,
        )
        df["$open_rank"] = df.groupby("datetime")["$open"].rank(method="min", pct=True)
        df["$high_rank"] = df.groupby("datetime")["$rank"].rank(method="min", pct=True)
        df["$volume_rank"] = df.groupby("datetime")["$volume"].rank(method="min", pct=True)
        # Alpha#8  (-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10))))
        df["alpha008"] = df.groupby("datetime")["alpha008"].rank(method="min", pct=True) * -1
        df["alpha014_rank"] = df.groupby("datetime")["alpha014_b"].rank(method="min", pct=True)
        df["alpha019b_rank"] = df.groupby("datetime")["alpha019_b"].rank(method="min", pct=True) + 1
        df["alpha020a_rank"] = df.groupby("datetime")["alpha020_a"].rank(method="min", pct=True)
        df["alpha020b_rank"] = df.groupby("datetime")["alpha020_b"].rank(method="min", pct=True)
        df["alpha020c_rank"] = df.groupby("datetime")["alpha020_c"].rank(method="min", pct=True)

        df = df.sort_values(["instrument", "datetime"])
        # Alpha#3: (-1 * correlation(rank(open), rank(volume), 10))
        df["alpha003"] = df['$open_rank'].rolling(window=10).corr(df['$volume_rank']) * -1

        # Alpha#6: (-1 * correlation(open, volume, 10))
        df["alpha006"] = df["$open"].rolling(window=10).corr(df["$volume"]) * -1

        # Alpha#14: ((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))
        df["alpha014"] = df["alpha014_rank"] * -1 * df["alpha014_a"]
        del df["alpha014_rank"]
        del df["alpha014_a"]
        del df["alpha014_b"]

        # Alpha#15 (-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))
        df["alpha015"] = df["$high_rank"].rolling(window=3).corr(df["$volume_rank"])
        df["alpha015_rank"] = df.groupby("datetime")["alpha015"].rank(method="min", pct=True)
        df["alpha015"] = df["alpha015"].rolling(window=3).sum()
        del df["alpha015_rank"]

        # Alpha#19 ((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns, 250)))))
        df["alpha019"] = df["alpha019_a"] * df["alpha019b_rank"]
        del df["alpha019_a"]
        del df["alpha019b_rank"]
        del df["alpha019b"]

        # Alpha#20  (((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open - delay(low, 1))))
        df["alpha020"] = -1 * df["alpha020a_rank"] * df["alpha020b_rank"] * df["alpha020c_rank"]
        del df["alpha020_a"]
        del df["alpha020_b"]
        del df["alpha020_c"]
        del df["alpha020a_rank"]
        del df["alpha020b_rank"]
        del df["alpha020c_rank"]

        del df["$open_rank"]
        del df["$volume_rank"]
        del df["$high_rank"]

        return df
