from typing import Callable, Union

import pandas as pd

from qlib.typehint import Literal
from ...data.data import LocalExpressionProvider
from ...data.dataset.handler import DataHandlerLP, DataHandler

DATA_KEY_TYPE = Literal["raw", "infer", "learn"]
DK_I: DATA_KEY_TYPE = "infer"


class Alpha191Handler(DataHandlerLP):

    def __init__(self,
                 instruments="csi500",
                 start_time=None,
                 end_time=None,
                 freq="day",
                 infer_processors=None,
                 learn_processors=None,
                 fit_start_time=None,
                 fit_end_time=None,
                 process_type=DataHandlerLP.PTYPE_A,
                 filter_pipe=None,
                 inst_processors=None,
                 **kwargs):
        infer_processors = []
        learn_processors = []

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

    def get_feature_config(self):
        return ["$open", "$close", "$high", "$low", "$volume"], ["$open", "$close", "$high", "$low", "$volume"]

    def get_label_config(self):
        return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]

    def extra_feature(self):
        return [
            # ("001", "-1 * Corr(Rank(Delta(Log($volume), 1)), Rank(($close - $open) / $open), 6)"),
            # ("002", "-1 * Delta(((($close - $low) - ($high - $close)) / ($high - $low)), 1)"),
            # ("005", "-1 * TSMax(Corr(TSRank($volume, 5), TSRank($high, 5), 5), 3)"),
            # ("007", "(Rank(Greater(($vwap - $close), 3)) + Rank(Less(($vwap - $close), 3))) * Rank(Delta($volume, 3))"),
            # ("008", "Rank(Delta((((($high + $low) / 2) * 0.2) + ($vwap * 0.8)), 4) * -1)"),
            # ("011", "Sum((($close-$low)-($high-$close))/($high-$low)*$volume / TSMax($volume, 20),6)"), # 011 我改了一点，否则volume量纲不一样,虽然效果也没变好
            # ("012", "Rank(($open - (Sum($vwap, 10) / 10))) * (-1 * (Rank(Abs(($close - $vwap)))))"),
            # ("013", "Power($high * $low,0.5) - $vwap"),
            # ("014", "$close-Ref($close,5)")  # 统一量纲在018有
            # ("015", "$open/Ref($close,1)-1")
            # ("016", "(-1 * TSMax(Rank(Corr(Rank($volume), Rank($vwap), 5)), 5))")
            # ("017", "Power(Rank(($vwap - Greater($vwap, 15))), Delta($close, 5))")  # 这个我实在看不懂……
            # ("018", "$close / Ref($close, 5)")
            # ("020", "$close / Ref($close, 6)") # 这个原公式是，(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100，没用，这个就等于018 - 1
        ]

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

        if df.columns.size == 1:
            return df

        instruments = set([])
        for index, row in df.iterrows():
            inst = index[1]
            instruments.add(inst)

        df['$vwap'] = (df["$high"] * 2 + df["$low"] * 2 + df["$close"] + df["$open"]) / 6
        output_df = pd.DataFrame()
        provider = LocalExpressionProvider()
        for (name, exp) in self.extra_feature():
            expression = provider.get_expression_instance(exp)
            series = expression.load_dataframe(instruments, df)
            output_df[name] = series

        out_columns = list(map(lambda x: ("feature", x[0]), self.extra_feature()))

        try:
            output_df["LABEL0"] = df["label"]["LABEL0"]
            out_columns += [("label", "LABEL0")]
        except:
            pass
        output_df.columns = pd.MultiIndex.from_tuples(out_columns)

        return output_df
