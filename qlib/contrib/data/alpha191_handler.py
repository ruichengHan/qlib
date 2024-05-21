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
            ("001", "-1 * Corr(Rank(Delta(Log($volume), 1)), Rank(($close - $open) / $open), 6)"),
            ("002", "-1 * Delta(((($close - $low) - ($high - $close)) / ($high - $low)), 1)")
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

        provider = LocalExpressionProvider()
        for (name, exp) in self.extra_feature():
            expression = provider.get_expression_instance(exp)
            series = expression.load_dataframe(self.instruments, df)
            df[name] = series

        return df
