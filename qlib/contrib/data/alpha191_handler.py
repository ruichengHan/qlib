from typing import Callable, Union

import pandas as pd
import os
import time

from qlib.typehint import Literal
from ...data.data import LocalExpressionProvider
from ...data.dataset.handler import DataHandlerLP, DataHandler
from ...data.dataset.series_processor import MinMaxSProcessor, ZScoreSProcessor
from ...config import C

DATA_KEY_TYPE = Literal["raw", "infer", "learn"]
DK_I: DATA_KEY_TYPE = "infer"


def extra_feature():
    return [
        # ("001", "-1 * Corr(Rank(Delta(Log($volume), 1)), Rank(($close - $open) / $open), 6)"),
        # ("002", "-1 * Delta(((($close - $low) - ($high - $close)) / ($high - $low)), 1)"),
        # ("005", "-1 * TSMax(Corr(TSRank($volume, 5), TSRank($high, 5), 5), 3)"),
        # # 006, 我觉得这个有问题,他那个sign之后就都变成1/-1了, 那还Rank毛啊
        # ("007", "(Rank(Greater(($vwap - $close), 3)) + Rank(Less(($vwap - $close), 3))) * Rank(Delta($volume, 3))"),
        # ("008", "Rank(Delta((((($high + $low) / 2) * 0.2) + ($vwap * 0.8)), 4) * -1)"),
        # ("009", "Sma((($high+$low)/2-(Ref($high,1)+Ref($low,1))/2)*($high-$low)/Log($volume),7,2)"),
        # # 010里面的rank(, 5)感觉写错了
        # ("011", "Sum((($close-$low)-($high-$close))/($high-$low)*$volume / TSMax($volume, 20),6)"), # 011 我改了一点，否则volume量纲不一样,虽然效果也没变好
        # ("012", "Rank(($open - (Sum($vwap, 10) / 10))) * (-1 * (Rank(Abs(($close - $vwap)))))"),
        # ("013", "Power($high * $low,0.5) - $vwap"),
        # ("014", "$close-Ref($close,5)"),  # 统一量纲在018有
        # ("015", "$open/Ref($close,1)-1"),
        # ("016", "(-1 * TSMax(Rank(Corr(Rank($volume), Rank($vwap), 5)), 5))"),
        # ("017", "Power(Rank(($vwap - Greater($vwap, 15))), Delta($close, 5))"),  # 这个我实在看不懂……
        # ("018", "$close / Ref($close, 5)"),
        # ("019", "If(Delta($close,5) > 0,Delta($close, 5)/Ref($close,5),Delta($close, 5)/$close)"),
        # ("020", "$close / Ref($close, 6)"), # 这个原公式是，(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100，没用，这个就等于018 - 1
        # ("021", "Slope(Mean($close,6),6)"),
        # # 022 SMEAN 没有定义，实现不了
        # ("023", "Sma(If(Delta($close, 1) > 0, Std($close,20), 0),20,1)/(Sma(If(Delta($close, 1) > 0, Std($close,20),0),20,1)+Sma(If(Delta($close, 1) <=0,Std($close,20),0),20,1))"),
        # ("024", "Sma($close / Ref($close,5),5,1)"),  # 没用减法，换成除法
        # ("026", "Sum($close, 7) / 7 - $close + Corr($vwap, Ref($close, 5), 230)"),  # 尝试小调一下参数, 意义不大,还是小于0.01
        # ("027", "WMA(Delta($close, 3)/Ref($close,3)*100+Delta($close, 6)/Ref($close,6)*100,12)"),
        # ("029", "Delta($close, 6)/Ref($close,6)*$volume"),
        # ("031", "($close-Mean($close,12))/Mean($close,12)"),
        # ("032", "-1 * Sum(Rank(Corr(Rank($high), Rank($volume), 3)), 3)"),
        # ("034", "Mean($close,12)/$close"),
        # ("035", "Less(Rank(WMA(Delta($open, 1), 15)), Rank(WMA(Corr($volume, $open, 17),7))) * -1"),
        # ("036", "Rank(Sum(Corr(Rank($volume), Rank($vwap), 6), 2))"),  # 这个怀疑原文括号匹配错了
        # ("037", "-1 * Rank((Sum($open, 5) * $close/Ref($close, 5)) - Ref((Sum($open, 5) * $close/Ref($close, 5)), 10))"),
        # ("038", "If((Sum($high, 20) / 20 - $high < 0) , -1 * Delta($high, 2) , 0)"),
        # ("040", "Sum(If(Delta($close,1) > 0,$volume,0),26)/Sum(If(Delta($close,1) <=0,$volume,0),26)*100"),
        # ("041", "-1 * Rank(TSMax(Delta($vwap, 3), 5))"),  # 原文是Max,我感觉写的不对。。。
        # ("042", "-1 * Rank(Std($high, 10)) * Corr($high, $volume, 10)"),
        # ("043", "Sum(Sign($close - Ref($close, 1)) * $volume,6) / $volume"),  # 加了volume一个归一化
        # ("044", "TSRank(WMA(Corr($low, Mean($volume,10), 7), 6),4) + TSRank(WMA(Delta($vwap, 3), 10), 15)"),
        # ("045", "Rank(Delta($close * 0.6 + $open *0.4, 1)) * Rank(Corr($vwap, Mean($volume,20), 15))"),  # 原始求180天平均，太大了,改成20看看
        # ("046", "(Mean($close,3)+Mean($close,6)+Mean($close,12)+Mean($close,24))/$close/4"),
        # ("047", "Sma((TSMax($high,6)-$close)/(TSMax($high,6)-TSMin($low,6))*100,9,1)"),
        # ("048", "-1*Rank(Sign(Delta($close, 1)) + Sign(Ref($close, 1) - Ref($close, 2)) + Sign(Ref($close, 2) - Ref($close, 3))) * Sum($volume, 5) / Sum($volume, 20)"),
        # # 49,50,51都巨长
        # ("052", "Less(Sum(Greater(0.1,$high-Ref(($high+$low+$close)/3,1)),26)/Sum(Greater(0.1,Ref(($high+$low+$close)/3,1)-$low),26), 3)"), # 原文写错了，盲猜L=low
        # ("053", "Sum(If($close-Ref($close,1)>0,1,0),12)/12"),
        # ("054", "-1 * Rank((Std(Abs($close - $open), 10) / ($close - $open)) + Corr($close, $open,10))"), # 原文写错了，Std和Corr量纲都不一样，我给他统一了，把中间一个+换成了/
        # ("057", "Sma(($close-TSMin($low,9))/(TSMax($high,9)-TSMin($low,9))*100,3,1)"),
        # ("058", "Sum(If($close-Ref($close,1) > 0,1,0),20)/20"),
        # ("059", "Sum(If($close - Ref($close,1) == 0,0,$close-If($close - Ref($close,1) > 0, Less($low,Ref($close,1)), Greater($high,Ref($close,1)))),20)"),
        # ("060", "Sum((($close-$low)-($high-$close))/($high-$low)*$volume,20) / Ref($volume, 20)"), # 我给加了一个volume的归一化
        # ("061", "-1 * Greater(Rank(WMA(Delta($vwap, 1), 12)), Rank(WMA(Rank(Corr($low,Mean($volume,80), 8)), 17)))"),
        # ("062", "-1 * Corr($high, Rank($volume), 5)"),
        # ("063", "Sma(Greater($close-Ref($close,1),0),6,1)/Sma(Abs($close-Ref($close,1)),6,1)"),
        # ("065", "Mean($close,6)/$close"),
        # ("066", "($close-Mean($close,6))/Mean($close,6)*100"),
        # ("067", "Sma(Greater(Delta($close,1),0),24,1)/Sma(Abs(Delta($close,1)),24,1)"),
        # ("068", "Sma((($high+$low)/2-(Ref($high,1)+Ref($low,1))/2)*($high-$low)/$volume,15,2)"),
        # ("071", "($close-Mean($close,24))/Mean($close,24)"),
        # ("072", "Sma((TSMax($high,6)-$close)/(TSMax($high,6)-TSMin($low,6)),15,1)"),
        # ("073", "-1 * (TSRank(WMA(WMA(Corr($close, $volume, 10), 16), 4),5) - Rank(WMA(Corr($vwap, Mean($volume,30), 4),3)))"),
        # ("074", "Rank(Corr(Sum($low * 0.35 + $vwap * 0.65, 20), Sum(Mean($volume,40), 20), 7)) + Rank(Corr(Rank($vwap), Rank($volume), 6))"),
        # # 075 有一个基准指数BANCHMARKINDEXCLOSE, 这个得想想怎么写
        # ("076", "Std(Abs(($close/Ref($close,1)-1))/$volume,20)/Mean(Abs(($close/Ref($close,1)-1))/$volume,20)"),
        # ("077", "Less(Rank(WMA((($high + $low) / 2) - $vwap, 20)), Rank(WMA(Corr((($high + $low) / 2), Mean($volume,40), 3), 6)))"), # 他原来的感觉写重了，high加完了又给减掉了
        # ("078", "(($high+$low+$close)/3-Mean(($high+$low+$close)/3,12))/(0.015*Mean(Abs($close-Mean(($high+$low+$close)/3,12)),12))"), #它里面有个MA，我改成了Mean,定义里面没有MA
        # ("079", "SMA(Greater(Delta($close,1),0),12,1)/SMA(ABS(Delta($close,1)),12,1)*100"),
        # ("080", "Delta($volume,5)/Ref($volume,5)"),
        # ("081", "Sma($volume, 21, 1)"),
        # ("082", "SMA((TSMAX($high,6)-$close)/(TSMAX($high,6)-TSMIN($low,6))*100,20,1)"),
        # ("083", "-1 * Rank(Cov(Rank($high), Rank($volume), 5))"),
        # ("084", "SUM(Sign(Delta($close, 1)) * $volume,20) / Ref($volume, 20)"), # 加了一个归一化
        # ("085", "TSRANK(($volume / MEAN($volume,20)), 20) * TSRANK((-1 * DELTA($close, 7)), 8)"),
        # ("086", "If((Ref($close, 20) - Ref($close, 10)) / 10 - (Ref($close, 10) - $close) / 10 - 0.25 > 0 ,-1,If((Ref($close, 20) - Ref($close, 10)) / 10 - (Ref($close, 10) - $close) / 10 < 0 , 1 , -1 * Delta($close , 1)))"),
        # ("087", "-1 * (RANK(WMA(DELTA($vwap, 4), 7)) + TSRANK(WMA(($low * 0.9 + $low * 0.1 - $vwap) / ($open - ($high + $low) / 2), 11), 7))"),
        # ("088", "Delta($close, 20)/Ref($close,20)"),
        # ("089", "SMA($close,13,2)-SMA($close,27,2)-SMA(SMA($close,13,2)-SMA($close,27,2),10,2)"),
        # ("090", "-1 * RANK(CORR(RANK($vwap), RANK($volume), 5))"),
        # ("091", "-1 *RANK(($close - TSMAX($close, 5)))*RANK(CORR((MEAN($volume,40)), $low, 5))"), # 它里面的MAX我换成了TSMAX,我觉得他写错了
        # ("092", "-1 * Greater(RANK(WMA(DELTA((($close * 0.35) + ($vwap *0.65)), 2), 3)),TSRANK(WMA(ABS(CORR((MEAN($volume,30)), $close, 13)), 5), 15))"),
        # ("093", "SUM(If(Delta($open, 1) >=0,0,Greater($open-$low,Delta($open,1))),20)"),
        # ("094", "Rank(SUM(Sign(Delta($close, 1)) * $volume,30))"),
        # 095有amount， 下次把amount加进来
        # ("096","SMA(SMA(($close-TSMIN($low,9))/(TSMAX($high,9)-TSMIN($low,9))*100,3,1),3,1)"),
        # ("097", "STD($volume,10)"),
        # ("099", "-1 * RANK(Cov(RANK($close), RANK($volume), 5))"),
        # ("100", "STD($volume,20)"),
        # ("101", "-1 * (RANK(CORR($close, SUM(MEAN($volume,30), 37), 15)) - RANK(CORR(RANK($high * 0.1 + $vwap * 0.9), RANK($volume), 11)))"),
        # ("102", "SMA(Greater(Delta($volume, 1),0),6,1)/SMA(ABS(Delta($volume, 1)),6,1)*100"),
        # ("116", "Slope($close,20)"),
        ("147", "Slope(Mean($close,12),12)")
    ]


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
        self.base_path = C.get("provider_uri")["__DEFAULT_FREQ"]+ "/ht191"

        self.norm_map = {
            "067": [ZScoreSProcessor(fit_start_time, fit_end_time)],
            "084": [MinMaxSProcessor(-25, 25)],
            "088": [MinMaxSProcessor(-1, 1)],
            "116": [MinMaxSProcessor(-5, 5)],
            "147": [MinMaxSProcessor(-5, 5)]
        }

    def get_feature_config(self):
        return ["$open", "$close", "$high", "$low", "$volume"], ["$open", "$close", "$high", "$low", "$volume"]

    def get_label_config(self):
        return (["Ref($close, -2)/Ref($close, -1) - 1", "Ref($close, -3)/Ref($close, -1) - 1",
                 "Ref($close, -4)/Ref($close, -1) - 1", "Ref($close, -5)/Ref($close, -1) - 1",
                 "Ref($close, -6)/Ref($close, -1) - 1"],
                ["LABEL0", "LABEL1", "LABEL2", "LABEL3", "LABEL4"])

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

        if col_set == 'label':
            return df

        instruments = df.index.levels[1].tolist()

        df['$vwap'] = (df["$high"] * 2 + df["$low"] * 2 + df["$close"] + df["$open"]) / 6
        output_df = pd.DataFrame()
        provider = LocalExpressionProvider()
        for (name, exp) in extra_feature():
            print("start process ", name)
            series = self.load_series(name)
            if series is None:
                expression = provider.get_expression_instance(exp)
                series = expression.load_dataframe(instruments, df)
                series.sort_index(inplace=True)
                processers = self.norm_map.get(name, [])
                for p in processers:
                    print(name, p)
                    series = p.fit(series)
                self.dump_series(name, series)
            output_df[name] = series

        out_columns = list(map(lambda x: ("feature", x[0]), extra_feature()))

        output_df.columns = pd.MultiIndex.from_tuples(out_columns)

        return output_df

    def get_feature_path(self, name):
        return self.base_path + "/" + name + '.bin'

    def dump_series(self, name, series: pd.Series):
        series.to_pickle(self.get_feature_path(name))

    def load_series(self, name):
        path = self.get_feature_path(name)
        if os.path.exists(path):
            return pd.read_pickle(path)
