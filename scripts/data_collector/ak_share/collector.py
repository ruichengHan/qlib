import abc
import datetime
import importlib
import multiprocessing
import sys
import time
from abc import ABC
from pathlib import Path

import akshare
import fire
import pandas as pd
import requests
from dateutil.tz import tzlocal
from loguru import logger

from qlib.constant import REG_CN as REGION_CN
from qlib.tests.data import GetData
from qlib.utils import exists_qlib_data

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))
import copy
import numpy as np
from dump_bin import DumpDataUpdate

from data_collector.base import BaseCollector
from data_collector.yahoo.collector import YahooNormalize1d, YahooNormalizeCN
from data_collector.base import BaseRun, Normalize
from data_collector.utils import (
    deco_retry,
)

INDEX_BENCH_URL = "http://push2his.eastmoney.com/api/qt/stock/kline/get?secid=1.{index_code}&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58&klt=101&fqt=0&beg={begin}&end={end}"


class AkNormalize1dExtend(YahooNormalize1d):
    COLUMNS = ["open", "close", "high", "low", "volume"]

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        # normalize
        df = self.normalize_ak(df, self._calendar_list, self._date_field_name, self._symbol_field_name)
        # adjusted price
        df = self.adjusted_price(df)
        return df

    def adjusted_price(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        df.set_index(self._date_field_name, inplace=True)
        df["factor"] = df["adjclose"] / df["close"]
        df["factor"] = df["factor"].fillna(method="ffill")
        return df.reset_index()

    @staticmethod
    def calc_change(df: pd.DataFrame) -> pd.Series:
        df = df.copy()
        _tmp_series = df["close"].fillna(method="ffill")
        _tmp_shift_series = _tmp_series.shift(1)
        _tmp_shift_series.iloc[0] = 0
        change_series = _tmp_series / _tmp_shift_series - 1
        return change_series

    @staticmethod
    def normalize_ak(
            df: pd.DataFrame,
            calendar_list: list = None,
            date_field_name: str = "date",
            symbol_field_name: str = "symbol",
            last_close: float = None,
    ):
        if df.empty:
            return df
        symbol = df.loc[df[symbol_field_name].first_valid_index(), symbol_field_name]
        columns = copy.deepcopy(AkNormalizeCN1dExtend.COLUMNS)
        df = df.copy()
        df.set_index(date_field_name, inplace=True)
        df.index = pd.to_datetime(df.index)
        df = df[~df.index.duplicated(keep="first")]
        if calendar_list is not None:
            df = df.reindex(
                pd.DataFrame(index=calendar_list)
                .loc[
                pd.Timestamp(df.index.min()).date(): pd.Timestamp(df.index.max()).date()
                                                     + pd.Timedelta(hours=23, minutes=59)
                ]
                .index
            )
        df.sort_index(inplace=True)
        df.loc[(df["volume"] <= 0) | np.isnan(df["volume"]), list(set(df.columns) - {symbol_field_name})] = np.nan

        df["change"] = AkNormalizeCN1dExtend.calc_change(df)

        columns += ["change"]
        df.loc[(df["volume"] <= 0) | np.isnan(df["volume"]), columns] = np.nan

        df[symbol_field_name] = symbol
        df.index.names = [date_field_name]
        return df.reset_index()

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        # normalize
        df = self.normalize_ak(
            df, self._calendar_list, self._date_field_name, self._symbol_field_name, last_close=None
        )
        # adjusted price
        df = self.adjusted_price(df)
        return df


class AkNormalizeCN1dExtend(YahooNormalizeCN, AkNormalize1dExtend):
    pass


class AKCollector(BaseCollector):
    retry = 5  # Configuration attribute.  How many times will it try to re-request the data if the network fails.

    def __init__(
            self,
            save_dir: [str, Path],
            start=None,
            end=None,
            interval="1d",
            max_workers=4,
            max_collector_count=2,
            delay=0,
            check_data_length: int = None,
            limit_nums: int = None,
    ):
        """

        Parameters
        ----------
        save_dir: str
            stock save dir
        max_workers: int
            workers, default 4
        max_collector_count: int
            default 2
        delay: float
            time.sleep(delay), default 0
        interval: str
            freq, value from [1min, 1d], default 1min
        start: str
            start datetime, default None
        end: str
            end datetime, default None
        check_data_length: int
            check data length, by default None
        limit_nums: int
            using for debug, by default None
        """
        super(AKCollector, self).__init__(
            save_dir=save_dir,
            start=start,
            end=end,
            interval=interval,
            max_workers=max_workers,
            max_collector_count=max_collector_count,
            delay=delay,
            check_data_length=check_data_length,
            limit_nums=limit_nums,
        )

        self.init_datetime()

    def init_datetime(self):
        if self.interval == self.INTERVAL_1min:
            self.start_datetime = max(self.start_datetime, self.DEFAULT_START_DATETIME_1MIN)
        elif self.interval == self.INTERVAL_1d:
            pass
        else:
            raise ValueError(f"interval error: {self.interval}")

        self.start_datetime = self.convert_datetime(self.start_datetime, self._timezone)
        self.end_datetime = self.convert_datetime(self.end_datetime, self._timezone)

    @staticmethod
    def convert_datetime(dt: [pd.Timestamp, datetime.date, str], timezone):
        try:
            dt = pd.Timestamp(dt, tz=timezone).timestamp()
            dt = pd.Timestamp(dt, tz=tzlocal(), unit="s")
        except ValueError as e:
            pass
        return dt

    @property
    @abc.abstractmethod
    def _timezone(self):
        raise NotImplementedError("rewrite get_timezone")

    @staticmethod
    def get_data_from_remote(symbol, interval, start, end, show_1min_logging: bool = False):
        try:
            start_date = start.strftime("%Y%m%d")
            end_date = end.strftime("%Y%m%d")
            df = akshare.stock_zh_a_hist(symbol=symbol[:6], start_date=start_date, end_date=end_date, period="daily",
                                         adjust="hfq")
            df["date"] = df["日期"]
            df["open"] = df["开盘"]
            df["high"] = df["最高"]
            df["low"] = df["最低"]
            df["volume"] = df["成交量"]
            df["adjclose"] = df["收盘"]
            df["symbol"] = symbol
            df = df[["date", "open", "high", "low", "volume", "adjclose", "symbol"]]

            origin_df = akshare.stock_zh_a_hist(symbol=symbol[:6], start_date=start_date, end_date=end_date,
                                                period="daily",
                                                adjust="")
            out_df = df.merge(origin_df, left_on='date', right_on='日期')
            out_df["close"] = out_df["收盘"]
            df = out_df[["date", "open", "close", "high", "low", "volume", "adjclose", "symbol"]]

            return df.reset_index()

        except Exception as e:
            import traceback
            traceback.print_exception(e)
            logger.warning(
                f"get data error: {symbol}--{start}--{end}"
                + "Your data request fails ak share! data"
            )

    def get_data(
            self, symbol: str, interval: str, start_datetime: pd.Timestamp, end_datetime: pd.Timestamp
    ) -> pd.DataFrame:
        @deco_retry(retry_sleep=self.delay, retry=self.retry)
        def _get_simple(start_, end_):
            self.sleep()
            _remote_interval = "1m" if interval == self.INTERVAL_1min else interval
            resp = self.get_data_from_remote(
                symbol,
                interval=_remote_interval,
                start=start_,
                end=end_,
            )
            if resp is None or resp.empty:
                raise ValueError(
                    f"get data error: {symbol}--{start_}--{end_}" + "The stock may be delisted, please check"
                )
            return resp

        _result = None
        if interval == self.INTERVAL_1d:
            try:
                _result = _get_simple(start_datetime, end_datetime)
            except ValueError as e:
                pass
        else:
            raise ValueError(f"cannot support {self.interval}")
        return pd.DataFrame() if _result is None else _result

    def collector_data(self):
        """collector data"""
        super(AKCollector, self).collector_data()
        self.download_index_data()

    @abc.abstractmethod
    def download_index_data(self):
        """download index data"""
        raise NotImplementedError("rewrite download_index_data")


class AKCollectorCN(AKCollector, ABC):
    def get_instrument_list(self):
        logger.info("get HS stock symbols......")
        # 只要沪深300的股票，别的先不要
        symbols = []

        index_list = ["000300", "000905"]
        for index in index_list:
            df = akshare.index_stock_cons_csindex(index)
            stock_list = sorted(list(df["成分券代码"]))
            symbols += list(map(lambda x: x + ".ss" if x[0] == '6' else x + ".sz", stock_list))

        logger.info(f"get {len(symbols)} symbols.")
        return symbols

    def normalize_symbol(self, symbol):
        symbol_s = symbol.split(".")
        symbol = f"sh{symbol_s[0]}" if symbol_s[-1] == "ss" else f"sz{symbol_s[0]}"
        return symbol

    @property
    def _timezone(self):
        return "Asia/Shanghai"


class AKCollectorCN1d(AKCollectorCN):
    def download_index_data(self):
        _format = "%Y%m%d"
        _begin = self.start_datetime.strftime(_format)
        _end = self.end_datetime.strftime(_format)
        for _index_name, _index_code in {"csi300": "000300", "csi100": "000903", "csi500": "000905"}.items():
            logger.info(f"get bench data: {_index_name}({_index_code})......")
            try:
                df = pd.DataFrame(
                    map(
                        lambda x: x.split(","),
                        requests.get(INDEX_BENCH_URL.format(index_code=_index_code, begin=_begin, end=_end)).json()[
                            "data"
                        ]["klines"],
                    )
                )
            except Exception as e:
                logger.warning(f"get {_index_name} error: {e}")
                continue
            df.columns = ["date", "open", "close", "high", "low", "volume", "money", "change"]
            df["date"] = pd.to_datetime(df["date"])
            df = df.astype(float, errors="ignore")
            df["adjclose"] = df["close"]
            df["symbol"] = f"sh{_index_code}"
            _path = self.save_dir.joinpath(f"sh{_index_code}.csv")
            if _path.exists():
                _old_df = pd.read_csv(_path)
                df = pd.concat([_old_df, df], sort=False)
            df.to_csv(_path, index=False)
            time.sleep(5)


class Run(BaseRun):
    def __init__(self, source_dir=None, normalize_dir=None, max_workers=1, interval="1d", region=REGION_CN):
        """

        Parameters
        ----------
        source_dir: str
            The directory where the raw data collected from the Internet is saved, default "Path(__file__).parent/source"
        normalize_dir: str
            Directory for normalize data, default "Path(__file__).parent/normalize"
        max_workers: int
            Concurrent number, default is 1; when collecting data, it is recommended that max_workers be set to 1
        interval: str
            freq, value from [1min, 1d], default 1d
        region: str
            region, value from ["CN", "US", "BR"], default "CN"
        """
        super().__init__(source_dir, normalize_dir, max_workers, interval)
        self.region = region

    @property
    def collector_class_name(self):
        return f"AKCollector{self.region.upper()}{self.interval}"

    @property
    def normalize_class_name(self):
        return f"AkNormalize{self.region.upper()}{self.interval}"

    @property
    def default_base_dir(self) -> [Path, str]:
        return CUR_DIR

    def download_data(
            self,
            max_collector_count=2,
            delay=0.5,
            start=None,
            end=None,
            check_data_length=None,
            limit_nums=None,
    ):
        """download data from Internet

        Parameters
        ----------
        max_collector_count: int
            default 2
        delay: float
            time.sleep(delay), default 0.5
        start: str
            start datetime, default "2000-01-01"; closed interval(including start)
        end: str
            end datetime, default ``pd.Timestamp(datetime.datetime.now() + pd.Timedelta(days=1))``; open interval(excluding end)
        check_data_length: int
            check data length, if not None and greater than 0, each symbol will be considered complete if its data length is greater than or equal to this value, otherwise it will be fetched again, the maximum number of fetches being (max_collector_count). By default None.
        limit_nums: int
            using for debug, by default None

        Notes
        -----
            check_data_length, example:
                daily, one year: 252 // 4
                us 1min, a week: 6.5 * 60 * 5
                cn 1min, a week: 4 * 60 * 5

        Examples
        ---------
            # get daily data
            $ python collector.py download_data --source_dir ~/.qlib/stock_data/source --region CN --start 2020-11-01 --end 2020-11-10 --delay 0.1 --interval 1d
            # get 1m data
            $ python collector.py download_data --source_dir ~/.qlib/stock_data/source --region CN --start 2020-11-01 --end 2020-11-10 --delay 0.1 --interval 1m
        """
        super(Run, self).download_data(max_collector_count, delay, start, end, check_data_length, limit_nums)

    def normalize_data(
            self,
            date_field_name: str = "date",
            symbol_field_name: str = "symbol",
            end_date: str = None,
            qlib_data_1d_dir: str = None,
    ):
        """normalize data

        Parameters
        ----------
        date_field_name: str
            date field name, default date
        symbol_field_name: str
            symbol field name, default symbol
        end_date: str
            if not None, normalize the last date saved (including end_date); if None, it will ignore this parameter; by default None
        qlib_data_1d_dir: str
            if interval==1min, qlib_data_1d_dir cannot be None, normalize 1min needs to use 1d data;

                qlib_data_1d can be obtained like this:
                    $ python scripts/get_data.py qlib_data --target_dir <qlib_data_1d_dir> --interval 1d
                    $ python scripts/data_collector/yahoo/collector.py update_data_to_bin --qlib_data_1d_dir <qlib_data_1d_dir> --trading_date 2021-06-01
                or:
                    download 1d data, reference: https://github.com/microsoft/qlib/tree/main/scripts/data_collector/yahoo#1d-from-yahoo

        Examples
        ---------
            $ python collector.py normalize_data --source_dir ~/.qlib/stock_data/source --normalize_dir ~/.qlib/stock_data/normalize --region cn --interval 1d
            $ python collector.py normalize_data --qlib_data_1d_dir ~/.qlib/qlib_data/cn_data --source_dir ~/.qlib/stock_data/source_cn_1min --normalize_dir ~/.qlib/stock_data/normalize_cn_1min --region CN --interval 1min
        """
        if self.interval.lower() == "1min":
            if qlib_data_1d_dir is None or not Path(qlib_data_1d_dir).expanduser().exists():
                raise ValueError(
                    "If normalize 1min, the qlib_data_1d_dir parameter must be set: --qlib_data_1d_dir <user qlib 1d data >, Reference: https://github.com/microsoft/qlib/tree/main/scripts/data_collector/yahoo#automatic-update-of-daily-frequency-datafrom-yahoo-finance"
                )
        super(Run, self).normalize_data(
            date_field_name, symbol_field_name, end_date=end_date, qlib_data_1d_dir=qlib_data_1d_dir
        )

    def normalize_data_1d_extend(
            self, date_field_name: str = "date", symbol_field_name: str = "symbol"
    ):
        """normalize data extend; extending yahoo qlib data(from: https://github.com/microsoft/qlib/tree/main/scripts#download-cn-data)

        Notes
        -----
            Steps to extend yahoo qlib data:

                1. download qlib data: https://github.com/microsoft/qlib/tree/main/scripts#download-cn-data; save to <dir1>

                2. collector source data: https://github.com/microsoft/qlib/tree/main/scripts/data_collector/yahoo#collector-data; save to <dir2>

                3. normalize new source data(from step 2): python scripts/data_collector/yahoo/collector.py normalize_data_1d_extend --old_qlib_dir <dir1> --source_dir <dir2> --normalize_dir <dir3> --region CN --interval 1d

                4. dump data: python scripts/dump_bin.py dump_update --csv_path <dir3> --qlib_dir <dir1> --freq day --date_field_name date --symbol_field_name symbol --exclude_fields symbol,date

                5. update instrument(eg. csi300): python python scripts/data_collector/cn_index/collector.py --index_name CSI300 --qlib_dir <dir1> --method parse_instruments

        Parameters
        ----------
        date_field_name: str
            date field name, default date
        symbol_field_name: str
            symbol field name, default symbol

        Examples
        ---------
            $ python collector.py normalize_data_1d_extend --old_qlib_dir ~/.qlib/qlib_data/cn_data --source_dir ~/.qlib/stock_data/source --normalize_dir ~/.qlib/stock_data/normalize --region CN --interval 1d
        """
        _class = getattr(self._cur_module, f"{self.normalize_class_name}Extend")
        yc = Normalize(
            source_dir=self.source_dir,
            target_dir=self.normalize_dir,
            normalize_class=_class,
            max_workers=self.max_workers,
            date_field_name=date_field_name,
            symbol_field_name=symbol_field_name
        )
        yc.normalize()

    def download_today_data(
            self,
            max_collector_count=2,
            delay=0.5,
            check_data_length=None,
            limit_nums=None,
    ):
        """download today data from Internet

        Parameters
        ----------
        max_collector_count: int
            default 2
        delay: float
            time.sleep(delay), default 0.5
        check_data_length: int
            check data length, if not None and greater than 0, each symbol will be considered complete if its data length is greater than or equal to this value, otherwise it will be fetched again, the maximum number of fetches being (max_collector_count). By default None.
        limit_nums: int
            using for debug, by default None

        Notes
        -----
            Download today's data:
                start_time = datetime.datetime.now().date(); closed interval(including start)
                end_time = pd.Timestamp(start_time + pd.Timedelta(days=1)).date(); open interval(excluding end)

            check_data_length, example:
                daily, one year: 252 // 4
                us 1min, a week: 6.5 * 60 * 5
                cn 1min, a week: 4 * 60 * 5

        Examples
        ---------
            # get daily data
            $ python collector.py download_today_data --source_dir ~/.qlib/stock_data/source --region CN --delay 0.1 --interval 1d
            # get 1m data
            $ python collector.py download_today_data --source_dir ~/.qlib/stock_data/source --region CN --delay 0.1 --interval 1m
        """
        start = datetime.datetime.now().date()
        end = pd.Timestamp(start + pd.Timedelta(days=1)).date()
        self.download_data(
            max_collector_count,
            delay,
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
            check_data_length,
            limit_nums,
        )

    def update_data_to_bin(
            self,
            qlib_data_1d_dir: str,
            trading_date: str = None,
            end_date: str = None,
            check_data_length: int = None,
            delay: float = 1,
    ):
        """update yahoo data to bin

        Parameters
        ----------
        qlib_data_1d_dir: str
            the qlib data to be updated for yahoo, usually from: https://github.com/microsoft/qlib/tree/main/scripts#download-cn-data

        trading_date: str
            trading days to be updated, by default ``datetime.datetime.now().strftime("%Y-%m-%d")``
        end_date: str
            end datetime, default ``pd.Timestamp(trading_date + pd.Timedelta(days=1))``; open interval(excluding end)
        check_data_length: int
            check data length, if not None and greater than 0, each symbol will be considered complete if its data length is greater than or equal to this value, otherwise it will be fetched again, the maximum number of fetches being (max_collector_count). By default None.
        delay: float
            time.sleep(delay), default 1
        Notes
        -----
            If the data in qlib_data_dir is incomplete, np.nan will be populated to trading_date for the previous trading day

        Examples
        -------
            $ python collector.py update_data_to_bin --qlib_data_1d_dir <user data dir> --trading_date <start date> --end_date <end date>
            # get 1m data
        """

        if self.interval.lower() != "1d":
            logger.warning(f"currently supports 1d data updates: --interval 1d")

        # start/end date
        if trading_date is None:
            trading_date = datetime.datetime.now().strftime("%Y-%m-%d")
            logger.warning(f"trading_date is None, use the current date: {trading_date}")

        if end_date is None:
            end_date = (pd.Timestamp(trading_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        # download qlib 1d data
        qlib_data_1d_dir = str(Path(qlib_data_1d_dir).expanduser().resolve())
        if not exists_qlib_data(qlib_data_1d_dir):
            GetData().qlib_data(target_dir=qlib_data_1d_dir, interval=self.interval, region=self.region)

        # download data from ak_share
        self.download_data(delay=delay, start=trading_date, end=end_date, check_data_length=check_data_length)
        # NOTE: a larger max_workers setting here would be faster
        self.max_workers = (
            max(multiprocessing.cpu_count() - 2, 1)
            if self.max_workers is None or self.max_workers <= 1
            else self.max_workers
        )
        # normalize data
        self.normalize_data_1d_extend(qlib_data_1d_dir)

        # dump bin
        _dump = DumpDataUpdate(
            csv_path=self.normalize_dir,
            qlib_dir=qlib_data_1d_dir,
            exclude_fields="symbol,date",
            max_workers=self.max_workers,
        )
        _dump.dump()

        # parse index
        _region = self.region.lower()
        if _region not in ["cn", "us"]:
            logger.warning(f"Unsupported region: region={_region}, component downloads will be ignored")
            return
        index_list = ["CSI100", "CSI300", "CSI500"] if _region == "cn" else ["SP500", "NASDAQ100", "DJIA", "SP400"]
        get_instruments = getattr(
            importlib.import_module(f"data_collector.{_region}_index.collector"), "get_instruments"
        )
        for _index in index_list:
            get_instruments(str(qlib_data_1d_dir), _index, market_index=f"{_region}_index")


if __name__ == "__main__":
    fire.Fire(Run)
