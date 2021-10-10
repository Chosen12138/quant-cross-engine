import os
import re
import msgpack
import pandas as pd
import numpy as np
from snappy import snappy
from bisect import bisect_left
from datetime import datetime
from typing import *

from quant_xpress import *
from conf import *

__all__ = [
    "COLUMN_MAPPING_EN_TO_CN",
    "COLUMN_MAPPING_CN_TO_EN",
    "DataFrameActor",
    "make_log_path",
    "datetime_to_unix",
    "unix_to_datetime",
    "search_closest",
    "get_symbol_prefix",
    "search_pivot_arrays",
    "compress_df_to_bin",
    "calculate_factors_using_graph",
    "load_df_from_bin",
    "load_payload_from_bin"
]

COLUMN_MAPPING_EN_TO_CN = {
    "TradingDay": "交易日",
    "InstrumentID": "合约代码",
    "ExchangeID": "交易所代码",
    "ExchangeInstID": "合约在交易所的代码",
    "LastPrice": "最新价",
    "PreSettlementPrice": "上次结算价",
    "PreClosePrice": "昨收盘",
    "PreOpenInterest": "昨持仓量",
    "OpenPrice": "今开盘",
    "HighPrice": "最高价",
    "LowPrice": "最低价",
    "Volume": "数量",
    "Turnover": "成交金额",
    "OpenInterest": "持仓量",
    "ClosePrice": "今收盘",
    "SettlementPrice": "本次结算价",
    "UpperLimitPrice": "涨停板价",
    "LowerLimitPrice": "跌停板价",
    "PreDelta": "昨虚实度",
    "CurrDelta": "今虚实度",
    "UpdateTime": "最后修改时间",
    "UpdateMillisec": "最后修改毫秒",
    "BidPrice1": "申买价一",
    "BidVolume1": "申买量一",
    "AskPrice1": "申卖价一",
    "AskVolume1": "申卖量一",
    "BidPrice2": "申买价二",
    "BidVolume2": "申买量二",
    "AskPrice2": "申卖价二",
    "AskVolume2": "申卖量二",
    "BidPrice3": "申买价三",
    "BidVolume3": "申买量三",
    "AskPrice3": "申卖价三",
    "AskVolume3": "申卖量三",
    "BidPrice4": "申买价四",
    "BidVolume4": "申买量四",
    "AskPrice4": "申卖价四",
    "AskVolume4": "申卖量四",
    "BidPrice5": "申买价五",
    "BidVolume5": "申买量五",
    "AskPrice5": "申卖价五",
    "AskVolume5": "申卖量五",
    "AveragePrice": "当日均价",
    "ActionDay": "业务日期",
}

COLUMN_MAPPING_CN_TO_EN = dict(
    [val, key] for key, val in COLUMN_MAPPING_EN_TO_CN.items()
)


def make_log_path():
    """创建log文件"""
    if not os.path.exists(LOG.log_path):
        os.mkdir(LOG.log_path)
    if not os.path.exists(LOG.today_log_path):
        os.mkdir(LOG.today_log_path)
    if not os.path.exists(os.path.join(LOG.today_log_path, "Strategy")):
        os.mkdir(os.path.join(LOG.today_log_path, "Strategy"))
    if not os.path.exists(os.path.join(LOG.today_log_path, "Engine")):
        os.mkdir(os.path.join(LOG.today_log_path, "Engine"))
    return


def unix_to_datetime(unix: Union[int, str]):
    if isinstance(unix, int):
        unix = str(unix)
    if isinstance(unix, str):
        if len(unix) > 10:
            unix = unix[0:10]
        unix = int(unix)
    _time = datetime.fromtimestamp(unix)
    return _time


def datetime_to_unix(_time: Union[datetime], length: int = 13):
    import time

    _unix = time.mktime(_time.timetuple())
    if length == 13:
        return round(_unix * 1000)
    else:
        return _unix


def search_closest(myList, myNumber, return_pos: bool = True):
    """
    Assumes myList is sorted. Returns closest value to myNumber.
    If two numbers are equally close, return the smallest number.
    If number is outside of min or max return False
    """
    if myNumber > myList[-1] or myNumber < myList[0]:
        return False
    pos = bisect_left(myList, myNumber)
    if return_pos:
        return pos
    else:
        if pos == 0:
            return myList[0]
        if pos == len(myList):
            return myList[-1]
        before = myList[pos - 1]
        after = myList[pos]
        if after - myNumber < myNumber - before:
            return after
        else:
            return before


def search_pivot_arrays(df, left_checks: int = 10, right_delays: int = 10):
    """
    from close dataframe->array, search high and low pivot from left_checks and right_delays
    """
    array = np.array(df).flatten()

    high_pivot_array = np.full(len(df), np.nan)
    low_pivot_array = np.full(len(df), np.nan)

    possible_high_pivot_point = None
    possible_low_pivot_point = None

    for i in range(array.shape[0]):

        if i < left_checks:
            continue

        elif i >= left_checks:
            close = array[i]
            last_close = array[i - 1]

            if last_close < close and all(array[i - left_checks : i - 2] > last_close):
                if (
                    possible_low_pivot_point is None
                    or possible_low_pivot_point[1] > last_close
                ):
                    possible_low_pivot_point = (i - 1, last_close)

            if last_close > close and all(array[i - left_checks : i - 2] < last_close):
                if (
                    possible_high_pivot_point is None
                    or possible_high_pivot_point[1] < last_close
                ):
                    possible_high_pivot_point = (i - 1, last_close)

            if possible_low_pivot_point is not None:
                if i - possible_low_pivot_point[0] < right_delays:
                    if close > possible_low_pivot_point[1]:
                        pass
                    else:
                        possible_low_pivot_point = None

                elif i - possible_low_pivot_point[0] == right_delays:
                    low_pivot_array[
                        possible_low_pivot_point[0]
                    ] = possible_low_pivot_point[1]
                    possible_low_pivot_point = None

            if possible_high_pivot_point is not None:
                if i - possible_high_pivot_point[0] < right_delays:
                    if close < possible_high_pivot_point[1]:
                        pass
                    else:
                        possible_high_pivot_point = None

                elif i - possible_high_pivot_point[0] == right_delays:
                    high_pivot_array[
                        possible_high_pivot_point[0]
                    ] = possible_high_pivot_point[1]
                    possible_high_pivot_point = None

    df = pd.DataFrame(data=high_pivot_array, index=df.index)
    df.columns = ["HIGH_PIVOT"]
    df["LOW_PIVOT"] = low_pivot_array
    df = df.fillna(method="ffill")
    return df


def get_symbol_prefix(s):
    """
    匹配字符串取出ＩＦ、ＩＣ等
    """
    SYMBOL_PREFIX_PATTERN = re.compile(r"^([A-Za-z]+)")
    if re.search("ZL", s) is not None:
        s = s[0 : re.search("ZL", "IFZL").span(0)[0]]
    return SYMBOL_PREFIX_PATTERN.match(s).group(1)


class DataFrameActor(object):
    def __init__(self, df, parent_path, filename):
        self.df = df
        self.parent_path = parent_path
        self.filename = filename

    def to_csv(self, index: bool = False, mode: str = "w"):
        self.df.to_csv(
            path_or_buf=os.path.join(self.parent_path, f"{self.filename}.csv"),
            index=index,
            mode=mode,
        )

    def to_hdf(self, key, format_: str = "table"):
        self.df.to_hdf(
            path_or_buf=os.path.join(self.parent_path, f"{self.filename}.hdf"),
            key=key,
            format=format_,
        )

    def to_bin(self):
        if "DATE" in self.df.columns:
            self.df.DATE = self.df.DATE.astype(str)
        if "DATETIME" in self.df.columns:
            self.df.DATETIME = self.df.DATETIME.astype(str)
        compress_df_to_bin(
            data=self.df,
            path_or_buf=os.path.join(self.parent_path, f"{self.filename}.bin"),
        )

    def from_bin(self):
        return load_df_from_bin(
            path_or_buf=os.path.join(self.parent_path, f"{self.filename}.bin")
        )

    def from_csv(self, parse_date: str = "DATE", low_memory: bool = False):
        return pd.read_csv(
            os.path.join(self.parent_path, f"{self.filename}.csv"),
            parse_dates=[parse_date],
            low_memory=low_memory,
        )

    def from_hdf(self, key: str = "Bar"):
        return pd.read_hdf(
            path_or_buf=os.path.join(self.parent_path, f"{self.filename}.hdf"), key=key
        )


def compress_df_to_bin(data: pd.DataFrame, path_or_buf: str):
    """
    compress df to bin
    :param data: original dataframe
    :param path_or_buf: file path
    :return:
    """
    #   创建output格式，fields和items
    output = {}
    fields = data.columns.tolist()
    items = []

    fields.insert(0, "indexId")

    for row_ in data.itertuples():
        items.append(list(row_))

    output["fields"] = fields
    output["items"] = items

    #   压缩和标记bytes
    output_bytes = b"\x7f\x91\x23\x01" + msgpack.dumps(output)

    output_compress = snappy.compress(output_bytes)
    with open(path_or_buf, "wb") as f:
        f.write(output_compress)

    return


def load_payload_from_bin(path_or_buf):
    """载入并解压缩bin文件，file_name格式务必一致"""
    msgpack_loads = msgpack.loads
    with open(path_or_buf, "rb") as f:
        cnt = snappy.uncompress(f.read())
        if len(cnt) < 4 or cnt[:3] != b"\x7f\x91\x23":
            raise IOError(f"Malformed file: {path_or_buf}")

        f = msgpack_loads
        payload = f(cnt[4:])

    return payload


def load_df_from_bin(path_or_buf):
    """载入并解压缩bin文件，file_name格式务必一致"""
    msgpack_loads = msgpack.loads
    with open(path_or_buf, "rb") as f:
        cnt = snappy.uncompress(f.read())
        if len(cnt) < 4 or cnt[:3] != b"\x7f\x91\x23":
            raise IOError(f"Malformed file: {path_or_buf}")

        f = msgpack_loads
        payload = f(cnt[4:])
    df = pd.DataFrame(payload["items"])
    df.columns = payload["fields"]
    if "indexId" in df.columns:
        df = df.drop(columns=["indexId"])
    return df


def calculate_factors_using_graph(df_bar: pd.DataFrame, factor_graph: Union[object], merge_org: bool = False):
    """

    :param df_bar: 用于计算因子的K线df
    :param factor_graph: 因子计算图
    :param merge_org: 是否拼接input
    :return: bar与factor合并df
    """

    if len(df_bar) < 250:
        raise NotImplementedError("data input ")
    df_bar.columns = [col.upper() for col in df_bar.columns]

    if "DATETIME" in df_bar.columns and "DATETIMEROUND" in df_bar.columns:
        df_bar = df_bar.drop(columns=["DATETIME"])

    column_mapper = {"DATETIMEROUND": "DATETIME", "WIND_CODE": "CODE", "SYMBOL": "CODE"}

    df_bar = df_bar.rename(columns=column_mapper)
    df_bar = df_bar[['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'DATETIME', 'CODE']].copy()
    factors_df = eval_with_df(
        graph_or_tensors=factor_graph,
        code_col="CODE",
        time_col="DATETIME",
        df=df_bar,
        show_progress=True,
    )
    if merge_org:
        factors_df = factors_df.merge(df_bar,
                                      left_on=['DATETIME', 'CODE'],
                                      right_on=['DATETIME', 'CODE'],
                                      how='inner')
    return factors_df


if __name__ == "__main__":
    parent_path = os.path.join(FUTURES.factor_db_path, "IF", "IF指数.CFFEX")
    actor = DataFrameActor(
        parent_path=parent_path, filename="HOU_01", df=pd.DataFrame()
    )
    df = actor.from_bin()
    print(df)
