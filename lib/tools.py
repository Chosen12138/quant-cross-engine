import os
import pandas as pd
import numpy as np
from typing import *
from quant_xpress import *

from conf.path_config import OPTIONS, FUTURES
from lib.utils import DataFrameActor

__all__ = ['load_option_day_bar_by_code',
           'load_future_day_data_by_symbol']


def load_option_day_bar_by_code(option_symbol, option_exchange, option_code):
    parent_path = os.path.join(
            OPTIONS.source_db_path,
            option_symbol,
            f"{option_code}.{option_exchange}")
    actor = DataFrameActor(parent_path=parent_path,
                           df=pd.DataFrame(),
                           filename='DAY')
    df = actor.from_bin()
    df.DATETIME = pd.to_datetime(df.DATETIME)
    df['DATE'] = df.DATETIME.apply(lambda r: r.strftime('%Y-%m-%d'))
    df["YESTERDAY_SETTLE"] = df.SETTLE.shift()
    df["YESTERDAY_CLOSE"] = df.CLOSE.shift()
    df = df.set_index("DATE")
    return df


def load_future_day_data_by_symbol(future_symbol, future_exchange):
    """期货指数日频数据, 提供日开高低收"""

    parent_path = os.path.join(
        FUTURES.source_db_path,
        future_symbol,
        f"{future_symbol}主连万得.{future_exchange}",
    )
    actor = DataFrameActor(df=pd.DataFrame(),
                           parent_path=parent_path,
                           filename='DAY')
    df = actor.from_bin()
    df["YESTERDAY_CLOSE"] = df.CLOSE.shift()
    df = df.set_index("DATE")
    return df

