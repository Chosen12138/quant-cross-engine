import numpy as np
import pandas as pd
from typing import *
from collections import namedtuple
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from dataclasses_json import dataclass_json
from conf import *

__all__ = [
    "OptionInfo",
    "OptionOrderTask",
    "FutureOrderTask",
    "FinishedOptionOrderTask",
    "FinishedFutureOrderTask",
    "UnderlyingInfo",
    "Bar",
    "Factor",
    "BarPro",
    "Order",
    "Task",
    "KPI",
    "TICK",
    "SubPosition",
    "MainPosition",
]

from conf import OptionTradeConfig

tick_fields = [
    "indexId",
    "tradingDay",
    "lastPrice",
    "preSettlement",
    "preClose",
    "preOpenInterest",
    "openPrice",
    "highPrice",
    "lowPrice",
    "volume",
    "turnover",
    "openInterest",
    "closePrice",
    "settlement",
    "limitUp",
    "limitDown",
    "preDelta",
    "currDelta",
    "updateTime",
    "updateMillisec",
    "bidPrice1",
    "bidVolume1",
    "askPrice1",
    "askVolume1",
    "bidPrice2",
    "bidVolume2",
    "askPrice2",
    "askVolume2",
    "bidPrice3",
    "bidVolume3",
    "askPrice3",
    "askVolume3",
    "bidPrice4",
    "bidVolume4",
    "askPrice4",
    "askVolume4",
    "bidPrice5",
    "bidVolume5",
    "askPrice5",
    "askVolume5",
    "avgPrice",
    "tickSource",
    "code",
    "exchange",
    "windCode",
    "dateTime",
    "tickVolume",
    "tickType",
]

TICK = namedtuple("TICK", field_names=tick_fields)


@dataclass_json
@dataclass
class OptionOrderTask:
    trade_date: datetime
    order_side: str
    symbol: str
    option_code: str
    info: Dict[str, type]


@dataclass_json
@dataclass
class FutureOrderTask:
    trade_date: datetime
    order_side: str
    symbol: str
    future_code: str
    info: Dict[str, type]


@dataclass_json
@dataclass
class FinishedOptionOrderTask:
    symbol: str
    option_code: str
    open_time: Union[datetime] = None
    cover_time: Union[datetime] = None
    open_price: Union[float] = None
    cover_price: Union[float] = None
    settle_price: Union[float] = None
    order_side: Union[str] = None
    deal_amount: Union[int] = 0
    fee: Union[float] = 0

    @property
    def profit(self):
        profit = 0
        fee = 0
        if self.cover_price is not None and self.open_price is not None:
            if self.order_side == 'cover_long':
                fee = OptionTradeConfig.fee_fixed_table[self.symbol] * self.deal_amount * 4
                profit = (self.cover_price - self.open_price) * self.deal_amount * OptionTradeConfig.multiplier_table[self.symbol]
            elif self.order_side == 'cover_short':
                fee = OptionTradeConfig.fee_fixed_table[self.symbol] * self.deal_amount * 4
                profit = (-self.cover_price + self.open_price) * self.deal_amount * OptionTradeConfig.multiplier_table[self.symbol]
        self.fee = fee
        return profit - fee


@dataclass_json
@dataclass
class FinishedFutureOrderTask:
    symbol: str
    future_code: str
    open_time: Union[datetime] = None
    cover_time: Union[datetime] = None
    open_price: Union[float] = None
    cover_price: Union[float] = None
    settle_price: Union[float] = None
    order_side: Union[str] = None
    deal_amount: Union[int] = 0
    fee: Union[float] = 0

    @property
    def profit(self):
        profit = 0
        fee = 0
        if self.cover_price is not None and self.open_price is not None:
            if self.symbol in FutureTradeConfig.fee_fixed_table.keys():
                fee = FutureTradeConfig.fee_fixed_table[self.symbol] * self.deal_amount * 4
            else:
                fee = FutureTradeConfig.fee_float_table[self.symbol] * self.deal_amount * self.cover_price * \
                      FutureTradeConfig.multiplier_table[self.symbol] * 4
            if self.order_side == 'cover_long':
                profit = (self.cover_price - self.open_price) * self.deal_amount * FutureTradeConfig.multiplier_table[self.symbol]
            elif self.order_side == 'cover_short':
                profit = (-self.cover_price + self.open_price) * self.deal_amount * FutureTradeConfig.multiplier_table[self.symbol]
        self.fee = fee
        return profit - fee

    @property
    def settle_profit(self):
        if np.isnan(self.settle_price) or self.settle_price is None:
            return 0
        if self.order_side == 'open_long':
            profit = (self.settle_price - self.open_price) * self.deal_amount * FutureTradeConfig.multiplier_table[self.symbol]
            return profit
        elif self.order_side == 'open_short':
            profit = (-self.settle_price + self.open_price) * self.deal_amount * FutureTradeConfig.multiplier_table[self.symbol]
            return profit
        else:
            return self.profit


@dataclass_json
@dataclass
class OptionInfo(object):
    """option information of one type each day"""

    wind_code: Union[str, float]
    bar_1min: np.ndarray
    bar_1min_df: pd.DataFrame
    bar_5min_df: pd.DataFrame
    bar_15min_df: pd.DataFrame

    def format_change(self):
        """??????matlab???????????????"""
        self.bar_1min_df["Datetime"] = pd.to_datetime(
            self.bar_1min_df.Timestamp - 719529, unit="D"
        )
        self.bar_1min_df["DatetimeRound"] = self.bar_1min_df.Datetime.dt.round(
            freq="min"
        )
        self.bar_1min_df["Code"] = self.wind_code


@dataclass_json
@dataclass
class UnderlyingInfo(object):
    """underlying information each day"""

    wind_code: Union[str, float]
    bar_1min: np.ndarray
    bar_1min_df: pd.DataFrame
    bar_5min_df: pd.DataFrame
    bar_15min_df: pd.DataFrame

    def format_change(self):
        self.bar_1min_df["Datetime"] = pd.to_datetime(
            self.bar_1min_df.Timestamp - 719529, unit="D"
        )
        self.bar_1min_df["DatetimeRound"] = self.bar_1min_df.Datetime.dt.round(
            freq="min"
        )
        self.bar_1min_df["Code"] = self.wind_code


@dataclass_json
@dataclass
class Bar(object):
    """Bar?????????????????????????????????"""

    CODE: str
    DATETIME: datetime
    OPEN: float
    HIGH: float
    LOW: float
    CLOSE: float
    VOLUME: float


@dataclass_json
@dataclass
class Factor(object):
    """???Bar???Factor?????????????????????????????????"""

    MA5: float
    MA10: float
    MA20: float


@dataclass_json
@dataclass
class BarPro(object):
    Bar_Min_1: Bar
    Bar_Min_5: Bar
    Bar_Min_15: Bar
    Bar_Min_30: Bar
    Bar_Min_60: Bar


@dataclass_json
@dataclass
class SubPosition(object):
    position_code: str
    holding_side: str
    holding_price: float
    holding_amount: int
    into_pos_time: datetime
    frozen_status: bool


@dataclass_json
@dataclass
class MainPosition(object):
    position_code: str
    holding_side: str
    holding_price: float
    holding_amount: int
    update_time: datetime
    sub_positions: List[SubPosition]


@dataclass_json
@dataclass
class Order(object):
    """Order?????????????????????"""

    #   ????????????, Virtual Code, IF????????????
    order_code: str
    #   ????????????, ??????(open_long, open_short, cover_long, cover_short)
    order_side: str
    #   ????????????
    order_amount: int
    #   ????????????
    order_price: float
    #   ????????????
    order_time: datetime
    #   ????????????, ?????????????????????, ?????????????????????????????????
    finished_time: Union[datetime, None] = None
    #   ????????????, ???????????????????????????
    effect_time: timedelta = timedelta(seconds=1)
    #   ????????????, Real Code, ????????????, ??????????????????order_code??????????????????
    real_code: str = ""
    #   ????????????
    asset_symbol: str = ""
    #   ???????????????
    asset_exchange: str = ""
    #   call??????put
    call_or_put: str = ""
    #   ??????ID, ??????????????????
    order_id: int = 0
    #   ????????????, ????????????
    deal_price: float = 0.0
    #   ????????????, ?????????????????????
    deal_amount: int = 0
    #   ??????????????????(??????finished, ?????????unfinished, ????????????killed, ????????????unknown)
    status: str = "unfinished"
    #   ??????????????????(FOK, ATK, STP)
    order_type: str = "FOK"
    #   ??????????????????(MKT, MID, LMT)
    price_type: str = "MKT"
    #   ???????????????, ??????????????????
    info: str = ""


@dataclass_json
@dataclass
class Task(object):
    """?????????????????????"""

    name: str
    send_time: datetime
    act_time: datetime
    timer: timedelta
    status: str
    obj: object
    reason: Optional[str] = None
    task_id: Optional[int] = None


@dataclass_json
@dataclass
class KPI(object):
    """key performance indicator"""

    clean_profit: float
    calmar: float
    max_draw_down: float
    trade_nums: float
    long_trade_nums: float
    short_trade_nums: float
    daily_win_rate: float
    daily_odds: float
    con_win_days: float
    con_lose_days: float
    long_profit: float
    short_profit: float
