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
        """处理matlab基准时间戳"""
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
    """Bar驱动回测需要的数据结构"""

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
    """随Bar的Factor驱动回测需要的数据结构"""

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
    """Order需要的数据结构"""

    #   标的代码, Virtual Code, IF主力连续
    order_code: str
    #   交易方向, 包括(open_long, open_short, cover_long, cover_short)
    order_side: str
    #   报单数量
    order_amount: int
    #   报单价格
    order_price: float
    #   报单时间
    order_time: datetime
    #   完成时间, 仅在成交下记录, 其余未成交订单均不记录
    finished_time: Union[datetime, None] = None
    #   有效时间, 对特定订单类型有效
    effect_time: timedelta = timedelta(seconds=1)
    #   交易代码, Real Code, 可以为空, 为空状态按照order_code索引交易标的
    real_code: str = ""
    #   品种代码
    asset_symbol: str = ""
    #   品种交易所
    asset_exchange: str = ""
    #   call或者put
    call_or_put: str = ""
    #   订单ID, 内部虚拟维护
    order_id: int = 0
    #   成交价格, 成交均价
    deal_price: float = 0.0
    #   成交数量, 不适用部分成交
    deal_amount: int = 0
    #   订单状态包括(完成finished, 未完成unfinished, 自动失效killed, 未知状态unknown)
    status: str = "unfinished"
    #   订单类型包括(FOK, ATK, STP)
    order_type: str = "FOK"
    #   价格类型包括(MKT, MID, LMT)
    price_type: str = "MKT"
    #   订单信息流, 记录发单目的
    info: str = ""


@dataclass_json
@dataclass
class Task(object):
    """策略产生的任务"""

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
