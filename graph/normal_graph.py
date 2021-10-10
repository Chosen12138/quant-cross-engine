from quant_xpress import *
from quant_xpress.ops import *
from quant_xpress.ops.ts_ops import *
from quant_xpress.ops.tech_ops import *

__all__ = ['FACTOR_JAY', 'FACTOR_YAF', 'FACTOR_CROSS']


class FACTOR_YAF(GraphDef):

    [OPEN, HIGH, LOW, CLOSE, VOLUME] = features('OPEN HIGH LOW CLOSE VOLUME')

    MA60 = ts_mean(CLOSE, d=60)
    MA90 = ts_mean(CLOSE, d=90)
    SPEED60 = (MA60 - ts_delay(MA60, d=7))/MA60 * 1000
    SPEED90 = (MA90 - ts_delay(MA90, d=7))/MA90 * 1000

    LAST_CLOSE = ts_delay(CLOSE, d=1)
    LAST_OPEN = ts_delay(OPEN, d=1)

    S_IND = ts_sum(where(CLOSE > OPEN, CLOSE - OPEN, 0), d=15) / ts_sum(abs(CLOSE - OPEN), d=15)


class FACTOR_CROSS(GraphDef):
    [OPEN, HIGH, LOW, CLOSE, VOLUME] = features('OPEN HIGH LOW CLOSE VOLUME')

    _MEAN_30 = ts_mean(CLOSE, d=30)
    _MEAN_60 = ts_mean(CLOSE, d=60)
    #   SPEED30 = (_MEAN_30 - ts_delay(_MEAN_30, d=6)) / _MEAN_30
    SLOPE240 = ts_detrend_slope(CLOSE, d=240)
    DETREND30 = ts_detrend(ts_mean(CLOSE, d=30), d=240) / _MEAN_30
    SPEED30 = (_MEAN_30 - ts_delay(_MEAN_30, d=6)) / _MEAN_30
    SPEED60 = (_MEAN_60 - ts_delay(_MEAN_60, d=6)) / _MEAN_60
    STD30 = ts_std(CLOSE/ts_delay(CLOSE, d=1), d=30)
    ATR30 = ATR(CLOSE, HIGH, LOW, d=30)
    STD30_TO_120 = ts_std(CLOSE/ts_delay(CLOSE, d=1), d=30) / ts_std(CLOSE/ts_delay(CLOSE, d=1), d=120)
    ATR30_TO_120 = ATR(CLOSE, HIGH, LOW, d=30) / ATR(CLOSE, HIGH, LOW, d=120)
    RANK_STD_30_TO_120 = rank(STD30_TO_120)
    RANK_STD_30 = rank(STD30)
    SCORE = rank(-SLOPE240) + rank(STD30_TO_120) + rank(SPEED30)
    # SCORE = rank(STD30_TO_120) + rank(SPEED60) + rank(-SLOPE240)


class FACTOR_JAY(GraphDef):

    [OPEN, HIGH, LOW, CLOSE, VOLUME] = features('OPEN HIGH LOW CLOSE VOLUME')

    TYP = (HIGH + LOW + CLOSE) / 3
    LAST_TYP = ts_delay(TYP, d=1)
    LAST_2_TYP = ts_delay(TYP, d=2)
    LAST_3_TYP = ts_delay(TYP, d=3)

    LAST_LOW = ts_delay(LOW, d=1)
    LAST_2_LOW = ts_delay(LOW, d=2)
    LAST_3_LOW = ts_delay(LOW, d=3)

    LAST_CLOSE = ts_delay(CLOSE, d=1)
    LAST_2_CLOSE = ts_delay(CLOSE, d=2)
    LAST_3_CLOSE = ts_delay(CLOSE, d=3)

    LAST_OPEN = ts_delay(OPEN, d=1)
    LAST_2_OPEN = ts_delay(OPEN, d=2)
    LAST_3_OPEN = ts_delay(OPEN, d=3)

    LAST_HIGH = ts_delay(HIGH, d=1)
    LAST_2_HIGH = ts_delay(HIGH, d=2)
    LAST_3_HIGH = ts_delay(HIGH, d=3)

    LAST_VOLUME = ts_delay(VOLUME, d=1)
    LAST_2_VOLUME = ts_delay(VOLUME, d=2)

    MA10 = ts_mean(CLOSE, d=10)
    LAST_MA10 = ts_delay(MA10, d=1)
    LAST_2_MA10 = ts_delay(MA10, d=2)
    LAST_3_MA10 = ts_delay(MA10, d=3)

    CON_UP_MA10 = ts_sum(CLOSE > MA10, d=4)
    CON_DOWN_MA10 = ts_sum(CLOSE < MA10, d=4)
